import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def load_route_schedule(path):
    """
    Load and return the route schedule DataFrame.
    """
    return pd.read_csv(path, index_col=0)

def load_forecasts(path):
    """
    Load and return the forecasts DataFrame.
    """
    return pd.read_csv(path, index_col=0)

def extract_route_data(route_schedule):
    """
    Extract relevant arrays from route_schedule DataFrame.
    """
    eco_energy = route_schedule.iloc[0, :].dropna().values
    eco_time = route_schedule.iloc[1, :].dropna().values
    stop_times = route_schedule.iloc[4, :].dropna().values
    can_charge = route_schedule.iloc[5, :].dropna().astype(bool).values
    return eco_energy, eco_time, stop_times, can_charge

def extract_grid_data(forecasts):
    """
    Extract electricity price and carbon intensity, converting units as needed.
    """
    electricity_price = forecasts['price'].values / 1000  # $/kWh
    carbon_intensity = forecasts['emissions'].values / 1000  # kgCO2/kWh
    return electricity_price, carbon_intensity

def interpolate_carbon_intensity(time_index, carbon_intensity, new_time_index):
    """
    Linearly interpolate carbon intensity to match the desired time resolution.
    """
    interp_func = interp1d(time_index, carbon_intensity, kind='linear', fill_value='extrapolate')
    return interp_func(new_time_index)

def prepare_time_series_data(forecasts, interval=1, total_minutes=1440):
    """
    Prepare 1-minute interval arrays for time, price, and carbon intensity.
    """
    # Original time grid (e.g., every 5 minutes)
    orig_time_index = np.arange(0, len(forecasts) * 5, 5)
    # Add endpoint if necessary
    orig_time_index = np.append(orig_time_index, total_minutes)
    electricity_price = forecasts['price'].values / 1000
    carbon_intensity = forecasts['emissions'].values / 1000
    electricity_price = np.append(electricity_price, electricity_price[-1])
    carbon_intensity = np.append(carbon_intensity, carbon_intensity[-1])

    # Interpolate
    interp_func = interp1d(orig_time_index, carbon_intensity, kind='linear', fill_value='extrapolate')
    time_index = np.arange(0, total_minutes + 1, interval)
    # Repeat price to 1-minute resolution
    price_1min = np.repeat(electricity_price[:-1], 5)
    price_1min = np.append(price_1min, electricity_price[-1])
    # Interpolated carbon
    carbon_1min = interp_func(time_index)
    return time_index, price_1min, carbon_1min

def save_time_series(time_index, electricity_price, carbon_intensity, outdir="."):
    """
    Save time series arrays to .txt files.
    """
    np.savetxt(f"{outdir}/time_index.txt", time_index, fmt='%d')
    np.savetxt(f"{outdir}/electricity_price.txt", electricity_price, fmt='%.8f')
    np.savetxt(f"{outdir}/carbon_intensity.txt", carbon_intensity, fmt='%.8f')




import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def prepare_milp_inputs(day_file, route_file, data_dir="./data"):
    """
    Prepares all the required inputs for the MILP function, given a day file and a route file.
    Returns: (lambd, electricity_price, carbon_intensity, time_index, 
              start_time_nominal, start_time_lower_bound, start_time_upper_bound, charge_windows)
    """
    # Paths
    day_path = os.path.join(data_dir, day_file)
    route_path = os.path.join(data_dir, route_file)
    
    # Load data
    forecasts = pd.read_csv(day_path, index_col=0)
    route_schedule = pd.read_csv(route_path, index_col=0)
    
    # Parameters (change as needed or make them arguments)
    start_time_nominal = 2.5 * 60  # 2:30 AM in minutes
    min_start_shift = 2 * 60
    max_start_shift = 2 * 60
    battery_capacity = 256
    min_soc = 0.10 * battery_capacity
    max_soc = 0.90 * battery_capacity
    initial_soc = max_soc
    delta_t = 1
    
    eco_energy = route_schedule.iloc[0, :].dropna().values
    eco_time = route_schedule.iloc[1, :].dropna().values
    stop_times = route_schedule.iloc[4, :].dropna().values
    can_charge = route_schedule.iloc[5, :].dropna().astype(bool).values

    # Grid data
    electricity_price = forecasts['price'].values / 1000  # $/kWh
    carbon_intensity = forecasts['emissions'].values / 1000  # kgCO2/kWh
    time_index = np.arange(0, len(forecasts) * 5, 5)

    # Add extra datapoint at 1440
    time_index = np.append(time_index, 1440)
    electricity_price = np.append(electricity_price, electricity_price[-1])
    carbon_intensity = np.append(carbon_intensity, carbon_intensity[-1])

    # Interpolate
    f_interp = interp1d(time_index, carbon_intensity, kind='linear')
    time_index = np.arange(0, 1441)  # 0 to 1440 inclusive
    electricity_price = np.repeat(electricity_price[:-1], 5)
    electricity_price = np.append(electricity_price, electricity_price[-1])
    carbon_intensity = f_interp(time_index)

    # Route/energy processing
    energy_list = np.cumsum(np.insert(eco_energy, 0, 0))
    required_charge = energy_list[-1] - max_soc + min_soc
    augmented_energy_list = []
    for i in range(len(energy_list)):
        if i != 0:
            augmented_energy_list.append(energy_list[i])
        if i < len(energy_list) - 1:
            augmented_energy_list.append(energy_list[i])

    if required_charge > 0:
        for i in range(len(augmented_energy_list)):
            if i != 0 and i % 2 == 0 and can_charge[(i-1) // 2]:
                for j in range(i, len(augmented_energy_list)):
                    augmented_energy_list[j] -= required_charge
                break

    augmented_energy_list.append(0)

    time_list = [0]
    for i in range(len(eco_time)):
        time_list.append(time_list[-1] + eco_time[i])
        time_list.append(time_list[-1] + stop_times[i])

    charge_windows = []
    for i in range(1, len(time_list)):
        if i % 2 == 0 and can_charge[(i-1) // 2]:
            charge_windows.append((time_list[i-1], time_list[i]))

    # Start time bounds
    start_time_lower_bound = max(0, start_time_nominal - min_start_shift)
    start_time_upper_bound = min(1440 - time_list[-1], start_time_nominal + max_start_shift)

    # Your lambda for carbon cost
    lambd = 0.18

    # Return all needed for milp
    return (lambd, electricity_price, carbon_intensity, time_index, 
            start_time_nominal, start_time_lower_bound, start_time_upper_bound, charge_windows)


# Example usage (to be run in a main script, not here):
# route_schedule = load_route_schedule('../data/route1_schedule.csv')
# forecasts = load_forecasts('../data/flat.csv')
# eco_energy, eco_time, stop_times, can_charge = extract_route_data(route_schedule)
# time_index, price_1min, carbon_1min = prepare_time_series_data(forecasts)
# save_time_series(time_index, price_1min, carbon_1min, outdir=".")
