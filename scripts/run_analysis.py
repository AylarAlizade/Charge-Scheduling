import sys
import os
import time
import matplotlib
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_loader import (
    load_route_schedule, load_forecasts, extract_route_data, extract_grid_data,
    prepare_time_series_data, save_time_series,prepare_milp_inputs
)
from optimization import (
    window_builder, milp,winows_lower, brute_force, lp, bellman_ford_schedule,
    calculate_total_cost, calculate_electricity_cost, calculate_cumulative_cost, calculate_sumulative_SOC
)
from utils import (
    set_plot_style, plot_electricity_price, plot_carbon_intensity, plot_remaining_energy, save_plot
)

def main():
    inputs = prepare_milp_inputs("2024-11-17.csv", "route1_schedule.csv")
    lambd, electricity_price, carbon_intensity, time_index,start_time_nominal, start_time_lower_bound, start_time_upper_bound, charge_windows = inputs
    optimized_start_time, optimized_total_cost, charge_flag = milp(*inputs)
    mip_windows = window_builder(charge_flag)
    ## Parameters
    # Define parameters
    battery_capacity = 256
    eta_c = 0.5  # charging efficiency
    eta_d = 0.9  # discharging efficiency
    lambd = 0.18  # weight for carbon intensity
    C_deg = 0  # degradation cost coefficient
    SOC_min = 0.1 * battery_capacity  # minimum SOC
    SOC_max = 0.9 * battery_capacity  # maximum SOC
    max_charge_power = 50  # kW, maximum charging power
    max_discharge_power = 50  # kW, maximum discharging power
    C_c_max = 1000  # kVA, maximum capacity of the transformer
    delta_t = 1  # minutes, time step size
    p_t = electricity_price  # electricity price at each time step
    g_t = carbon_intensity  # grid carbon intensity at each time step
    # Required charging (given by route scheduler)
    SOC_init = SOC_min
    SOC_target = 0.5 * battery_capacity  ## Change this one instead of the SOC max

    # Define parameters
    charging_window = charge_windows[0]
    charge_window_start = optimized_start_time + charging_window[0]  # start time of the charging window
    charge_window_end = min(1440,optimized_start_time + charging_window[1])  # end time of the charging window
    start_idx = int(charge_window_start // delta_t)
    end_idx = int(charge_window_end // delta_t)
    T = end_idx - start_idx  # number of time steps
    delta_t = 1  # minutes, time step size

    n = T

    p_t = electricity_price[start_idx:end_idx+1]  # electricity price at each time step
    g_t = carbon_intensity[start_idx:end_idx+1]  # grid carbon intensity at each time step

    # Load the data

    def T_builder(cw):
        charge_window_start = optimized_start_time + cw[0]  # start time of the charging window
        charge_window_end = min(1440,optimized_start_time + cw[1])  # end time of the charging window
        start_idx = int(charge_window_start // delta_t)
        end_idx = int(charge_window_end // delta_t)
        T = end_idx - start_idx  # number of time steps
        p_t = electricity_price[start_idx:end_idx+1]  # electricity price at each time step
        g_t = carbon_intensity[start_idx:end_idx+1]  # grid carbon intensity at each time step
        return T,p_t,g_t,start_idx,end_idx

    def marker(c_t,charge_flag,a,b):
        j = 0
        for i in range(a,b):
            if c_t[j] > 0:
                charge_flag[i] = 50
            j = j + 1
            return charge_flag
        
    SOC_init = SOC_min
    charge_flag_milp = [0] * 1440
    discharge_flag_milp = [0] * 1440
    start_lower_lp = time.time()
    for i,cw in enumerate(charge_windows):
        T,p_t,g_t,start_idx,end_idx = T_builder(cw)
        SOC_init = SOC_init + (delta_t / 60.0) * (sum(charge_flag_milp) * eta_c - sum(discharge_flag_milp) * eta_d)
        print(SOC_init)
        low = max(SOC_init - T * (delta_t / 60.0),SOC_min)
        high = min(SOC_init + T * (delta_t / 60.0),SOC_max)
        if i != (len(charge_windows)-1):
            optimized_c_t, optimized_d_t, optimized_SOC = lp(p_t, g_t, T, delta_t,battery_capacity,eta_c,eta_d,C_deg,max_charge_power,max_discharge_power,C_c_max,SOC_min,SOC_max,SOC_init,SOC_target = None,SOC_range=(low,high))

            

        else:
            optimized_c_t, optimized_d_t, optimized_SOC = lp(p_t, g_t, T, delta_t,battery_capacity,eta_c,eta_d,C_deg,max_charge_power,max_discharge_power,C_c_max,SOC_min,SOC_max,SOC_init,SOC_target=SOC_target,SOC_range=None)
            SOC_init = optimized_SOC[-1]
            
        
        charge_flag_milp =  marker(optimized_c_t,charge_flag_milp,start_idx,end_idx)
        discharge_flag_milp =  marker(optimized_d_t,discharge_flag_milp,start_idx,end_idx)
    end_lower_lp = time.time()
    duration_lower_lp = end_lower_lp - start_lower_lp
    print(f"Elapsed time: {end_lower_lp - start_lower_lp:.4f} seconds")


    SOC_init = SOC_min
    charge_flag_bf = [0] * 1440
    discharge_flag_bf = [0] * 1440
    start_lower_bf = time.time()
    for i,cw in enumerate(charge_windows):
        # SOC_init = SOC_init + (delta_t / 60.0) * (sum(charge_flag_bf) * eta_c + sum(discharge_flag_bf) * eta_d)
        T,p_t,g_t,start_idx,end_idx = T_builder(cw)
        low = max(SOC_init - T * (delta_t / 60.0),SOC_min)
        high = min(SOC_init + T * (delta_t / 60.0),SOC_max)
        if i != (len(charge_windows)-1):
            best_cost, schedule = bellman_ford_schedule(p_t, g_t, T, delta_t,
                                                SOC_init, battery_capacity,
                                                C_c_max,SOC_min,SOC_max, max_charge_power, max_discharge_power,
                                                eta_c, eta_d, lambd, C_deg,SOC_target=None,SOC_range=(low,high), resolution=1)
            
        else:
            best_cost, schedule = bellman_ford_schedule(p_t, g_t, T, delta_t,
                                                SOC_init, battery_capacity,
                                                C_c_max,SOC_min,SOC_max, max_charge_power, max_discharge_power,
                                                eta_c, eta_d, lambd, C_deg,SOC_target=SOC_target,resolution=1)
        SOC_init = schedule[-1][3]
            
            
        c_t_bf = [0] * len(schedule)
        d_t_bf = [0] * len(schedule)
        for j in range(len(schedule)):

            if schedule[j][1] == 'charge':

                c_t_bf[j] = 1
            elif schedule[j][1] == 'discharge':
                d_t_bf[j] = 1

        charge_flag_bf =  marker(c_t_bf,charge_flag_bf,start_idx,end_idx)
        discharge_flag_bf =  marker(d_t_bf,discharge_flag_bf,start_idx,end_idx)
    end_lower_bf = time.time()
    duration_lower_bf = end_lower_bf - start_lower_bf
    print(f"Elapsed time: {duration_lower_bf:.4f} seconds")

if __name__ == "__main__":
    main()
