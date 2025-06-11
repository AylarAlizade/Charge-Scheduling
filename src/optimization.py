import math
import numpy as np
import pandas as pd
import datetime
from scipy.interpolate import interp1d

import pyomo.environ as pyomo
import pulp

import os
import csv

import matplotlib
import matplotlib.pyplot as plt

def window_builder(charge_flag):
    indexes = [i for i, val in enumerate(charge_flag) if val == 1]
    windows = []
    a = 0
    b = 0
    for i in range(len(indexes)):
        if i == 0 :
            a = indexes[i]
        elif i == len(indexes) - 1:
            b = indexes[-1]
        elif indexes[i] - indexes[i-1] > 1 :
            b = indexes[i-1]
            windows.append((a,b))
            a = indexes[i]
    windows.append((a,b))
    return windows

def winows_lower(c_t,windows,charge_flag):
    charge_flag_new = [0] * len(charge_flag)
    (a,b) = windows
    j = 0
    for i in range(a,b):
        if c_t[j] > 0:
            charge_flag_new[i] = 50
        j = j + 1
    
    return charge_flag_new



def milp(lambd,electricity_price,carbon_intensity,time_index,start_time_nominal,start_time_lower_bound,start_time_upper_bound,charge_windows):
    price_dict = dict(zip(time_index, electricity_price))
    emission_dict = dict(zip(time_index, carbon_intensity))

    # Create duration_dict with the time gap over which the current measure is valid
    duration_dict = {t: 1/60 for t in time_index}
    # Initialize optimization problem
    prob = pulp.LpProblem("Schedule_Route_Start", pulp.LpMinimize)

    # Declare time index variable
    #time = pulp.LpVariable.dicts("t", time_index, cat='Continuous')

    # Declare decision variable(s)
    t0 = pulp.LpVariable("t0", lowBound=start_time_lower_bound, upBound=start_time_upper_bound, cat='Integer')
    charge_window_dict = pulp.LpVariable.dicts("in_window", time_index, cat='Binary')

    # Add objective function to problem first
    prob += pulp.lpSum([charge_window_dict[t] * (price_dict[t] + lambd * emission_dict[t]) * duration_dict[t] for t in time_index])

    # Create auxiliary binary variables for each charge window
    charge_window_vars = {}
    for i, (start, end) in enumerate(charge_windows):
        rounded_start = round(start)-0.001   # NOTE: count cost from first minute (change with -0.001)
        rounded_end = round(end)-0.001 # NOTE: do not count cost from extra minute (change with -0.001)
        charge_window_vars[i] = pulp.LpVariable.dicts(f"window_{i}", time_index, lowBound=0, upBound=1, cat='Binary')
        for t in time_index:
            # Create auxiliary variables to check if t is within the charge window
            within_window_start = pulp.LpVariable(f"within_window_start_{i}_{t}", cat='Binary')
            within_window_end = pulp.LpVariable(f"within_window_end_{i}_{t}", cat='Binary')

            prob += within_window_start <= (t - (rounded_start + t0)) / 1440 + 1
            prob += within_window_start >= (t - (rounded_start + t0)) / 1440

            prob += within_window_end <= ((rounded_end + t0) - t) / 1440 + 1
            prob += within_window_end >= ((rounded_end + t0) - t) / 1440

            prob += charge_window_vars[i][t] <= within_window_start
            prob += charge_window_vars[i][t] <= within_window_end
            prob += charge_window_vars[i][t] >= within_window_start + within_window_end - 1

    # Combine charge window variables to determine if the time falls within any window
    for t in time_index:
        prob += charge_window_dict[t] == pulp.lpSum([charge_window_vars[i][t] for i in range(len(charge_windows))])

    # Option: set desired t0 to show computed cost
    #prob += t0 == 130.

    # Solve the problem
    prob.solve()
    # Get the optimized start time
    optimized_start_time = pulp.value(t0)
# Evaluate the total cost with the optimized values
    optimized_total_cost = sum(
        pulp.value(charge_window_dict[t]) * (price_dict[t] + lambd * emission_dict[t]) * duration_dict[t]
        for t in time_index
    )

    charge_flag = [pulp.value(charge_window_dict[t]) for t in time_index]

    return optimized_start_time, optimized_total_cost , charge_flag

def brute_force(lambd,electricity_price,carbon_intensity,time_index,start_time_nominal,start_time_lower_bound,start_time_upper_bound,charge_windows):
    #start
    delta_t = 1 # 1-minute intervals
    possible_start_times = np.arange(start_time_lower_bound, start_time_upper_bound + 1, delta_t)  # 1-minute intervals
    carbon_intensity_interp = interp1d(time_index, carbon_intensity, kind='linear', fill_value="extrapolate")

    # Initialize variables to store the optimal start time and its cost
    optimal_start_time = None
    optimal_rounded_start_time = None
    minimal_cost = float('inf')
    minimal_rounded_cost = float('inf')
    # Brute-force search over possible start times
    for start_time in possible_start_times:
        total_cost = 0
        total_rounded_cost = 0

        for (start, end) in charge_windows:
            start_time_adjusted = start_time + start
            end_time_adjusted = start_time + end
            start_time_rounded = round(start_time) + round(start)   # NOTE: count cost from first minute (change with -0.001?)
            end_time_rounded = round(start_time) + round(end) # NOTE: do not count cost from extra minute (change with +/-0.001?)

            # Ensure we are within the bounds of the time_index
            if start_time_adjusted >= time_index[-1] or end_time_adjusted <= time_index[0]:
                continue

            # Calculate the cost within the charging window
            while start_time_adjusted < end_time_adjusted:
                next_interval = start_time_adjusted + delta_t # 1-minute intervals
                if next_interval > end_time_adjusted:
                    next_interval = end_time_adjusted

                # Electricity price is constant within each hour
                current_price_interval = int(start_time_adjusted // delta_t) # 1-minute intervals
                price = electricity_price[current_price_interval]

                # Carbon intensity (interpolated)
                carbon_start = carbon_intensity_interp(start_time_adjusted)
                carbon_end = carbon_intensity_interp(next_interval)
                avg_carbon = (carbon_start + carbon_end) / 2

                # Duration of this segment
                duration = (next_interval - start_time_adjusted) / 60  # converting to hours

                # Incremental cost
                total_cost += (price + lambd * avg_carbon) * duration

                start_time_adjusted = next_interval

            # Calculate the cost within the rounded charging window
            while start_time_rounded < end_time_rounded:
                next_interval = start_time_rounded + delta_t # 1-minute intervals
                if next_interval >= end_time_rounded:
                    next_interval = end_time_rounded

                # Electricity price is constant within each hour
                current_price_interval = int(start_time_rounded // delta_t) # 1-minute intervals
                price = electricity_price[current_price_interval]

                # Carbon intensity (interpolated)
                carbon_start = carbon_intensity_interp(start_time_rounded)
                carbon_end = carbon_intensity_interp(next_interval)
                avg_carbon = (carbon_start + carbon_end) / 2

                # Duration of this segment
                duration = (next_interval - start_time_rounded) / 60  # converting to hours

                # Incremental cost
                #print(f"Cost at {start_time_rounded}:",(price + lambd * avg_carbon) * duration)
                total_rounded_cost += (price + lambd * avg_carbon) * duration

                start_time_rounded = next_interval

        # Update the optimal start time if the current total cost is lower
        if total_cost < minimal_cost:
            minimal_cost = total_cost
            optimal_start_time = start_time
        if total_rounded_cost < minimal_rounded_cost:
            minimal_rounded_cost = total_rounded_cost
            optimal_rounded_start_time = start_time
    # Optimal charge windows
    optimal_charge_windows = []
    optimal_rounded_charge_windows = []
    for i,(start, end) in enumerate(charge_windows):
        start_time_adjusted = optimal_start_time + start
        end_time_adjusted = optimal_start_time + end
        optimal_charge_windows.append((start_time_adjusted,end_time_adjusted))

        start_time_rounded = optimal_rounded_start_time + round(start)
        end_time_rounded = optimal_rounded_start_time + round(end)
        optimal_rounded_charge_windows.append((start_time_rounded,end_time_rounded))

    charge_flag_brute = [0]*1440
    
    for i in range(len(charge_flag_brute)):
        for (a,b) in optimal_rounded_charge_windows:
            if i > a and i < b:
                charge_flag_brute[i] = 1

    return optimal_start_time, minimal_cost , charge_flag_brute


def lp(p_t, g_t, T, delta_t,battery_capacity,eta_c,eta_d,C_deg,max_charge_power,max_discharge_power,C_c_max,SOC_min,SOC_max,SOC_init,SOC_target = None,SOC_range=None):
    # Define parameters
    if SOC_range is None and SOC_target is None:
        raise ValueError("Provide either SOC_target or SOC_range.")
    lambd = 0.18  # weight for carbon intensity


    # Assume these are given or calculated previously
    L_net = np.zeros(T)  # net load at each time step C_C_max
    SOC_togo = 0  # kWh, required SOC to complete the route
    # Define the optimization problem
    prob = pulp.LpProblem("LowLevelChargeScheduler", pulp.LpMinimize)

    # Define decision variables
    c_t = pulp.LpVariable.dicts("c_t", range(T), lowBound=0, upBound=max_charge_power)
    d_t = pulp.LpVariable.dicts("d_t", range(T), lowBound=0, upBound=max_discharge_power)
    omega_t = pulp.LpVariable.dicts("omega_t", range(T), cat='Binary')

    # Define SOC variable and initial SOC
    SOC = [pulp.LpVariable(f"SOC_{t}", lowBound=SOC_min, upBound=SOC_max) for t in range(T + 1)]
    SOC[0] = SOC_init

    # Objective function
    objective = pulp.lpSum([
    (p_t[t] * (c_t[t] * eta_c - d_t[t] * eta_d) + lambd * g_t[t] * (c_t[t] * eta_c - d_t[t] * eta_d)) * (delta_t / 60) + C_deg * d_t[t]
    for t in range(T)
    ])
    prob += objective

    # SOC dynamics and constraints
    for t in range(T):
        prob += SOC[t + 1] == SOC[t] + (c_t[t] * eta_c - d_t[t] * eta_d) * (delta_t / 60)
        prob += c_t[t] <= max_charge_power * omega_t[t]
        prob += d_t[t] <= max_discharge_power * (1 - omega_t[t])

    # Net load constraint
    for t in range(T):
        prob += L_net[t] + c_t[t] * eta_c - d_t[t] * eta_d <= C_c_max

    # Window SOC constraint
    if SOC_range is not None:
        lo, hi = SOC_range
        prob += SOC[T] >= lo
        prob += SOC[T] <= hi
    else:
        print(SOC_target)
        prob += SOC[T] >= SOC_target

    # Solve the problem
    prob.solve()

    # Retrieve the optimized schedules
    optimized_c_t = [pulp.value(c_t[t]) for t in range(T)]
    optimized_d_t = [pulp.value(d_t[t]) for t in range(T)]
    optimized_SOC = [pulp.value(SOC[t]) for t in range(T + 1)]


    return optimized_c_t, optimized_d_t, optimized_SOC



def bellman_ford_schedule(p_t, g_t, T, delta_t, 
                          SOC_init, battery_capacity, 
                          C_c_max,SOC_min,SOC_max, max_charge_power, max_discharge_power, 
                          eta_c, eta_d, lambd, C_deg,SOC_target = None,SOC_range=None,resolution=1.0):
    if SOC_range is None and SOC_target is None:
        raise ValueError("Provide either SOC_target or SOC_range.")
    
    # Pre-calculate energy per minute (in kWh)
    energy_rate = 50 * (delta_t / 60.0)  # same for charging; for discharging, we later apply efficiency

    # Initialize dynamic programming tables:
    # dp[t] is a dictionary mapping SOC (rounded to resolution) to minimal cost to get there at time t.
    dp = [{} for _ in range(T + 1)]
    dp[0][round(SOC_init, 2)] = 0  # starting at t=0 with SOC_init and zero cost.

    # To backtrack the decisions, store predecessors.
    predecessor = [{} for _ in range(T + 1)]
    
    # Define the three possible actions.
    # For each action, we compute the SOC change over one minute.
    # Note: In our cost function, the net energy change for charging/discharging is:
    #    net_energy = (power * efficiency), where power is always 50 kW (converted to kWh per minute).
    actions = {
        'charge': {
            # 'soc_change': energy_rate * eta_c,       # Increase by energy_rate*eta_c kWh
            'soc_change': energy_rate * eta_c,       # Increase by energy_rate*eta_c kWh
            'degradation': 0,                         # No degradation cost for charging
            'multiplier': 1,
        },
        'discharge': {
            'soc_change': -energy_rate * eta_d ,        # Decrease by energy_rate*eta_d kWh
            'degradation': C_deg * max_discharge_power, # Degradation cost penalty
            'multiplier': 1,
        },
        'idle': {
            'soc_change': 0,
            'degradation': 0,
            'multiplier': 0,
        }
    }
    
    # Iterate over each time step in the planning horizon.
    for t in range(T):
        # Loop over each state (SOC) reached at time step t.
        for soc in list(dp[t].keys()):
            current_cost = dp[t][soc]
            
            # Try each action.
            for action, params in actions.items():
                new_soc = soc + params['soc_change']
                
                # Check SOC feasibility (keeping within physical bounds)
                if new_soc < SOC_min or new_soc > SOC_max:
                    continue  # discard this transition if it violates limits
                
                # Round new_soc to reduce floating-point issues.
                new_soc = round(new_soc, 3)
                
                # Compute the net energy transferred (in kWh) with its sign.
                net_energy = params['soc_change']  # positive for charge, negative for discharge, zero for idle
                if net_energy > C_c_max:
                    continue  # discard this transition if it violates limits
                # Compute the cost incurred at time t for taking this action.
                # Cost components: 
                #   (i) energy cost: p_t[t] * net_energy
                #  (ii) carbon cost: lambd * g_t[t] * net_energy
                #  (iii) degradation cost if discharging: provided by params['degradation']
                # Multiply by (delta_t/60) to adjust the units.
                base_cost = (p_t[t] * net_energy + lambd * g_t[t] * net_energy) * (delta_t / 60.0) * params['multiplier']
                transition_cost = base_cost + params['degradation']
                # transition_cost = base_cost 

                total_cost = current_cost + transition_cost
                
                # Update the DP table for time t+1 if this transition offers a lower cost.
                if new_soc not in dp[t + 1] or total_cost < dp[t + 1][new_soc]:
                    dp[t + 1][new_soc] = total_cost
                    predecessor[t + 1][new_soc] = (soc, action)
    
    # Terminal: look for any state at time T that meets the target SOC (within a small tolerance)
    tolerance = resolution
    if SOC_range is not None:
        lo, hi = SOC_range
        def admissible(s): return (lo - tolerance) <= s <= (hi + tolerance)
    else:
        target = SOC_target
        def admissible(s): return abs(s - target) <= tolerance
    best_soc = None
    best_cost = float('inf')
    for soc, cost in dp[T].items():
        if admissible(soc) and cost < best_cost:
            best_cost = cost
            best_soc = soc
    if best_soc is None:
        print("No feasible solution that reaches the target SOC was found.")
        return None, None
    
    # Backtracking to retrieve the optimal schedule.
    schedule = []
    soc = best_soc
    for t in range(T, 0, -1):
        prev_soc, action = predecessor[t][soc]
        schedule.append((t - 1, action, prev_soc, soc))
        soc = prev_soc
    schedule.reverse()  # The schedule is built backward so reverse it for chronological order.
    

    return best_cost, schedule


def calculate_total_cost(c_t, d_t, p_t, g_t, eta_c, eta_d, lambd, C_deg, delta_t):
    total_cost = sum([
        (p_t[t] * (c_t[t] * eta_c - d_t[t] * eta_d) + lambd * g_t[t] * (c_t[t] * eta_c - d_t[t] * eta_d)) * (delta_t / 60)
        + C_deg * d_t[t]
        for t in range(len(c_t))
    ])
    return total_cost


def calculate_electricity_cost(c_t, d_t, p_t, g_t, eta_c, eta_d, lambd, C_deg, delta_t):
    electricity_cost = []
    total_cost = 0
    for t in range(len(c_t)):
        cost_at_t = (p_t[t] * (c_t[t] * eta_c - d_t[t] * eta_d)) * (delta_t / 60)
        cost_at_t += C_deg * d_t[t]
        total_cost += cost_at_t
        electricity_cost.append(total_cost)
    return  electricity_cost


def calculate_cumulative_cost(c_t, d_t, p_t, g_t, eta_c, eta_d, lambd, C_deg, delta_t):
    cumulative_cost = []
    total_cost = 0
    for t in range(len(c_t)):
        cost_at_t = (p_t[t] * (c_t[t] * eta_c - d_t[t] * eta_d) + lambd * g_t[t] * (c_t[t] * eta_c - d_t[t] * eta_d)) * (delta_t / 60)
        cost_at_t += C_deg * d_t[t]
        total_cost += cost_at_t
        cumulative_cost.append(total_cost)
    return cumulative_cost


def calculate_sumulative_SOC(c_t, d_t, p_t, g_t, eta_c, eta_d, lambd, C_deg, delta_t,SOC_init):
    cumulative_soc = [SOC_init]
    total_cost = SOC_init
    for t in range(len(c_t)):
        cost_at_t = ((c_t[t] * eta_c - d_t[t] * eta_d )) * (delta_t / 60)
        total_cost += cost_at_t
        cumulative_soc.append(total_cost)
    return cumulative_soc







