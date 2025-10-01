import numpy as np
import parameters as p

def get_solar_power(time_seconds):
    """Generates a synthetic solar power profile for a fixed 8-hour day."""
    total_day_seconds = p.RACE_DURATION_HOURS * 3600
    if time_seconds > total_day_seconds: return 0 # No sun after 8 hours
    irradiance = 1000 * np.sin(np.pi * time_seconds / total_day_seconds)
    return max(0, irradiance) * p.PANEL_AREA * p.PANEL_EFFICIENCY

def simulate_race(velocity_profile_kph):
    """
    Simulates the entire race for a given velocity profile.
    Returns:
        dict: A dictionary containing the simulation results.
    """
    v_profile_mps = velocity_profile_kph / 3.6
    
    battery_energy = p.BATTERY_CAPACITY_J * p.INITIAL_SOC
    total_distance = 0
    total_time = 0

    # Initializing history
    history = {
        'time': [0],
        'velocity': [v_profile_mps[0] * 3.6],
        'distance': [0],
        'soc': [p.INITIAL_SOC * 100],
        'power_in': [get_solar_power(0)],
        'power_out': [0],
        'net_power': [0]
    }

    min_allowed_energy = p.MIN_SOC * p.BATTERY_CAPACITY_J

    for i in range(p.N_STEPS):
        v = v_profile_mps[i]
        
        # Calculating Power Consumption
        F_drag = 0.5 * p.AIR_DENSITY * p.C_D * p.FRONTAL_AREA * v**2
        F_rolling = p.C_RR * p.CAR_MASS * p.GRAVITY
        F_resistive = F_drag + F_rolling
        
        power_out_mech = F_resistive * v
        power_out_elec = power_out_mech / p.MOTOR_EFFICIENCY
        
        # Calculating Power Generation
        current_time = i * p.DT_SECONDS
        power_in = get_solar_power(current_time)
        
        # Updating Battery and Checking Constraints
        net_power = power_in - power_out_elec
        battery_energy += net_power * p.DT_SECONDS

        if battery_energy < min_allowed_energy:
            # Failure: Battery died before the race finished
            return {'success': False, 'finish_time': float('inf'), 'final_soc': 0, 'history': history, 'final_distance': total_distance}
        
        if battery_energy > p.BATTERY_CAPACITY_J:
            # Cap battery at 100%
            battery_energy = p.BATTERY_CAPACITY_J

        # Updating Progress
        distance_step = v * p.DT_SECONDS
        total_distance += distance_step
        total_time += p.DT_SECONDS
        
        # --- Recording History ---
        history['time'].append(total_time / 3600)
        history['velocity'].append(v * 3.6)
        history['distance'].append(total_distance / 1000)
        history['soc'].append(battery_energy / p.BATTERY_CAPACITY_J * 100)
        history['power_in'].append(power_in)
        history['power_out'].append(power_out_elec)
        history['net_power'].append(net_power)
        
        if total_distance >= p.RACE_DISTANCE_M:
            # Success: Race finished
            # Interpolating to find exact finish time
            overshoot_dist = total_distance - p.RACE_DISTANCE_M
            time_to_cover_overshoot = overshoot_dist / v
            final_time = total_time - time_to_cover_overshoot
            
            return {'success': True, 'finish_time': final_time, 'final_soc': battery_energy / p.BATTERY_CAPACITY_J, 'history': history}

    # Failure: Did not finish the race in the allotted time
    return {'success': False, 'finish_time': float('inf'), 'final_soc': battery_energy / p.BATTERY_CAPACITY_J, 'history': history, 'final_distance': total_distance}