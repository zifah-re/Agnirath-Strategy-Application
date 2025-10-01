import numpy as np
import pandas as pd
import parameters as p

def get_solar_power(time_seconds):
    """Generates a synthetic solar power profile for a fixed 8-hour day."""
    total_day_seconds = p.RACE_DURATION_HOURS * 3600
    if time_seconds > total_day_seconds: return 0 # No sun after 8 hours
    irradiance = 1000 * np.sin(np.pi * time_seconds / total_day_seconds)
    return max(0, irradiance) * p.PANEL_AREA * p.PANEL_EFFICIENCY

def get_power_out_at_v(v, slope_angle):
    """Calculates the electrical power needed to maintain velocity v on a given slope."""
    F_drag = 0.5 * p.AIR_DENSITY * p.C_D * p.FRONTAL_AREA * v**2
    F_rolling = p.C_RR * p.CAR_MASS * p.GRAVITY * np.cos(slope_angle)
    F_gravity = p.GRAVITY * p.CAR_MASS * np.sin(slope_angle)
    F_resistive = F_drag + F_rolling + F_gravity
    
    if F_resistive < 0:
        return F_resistive * v * p.MOTOR_EFFICIENCY # Regen
    else:
        return (F_resistive * v) / p.MOTOR_EFFICIENCY

def simulate_race(velocity_profile_kph, route_data):
    """
    Simulates the entire race for a given velocity profile.
    Always returns a dict containing:
      - success (bool)
      - total_distance (meters, float)
      - final_soc (fraction 0..1, float)
      - history (dict with time, velocity, distance(km), soc(%), power_in, power_out, net_power)
    """
    v_profile_mps = np.asarray(velocity_profile_kph, dtype=float) / 3.6

    battery_energy = p.BATTERY_CAPACITY_J * p.INITIAL_SOC
    total_distance = 0.0
    total_time = 0.0

    # Initializing history
    history = {
        'time': [0.0],
        'velocity': [float(velocity_profile_kph[0]) if len(velocity_profile_kph) > 0 else 0.0],
        'distance': [0.0],
        'soc': [p.INITIAL_SOC * 100.0],
        'power_in': [0.0],
        'power_out': [0.0],
        'net_power': [0.0],
    }

    min_allowed_energy = p.MIN_SOC * p.BATTERY_CAPACITY_J

    for i in range(len(v_profile_mps)):
        v = float(v_profile_mps[i])
        if i + 1 >= len(route_data):
            return {'success': False, 'total_distance': total_distance, 'final_soc': battery_energy / p.BATTERY_CAPACITY_J, 'history': history}

        a = route_data.iloc[i]
        b = route_data.iloc[i + 1]

        segment_dist = haversine(a['latitude'], a['longitude'], b['latitude'], b['longitude'])
        # Avoid zero velocity at all costs
        if v <= 1e-6:
            return {'success': False, 'total_distance': total_distance, 'final_soc': battery_energy / p.BATTERY_CAPACITY_J, 'history': history}

        segment_time = segment_dist / v if v > 0 else float('inf')
        altitude_change = float(b['altitude']) - float(a['altitude'])
        slope_angle = np.arctan2(altitude_change, segment_dist if segment_dist > 0 else 1.0)

        power_out_elec = get_power_out_at_v(v, slope_angle)
        time_of_day = (total_distance / p.RACE_DISTANCE_M) * (p.RACE_DURATION_HOURS * 3600) if p.RACE_DISTANCE_M > 0 else 0.0
        power_in_solar = get_solar_power(time_of_day)
        net_power = power_in_solar - power_out_elec

        battery_energy += net_power * segment_time

        # Record this step into history before possible failure to keep lengths consistent
        total_distance_next = total_distance + segment_dist
        total_time_next = total_time + segment_time
        history['time'].append(total_time_next / 3600.0)
        history['velocity'].append(v * 3.6)
        history['distance'].append(total_distance_next / 1000.0)
        history['soc'].append(max(0.0, min(100.0, battery_energy / p.BATTERY_CAPACITY_J * 100.0)))
        history['power_in'].append(power_in_solar)
        history['power_out'].append(power_out_elec)
        history['net_power'].append(net_power)

        if battery_energy < min_allowed_energy:
            return {'success': False, 'total_distance': total_distance_next, 'final_soc': battery_energy / p.BATTERY_CAPACITY_J, 'history': history}

        if battery_energy > p.BATTERY_CAPACITY_J:
            battery_energy = p.BATTERY_CAPACITY_J

        total_distance = total_distance_next
        total_time = total_time_next

        if total_distance >= p.RACE_DISTANCE_M:
            overshoot = total_distance - p.RACE_DISTANCE_M
            time_to_cover_overshoot = overshoot / v if v > 0 else 0.0
            final_time = total_time - time_to_cover_overshoot
            return {'success': True, 'finish_time': final_time, 'total_distance': total_distance, 'final_soc': battery_energy / p.BATTERY_CAPACITY_J, 'history': history}

    # If reached end of provided route, treat as success
    return {'success': True, 'finish_time': total_time, 'total_distance': total_distance, 'final_soc': battery_energy / p.BATTERY_CAPACITY_J, 'history': history}

def haversine(lat1, lon1, lat2, lon2):
    """Calculates the great-circle distance between two points on Earth using latitude and longitude coordinates"""
    R = 6371e3
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))