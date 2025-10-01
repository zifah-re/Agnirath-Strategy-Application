import numpy as np
import parameters as p
from simulation import simulate_race, get_power_out_at_v, get_solar_power
from scipy.optimize import fsolve

# STAGE 1: Initial Guessing
def calculate_env_profile(route_data):
    """
    STAGE 1: Calculate a robust initial guess velocity profile.
    Rules-based approach with an attempt to find the energy-neutral velocity
    using fsolve. Falls back to safe bounds on failure.
    Returns a numpy array of velocities in kph for each segment.
    """
    print("[INFO] Stage 1: Calculating initial guess...")
    initial_velocities_kph = []
    num_segments = max(1, len(route_data) - 1)

    for i in range(num_segments):
        start_point, end_point = route_data.iloc[i], route_data.iloc[i+1]
        # Use a representative horizontal distance for slope to avoid degenerate small-distances
        segment_dist_for_slope = 1000.0
        altitude_change = end_point['altitude'] - start_point['altitude']
        slope_angle = np.arctan2(altitude_change, segment_dist_for_slope)
        # Estimate time-of-day for solar calculation based on progress through route
        time_of_day = (i / float(num_segments)) * (p.RACE_DURATION_HOURS * 3600)
        power_in = get_solar_power(time_of_day)
        # Edge checks using min/max velocities (convert kph->mps)
        power_out_at_min_speed = get_power_out_at_v(p.MIN_VELOCITY_KPH / 3.6, slope_angle)
        power_out_at_max_speed = get_power_out_at_v(p.MAX_VELOCITY_KPH / 3.6, slope_angle)
        if power_out_at_min_speed > power_in:
            # Not enough solar even at min speed -> go slow to conserve battery
            final_v_kph = p.MIN_VELOCITY_KPH
        elif power_out_at_max_speed < power_in and power_out_at_max_speed < 0:
            # Regenerating even at max speed (big descent) -> go max
            final_v_kph = p.MAX_VELOCITY_KPH
        else:
            # Try to find v such that power_out(v) == power_in
            def power_balance_equation(v_mps):
                # fsolve might pass arrays; make scalar
                vm = float(np.atleast_1d(v_mps)[0])
                return get_power_out_at_v(vm, slope_angle) - power_in
            try:
                sol, infodict, ier, mesg = fsolve(power_balance_equation, x0=15.0, full_output=True)
                if ier == 1 and sol is not None and np.isfinite(sol).all():
                    final_v_kph = float(sol[0]) * 3.6
                else:
                    # fsolve did not converge: pick a conservative speed
                    final_v_kph = p.MIN_VELOCITY_KPH
            except Exception:
                final_v_kph = p.MIN_VELOCITY_KPH

        # Clamp within bounds
        final_v_kph = max(p.MIN_VELOCITY_KPH, min(p.MAX_VELOCITY_KPH, final_v_kph))
        initial_velocities_kph.append(final_v_kph)

    return np.array(initial_velocities_kph)

# STAGE 2: Objective & Constraint (robust penalties)
def objective_function(velocity_profile_kph, route_data):
    """
    Minimize race finish time. If simulation fails, return a smooth penalty
    depending on how far the candidate progressed (so that the optimizer
    gets useful directional information).
    """
    vel = np.asarray(velocity_profile_kph, dtype=float)
    vel = np.clip(vel, p.MIN_VELOCITY_KPH, p.MAX_VELOCITY_KPH)
    result = simulate_race(vel, route_data)

    if result.get('success', False):
        finish_time = float(result['finish_time'])
        # Small smoothness regularizer to discourage oscillations
        reg = 1e-4 * np.sum(np.diff(vel)**2)
        return finish_time + reg

    # Smooth penalty for infeasible runs, which depends on missing distance
    dist_done = float(result.get('total_distance', 0.0))
    missing = max(0.0, p.RACE_DISTANCE_M - dist_done)
    penalty = 1e6 + (missing / 1000.0)**2 * 1e3
    avg_v = float(np.mean(vel))
    penalty *= 1.0 + ((p.MAX_VELOCITY_KPH - avg_v) / (p.MAX_VELOCITY_KPH - p.MIN_VELOCITY_KPH + 1e-9)) * 0.1
    return float(min(penalty, 1e12))

def constraint_function(velocity_profile_kph, route_data):
    """
    Inequality constraint: final_soc - MIN_SOC >= 0 is required.
    If simulation fails, return a smooth negative number proportional to
    fraction completed so the constraint gradient is informative.
    """
    vel = np.asarray(velocity_profile_kph, dtype=float)
    vel = np.clip(vel, p.MIN_VELOCITY_KPH, p.MAX_VELOCITY_KPH)
    result = simulate_race(vel, route_data)
    if result.get('success', False):
        return float(result['final_soc'] - p.MIN_SOC)
    # Smooth negative value: frac_done in [0,1] -> value in [-1, 0]
    frac_done = float(result.get('total_distance', 0.0)) / max(1.0, p.RACE_DISTANCE_M)
    return float(frac_done - 1.0)
