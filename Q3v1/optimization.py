from simulation import simulate_race
import parameters as p

def objective_function(velocity_profile_kph):
    """
    The objective for the optimizer to minimize.
    Returns the total race time or a large penalty if the race fails.
    """
    result = simulate_race(velocity_profile_kph)
    
    if not result['success']:
        # Apply a heavy penalty for not finishing.
        # The penalty is smaller the closer the car got to the finish line.
        final_distance = result['final_distance']
        penalty = 1e9 + (p.RACE_DISTANCE_M - final_distance)
        return penalty
        
    return result['finish_time']

def constraint_function(velocity_profile_kph):
    """
    The constraint for the optimizer.
    Ensures the final battery SOC is above the minimum threshold.
    SciPy's SLSQP requires this to return a value >= 0 for a valid solution.
    """
    result = simulate_race(velocity_profile_kph)
    if not result['success']:
        return -1
    return result['final_soc'] - p.MIN_SOC