import numpy as np
from scipy.optimize import minimize
import parameters as p
from optimization import objective_function, constraint_function
from simulation import simulate_race
from visualization import plot_results

if __name__ == "__main__":
    print("--- Agnirath Race Strategy Optimizer ---")
    # Defining bounds for each velocity step
    bounds = [(p.MIN_VELOCITY_KPH, p.MAX_VELOCITY_KPH) for _ in range(p.N_STEPS)]
    # Defining the constraint object for SciPy
    constraints = [{'type': 'ineq', 'fun': constraint_function}]
    # Creating an initial guess: A constant 60 km/h velocity profile
    initial_guess = np.full(p.N_STEPS, 60.0)
    print("\n[INFO] Starting optimization...")
    print(f"[INFO] Objective: Minimize time for {p.RACE_DISTANCE_M/1000} km.")
    print(f"[INFO] Constraints: Final SOC >= {p.MIN_SOC:.0%}.")
    # Calling the optimizer
    result = minimize(
        objective_function,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'disp': True, 'ftol': 1e-6, 'maxiter': 200}
    )
    print("\n[INFO] Optimization finished!")
    # Processing and displaying final results
    if result.success:
        optimal_velocities_kph = result.x
        final_result = simulate_race(optimal_velocities_kph)
        final_time_h = final_result['finish_time'] / 3600
        final_soc = final_result['final_soc']
        print("\n--- Optimal Strategy Found ---")
        print(f"Optimal Race Time: {final_time_h:.2f} hours")
        print(f"Final Battery SOC: {final_soc:.1%}")
        # Plotting the results
        plot_results(final_result['history'])
    else:
        print("\n[ERROR] Optimization failed to find a solution.")
        print(f"Message: {result.message}")