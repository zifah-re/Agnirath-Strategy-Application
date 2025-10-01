import numpy as np
import pandas as pd
import parameters as p
from simulation import simulate_race, haversine
from optimization import objective_function, constraint_function, calculate_env_profile
from visualization import plot_results
from scipy.optimize import differential_evolution, minimize
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)

def compute_route_length_m(route_df):
    total = 0.0
    for i in range(len(route_df) - 1):
        a = route_df.iloc[i]
        b = route_df.iloc[i + 1]
        total += haversine(a['latitude'], a['longitude'], b['latitude'], b['longitude'])
    return total

def attempt_repair_initial_guess(vel_profile, route_data, scale_min=0.4, n_steps=30):
    """Try progressively scaling down the velocity profile to find a feasible start."""
    vel = vel_profile.copy()
    for alpha in np.linspace(1.0, scale_min, n_steps):
        cand = np.clip(vel * alpha, p.MIN_VELOCITY_KPH, p.MAX_VELOCITY_KPH)
        r = simulate_race(cand, route_data)
        if r.get('success', False) and r.get('final_soc', 0.0) >= p.MIN_SOC:
            return cand, r
    return None, None

def upsample_to_full_profile(opt_velocities_kph, route_df_downsampled, original_route_df, sample_rate):
    """
    Expand the optimized coarse velocities to one per original segment.
    Returns an array of length (len(original_route_df)-1).
    """
    N_segments_full = len(original_route_df) - 1
    full_profile = np.zeros(N_segments_full, dtype=float)
    # Indices of points in the original route that correspond to the downsampled rows
    down_point_indices = (route_df_downsampled.index * sample_rate).astype(int).tolist()
    # Fill segment velocities
    for i in range(len(opt_velocities_kph)):
        start_idx = down_point_indices[i]
        end_idx = down_point_indices[i + 1] if (i + 1) < len(down_point_indices) else N_segments_full
        start_idx = max(0, min(start_idx, N_segments_full - 1))
        end_idx = max(start_idx + 1, min(end_idx, N_segments_full))
        full_profile[start_idx:end_idx] = opt_velocities_kph[i]
    # If any segments remained zero (edge cases), fill them with nearest neighbor or mean
    zeros = np.where(full_profile == 0)[0]
    if zeros.size > 0:
        # Fill forward, then backward
        for idx in zeros:
            # Try the previous one
            if idx > 0 and full_profile[idx - 1] > 0:
                full_profile[idx] = full_profile[idx - 1]
            elif idx + 1 < len(full_profile) and full_profile[idx + 1] > 0:
                full_profile[idx] = full_profile[idx + 1]
            else:
                full_profile[idx] = np.clip((p.MIN_VELOCITY_KPH + p.MAX_VELOCITY_KPH) / 2.0, p.MIN_VELOCITY_KPH, p.MAX_VELOCITY_KPH)
    return full_profile

if __name__ == "__main__":
    print("--- Agnirath Race Strategy Optimizer ---")
    try:
        original_route_df = pd.read_csv(p.ROUTE_CSV_PATH)
    except FileNotFoundError:
        print(f"[ERROR] Route data file not found at '{p.ROUTE_CSV_PATH}'")
        print("[ERROR] Please ensure the CSV is present and the path is correct in parameters.py")
        exit(1)

    # Computing the actual route distance and override parameter so our simulation and optimization are consistent.
    total_route_m = compute_route_length_m(original_route_df)
    print(f"[INFO] Route CSV distance = {total_route_m/1000:.2f} km (overriding p.RACE_DISTANCE_M)")
    p.RACE_DISTANCE_M = total_route_m

    # Downsample for optimization speed
    SAMPLE_RATE = 30 
    route_df_downsampled = original_route_df.iloc[::SAMPLE_RATE].reset_index(drop=True)
    N_STEPS = len(route_df_downsampled) - 1
    if N_STEPS < 1:
        print("[ERROR] Route too short after downsampling.")
        exit(1)
    print(f"[INFO] Downsampled route to {N_STEPS + 1} points for optimization (N_STEPS={N_STEPS}).")

    # Initial guess
    initial_guess = calculate_env_profile(route_df_downsampled)

    # Ensuring shape correctness
    initial_guess = np.asarray(initial_guess, dtype=float)
    if initial_guess.shape[0] != N_STEPS:
        print(f"[WARN] calculate_env_profile returned length {initial_guess.shape[0]} but expected {N_STEPS}. Using flat initial guess.")
        initial_guess = np.full(N_STEPS, (p.MIN_VELOCITY_KPH + p.MAX_VELOCITY_KPH) / 2.0)

    # Quick feasibility check of initial guess
    print("[DBG] Checking initial_guess feasibility before optimization...")
    init_check = simulate_race(initial_guess, route_df_downsampled)
    td = init_check.get('total_distance', 0.0)
    if td is None:
        td = 0.0
    final_soc = init_check.get('final_soc')
    final_soc_str = f"{final_soc:.6f}" if isinstance(final_soc, (int, float)) else str(final_soc)
    print(f"[DBG] initial_guess -> success={init_check.get('success')}, distance={td/1000.0:.2f} km, final_soc={final_soc_str}")


    # If infeasible, try to repair by scaling speeds down
    if not init_check.get('success', False) or (init_check.get('final_soc') is not None and init_check.get('final_soc') < p.MIN_SOC):
        print("[INFO] initial_guess infeasible -> attempting automatic repair by scaling velocities down...")
        repaired, repaired_result = attempt_repair_initial_guess(initial_guess, route_df_downsampled)
        if repaired is not None:
            print("[INFO] Found repaired initial guess (scaled down speeds). Using this to start optimization.")
            initial_guess = repaired
            print(f"[DBG] repaired final_soc={repaired_result.get('final_soc'):.3f}, distance={repaired_result.get('total_distance')/1000:.2f} km")
        else:
            print("[WARN] Automatic repair failed: no feasible scaled profile found. Optimization may still fail.")

    print("\n[INFO] Stage 2: Starting main optimization with intelligent guess...")
    bounds = [(p.MIN_VELOCITY_KPH, p.MAX_VELOCITY_KPH) for _ in range(N_STEPS)]

    # Try a global search first (DE) then refine with SLSQP
    try:
        print("[INFO] Phase 1: Running differential_evolution (global search)...")
        de_res = differential_evolution(
            lambda x: objective_function(x, route_df_downsampled),
            bounds,
            maxiter=15,
            popsize=8,
            polish=False,
            disp=True
        )
        print("[INFO] DE finished. best objective:", de_res.fun)
        x0 = de_res.x
    except Exception as e:
        print(f"[WARN] differential_evolution failed or unavailable: {e}")
        print("[INFO] Falling back to the intelligent initial guess as starting point for local optimizer.")
        x0 = initial_guess

    # Local refinement with SLSQP, which requires `constraint_function` to be robust
    try:
        result = minimize(
            objective_function,
            x0,
            method='SLSQP',
            args=(route_df_downsampled,),
            bounds=bounds,
            constraints=[{'type': 'ineq', 'fun': constraint_function, 'args': (route_df_downsampled,)}],
            options={'disp': True, 'ftol': 1e-6, 'maxiter': 200}
        )
    except Exception as e:
        print(f"[ERROR] Local optimization failed with exception: {e}")
        result = None

    print("\n[INFO] Optimization finished!")

    if result is None or not getattr(result, "success", False):
        print("\n[ERROR] Optimization failed to find a solution.")
        if result is not None:
            print(f"Message: {getattr(result, 'message', '<no message>')}")
        print("[INFO] You can try:\n - increasing SAMPLE_RATE (faster, but coarser)\n - relaxing MIN_SOC slightly in parameters for testing\n - running the DE-only solution (de_res.x) as a fallback\n")
        if 'x0' in locals():
            print("[INFO] Running final high-resolution simulation with the best available candidate (fallback).")
            best_candidate = x0
        else:
            print("[ERROR] No candidate available to simulate. Exiting.")
            exit(1)
    else:
        optimal_velocities_kph = result.x
        best_candidate = optimal_velocities_kph
        print("[INFO] Local optimization success!")

    # Upsample to full-resolution profile for the original route and simulate for plotting
    full_velocity_profile_kph = upsample_to_full_profile(best_candidate, route_df_downsampled, original_route_df, SAMPLE_RATE)

    print("\n[INFO] Running final high-resolution simulation for plotting...")
    final_result = simulate_race(full_velocity_profile_kph, original_route_df)

    if final_result.get('success', False):
        final_time_h = final_result['finish_time'] / 3600.0
        final_soc = final_result['final_soc']
        print("\n--- Final Simulation Results ---")
        print(f"Race Time   : {final_time_h:.2f} hours")
        print(f"Final SOC   : {final_soc:.1%}")
        out_fig = "race_strategy_results.png"
        try:
            plot_results(final_result['history'])
        except Exception as e:
            print(f"[WARN] Plotting failed: {e}")
            hist = final_result.get('history', {})
            if hist:
                df_hist = pd.DataFrame(hist)
                csv_out = "race_history_debug.csv"
                df_hist.to_csv(csv_out, index=False)
                print(f"[INFO] Simulation history written to {os.path.abspath(csv_out)}")
    else:
        print("\n[ERROR] The final high-resolution simulation failed.")
        print(f"Reached distance: {final_result.get('total_distance', 0.0)/1000.0:.2f} km")
        print("Try increasing battery capacity or relaxing constraints for debugging.")
