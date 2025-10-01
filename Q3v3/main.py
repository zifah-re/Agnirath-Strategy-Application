import casadi as ca
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import parameters as p
import os

def get_solar_power_numeric(time_of_day):
    total_day_seconds = p.RACE_DURATION_HOURS * 3600
    if time_of_day > total_day_seconds or time_of_day < 0:
        return 0
    power = 1000 * np.sin(np.pi * time_of_day / total_day_seconds) * p.PANEL_AREA * p.PANEL_EFFICIENCY
    return max(0, power)

def get_power_out_numeric(v_mps, slope_angle):
    F_drag = 0.5 * p.AIR_DENSITY * p.C_D * p.FRONTAL_AREA * v_mps**2
    F_rolling = p.C_RR * p.CAR_MASS * p.GRAVITY * np.cos(slope_angle)
    F_gravity = p.GRAVITY * p.CAR_MASS * np.sin(slope_angle)
    F_resistive = F_drag + F_rolling + F_gravity
    if F_resistive < 0:
        return F_resistive * v_mps * p.MOTOR_EFFICIENCY 
    else:
        return (F_resistive * v_mps) / p.MOTOR_EFFICIENCY

def run_casadi_optimizer():
    """
    Solves the optimal control problem using the CasADi framework.
    """
    print("--- Agnirath Race Strategy Optimizer ---")
    print("[INFO] Preparing route data...")
    try:
        route_df = pd.read_csv(p.ROUTE_CSV_PATH)
    except FileNotFoundError:
        print(f"[ERROR] Route data file not found at '{p.ROUTE_CSV_PATH}'")
        return

    # Downsample the route for the optimization grid
    N = 100
    sample_indices = np.linspace(0, len(route_df) - 1, N + 1, dtype=int)
    route_data = route_df.iloc[sample_indices].reset_index(drop=True)
    segment_distances = np.array([
        haversine(route_data.iloc[i], route_data.iloc[i+1]) for i in range(N)
    ])
    slope_angles = np.array([
        np.arctan2(route_data.iloc[i+1]['altitude'] - route_data.iloc[i]['altitude'], segment_distances[i])
        if segment_distances[i] > 0 else 0
        for i in range(N)
    ])
    
    print(f"[INFO] Route processed into {N} segments.")

    # Setting up the Optimal Control Problem
    opti = ca.Opti()
    X = opti.variable(2, N + 1)
    E, D = X[0, :], X[1, :]
    U = opti.variable(1, N)
    T = opti.variable()
    opti.minimize(T)
    dt = T / N

    for k in range(N):
        v = U[k]
        slope = slope_angles[k]
        F_drag = 0.5 * p.AIR_DENSITY * p.C_D * p.FRONTAL_AREA * v**2
        F_rolling = p.C_RR * p.CAR_MASS * p.GRAVITY * ca.cos(slope)
        F_gravity = p.GRAVITY * p.CAR_MASS * ca.sin(slope)
        F_resistive = F_drag + F_rolling + F_gravity
        power_out_elec = (F_resistive * v) / p.MOTOR_EFFICIENCY
        time_of_day = (k / N) * (p.RACE_DURATION_HOURS * 3600)
        power_in_solar = 1000 * ca.sin(np.pi * time_of_day / (p.RACE_DURATION_HOURS * 3600)) * p.PANEL_AREA * p.PANEL_EFFICIENCY
        net_power = power_in_solar - power_out_elec
        opti.subject_to(E[k+1] == E[k] + net_power * dt)
        opti.subject_to(D[k+1] == D[k] + v * dt)
    
    opti.subject_to(E[0] == p.BATTERY_CAPACITY_J * p.INITIAL_SOC)
    opti.subject_to(D[0] == 0)
    opti.subject_to(D[N] == p.RACE_DISTANCE_M)
    opti.subject_to(opti.bounded(p.MIN_SOC * p.BATTERY_CAPACITY_J, E, p.BATTERY_CAPACITY_J))
    opti.subject_to(opti.bounded(p.MIN_VELOCITY_KPH / 3.6, U, p.MAX_VELOCITY_KPH / 3.6))
    opti.subject_to(T > 0)
    opti.set_initial(T, 8 * 3600)
    opti.set_initial(U, 60 / 3.6)

    print("\n[INFO] Starting CasADi optimizer")
    opti.solver('ipopt')
    
    try:
        sol = opti.solve()
        print("\n--- Optimal Strategy Found ---")
        
        # Extract optimal trajectories
        T_opt = sol.value(T)
        U_opt_mps = sol.value(U)
        E_opt = sol.value(E)
        
        SOC_opt = (E_opt / p.BATTERY_CAPACITY_J) * 100
        time_grid_h = np.linspace(0, T_opt / 3600, N + 1)
        power_in_hist, power_out_hist, net_power_hist = [], [], []
        for k in range(N):
            v_mps_k = U_opt_mps[k]
            slope_k = slope_angles[k]
            time_of_day_k = (k / N) * (p.RACE_DURATION_HOURS * 3600)
            
            power_out_k = get_power_out_numeric(v_mps_k, slope_k)
            power_in_k = get_solar_power_numeric(time_of_day_k)
            
            power_in_hist.append(power_in_k)
            power_out_hist.append(power_out_k)
            net_power_hist.append(power_in_k - power_out_k)
            
        print(f"Optimal Race Time: {T_opt/3600:.2f} hours")
        print(f"Final Battery SOC: {SOC_opt[-1]:.1f}%")
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle('Globally Optimal Race Strategy (CasADi)', fontsize=16)
        ax1.plot(time_grid_h[:-1], U_opt_mps * 3.6, 'b-', marker='.', markersize=4, label='Optimal Velocity')
        ax1.set_ylabel('Velocity (km/h)')
        ax1.grid(True)
        ax1.legend()
        ax2.plot(time_grid_h, SOC_opt, 'g-', marker='.', markersize=4, label='Battery SOC')
        ax2.axhline(y=p.MIN_SOC * 100, color='r', linestyle='--', label=f'Min SOC ({p.MIN_SOC:.0%})')
        ax2.set_ylabel('State of Charge (%)')
        ax2.set_ylim(0, 105)
        ax2.grid(True)
        ax2.legend()
        ax3.plot(time_grid_h[:-1], power_in_hist, 'y-', label='Solar Power In')
        ax3.plot(time_grid_h[:-1], power_out_hist, 'r-', label='Motor Power Out')
        ax3.plot(time_grid_h[:-1], net_power_hist, 'k--', label='Net Power (to Battery)')
        ax3.set_ylabel('Power (Watts)')
        ax3.set_xlabel('Time (hours)')
        ax3.grid(True)
        ax3.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        output_filename = "casadi_strategy_results.png"
        plt.savefig(output_filename, dpi=300)
        print(f"\n[INFO] Full result plots saved to '{os.path.abspath(output_filename)}'")
        plt.show()

    except RuntimeError as e:
        print("\n[ERROR] Optimization failed.")
        print(f"Solver message: {e}")

def haversine(p1, p2):
    """Helper function to calculate distance between two points from the DataFrame."""
    lat1, lon1 = p1['latitude'], p1['longitude']
    lat2, lon2 = p2['latitude'], p2['longitude']
    R = 6371e3
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

if __name__ == "__main__":
    run_casadi_optimizer()