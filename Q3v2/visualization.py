import matplotlib.pyplot as plt
import parameters as p
import os
import numpy as np

def plot_results(simulation_history):
    """
    Generates and saves plots for the simulation results to the current directory.
    """
    history = simulation_history
    time_h = history.get('time', [])
    if len(time_h) == 0:
        print("[WARN] Empty history — nothing to plot.")
        return
    velocity = history.get('velocity', [np.nan] * len(time_h))
    soc = history.get('soc', [np.nan] * len(time_h))
    power_in = history.get('power_in')
    power_out = history.get('power_out')
    net_power = history.get('net_power')
    n = len(time_h)
    def safe_array(arr):
        if arr is None:
            return np.zeros(n)
        arr = np.asarray(arr)
        if arr.shape[0] != n:
            if arr.shape[0] > n:
                return arr[:n]
            else:
                pad = np.zeros(n - arr.shape[0])
                return np.concatenate([arr, pad])
        return arr

    power_in = safe_array(power_in)
    power_out = safe_array(power_out)
    net_power = safe_array(net_power)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Velocity Profile
    ax1.plot(time_h, velocity, marker='.', label='Velocity')
    ax1.set_title('Optimal Race Strategy', fontsize=16)
    ax1.set_ylabel('Velocity (km/h)')
    ax1.grid(True)
    ax1.legend()

    # SOC
    ax2.plot(time_h, soc, marker='.', label='Battery SOC')
    ax2.axhline(y=p.MIN_SOC * 100.0, linestyle='--', label=f'Min SOC ({p.MIN_SOC:.0%})')
    ax2.set_ylabel('State of Charge (%)')
    ax2.set_ylim(0, 110)
    ax2.grid(True)
    ax2.legend()

    # Power Components
    ax3.plot(time_h, power_in, marker='.', label='Solar Power In')
    ax3.plot(time_h, power_out, marker='.', label='Motor Power Out')
    ax3.plot(time_h, net_power, linestyle='--', label='Net Power')
    ax3.set_ylabel('Power (Watts)')
    ax3.set_xlabel('Time (hours)')
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    output_filename = "race_strategy_results2.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\n[INFO] Result plots saved to '{os.path.abspath(output_filename)}'")
