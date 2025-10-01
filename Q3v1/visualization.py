import matplotlib.pyplot as plt
import parameters as p
import os

def plot_results(simulation_history):
    """
    Generates and saves plots for the simulation results to the current directory.
    """
    history = simulation_history
    time_h = history['time']
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Velocity Profile
    ax1.plot(time_h, history['velocity'], 'b-', marker='.', label='Optimal Velocity')
    ax1.set_title('Optimal Race Strategy', fontsize=16)
    ax1.set_ylabel('Velocity (km/h)')
    ax1.grid(True)
    ax1.legend()
    
    # SOC
    ax2.plot(time_h, history['soc'], 'g-', marker='.', label='Battery SOC')
    ax2.axhline(y=p.MIN_SOC * 100, color='r', linestyle='--', label=f'Min SOC ({p.MIN_SOC:.0%})')
    ax2.set_ylabel('State of Charge (%)')
    ax2.set_ylim(0, 110)
    ax2.grid(True)
    ax2.legend()
    
    # Power Components
    ax3.plot(time_h, history['power_in'], 'y-', marker='.', label='Solar Power In')
    ax3.plot(time_h, history['power_out'], 'r-', marker='.', label='Motor Power Out')
    ax3.plot(time_h, history['net_power'], 'k--', label='Net Power')
    ax3.set_ylabel('Power (Watts)')
    ax3.set_xlabel('Time (hours)')
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    output_filename = "race_strategy_results1.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\n[INFO] Result plots saved to '{os.path.abspath(output_filename)}'")
    