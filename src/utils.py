import matplotlib.pyplot as plt

def set_plot_style():
    """
    Set matplotlib global style for all plots.
    """
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({'font.size': 16})
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.5
    plt.rcParams['axes.titlepad'] = 6.0
    plt.rcParams['axes.labelpad'] = 2.0
    plt.rcParams['axes.formatter.use_mathtext'] = True
    plt.rcParams['figure.figsize'] = [12, 6]
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.major.size'] = 5
    plt.rcParams['ytick.minor.size'] = 3
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['scatter.edgecolors'] = 'face'

def save_plot(fig, filename):
    """
    Save a matplotlib figure to disk and close it to free memory.
    """
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_electricity_price(time_index, electricity_price, time_list=None, start_time_nominal=None, can_charge=None, start_time_lower_bound=None, start_time_upper_bound=None):
    """
    Plot electricity price, with optional vertical lines for charge windows and time bounds.
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(time_index, electricity_price, label='Electricity Price')
    if time_list is not None and start_time_nominal is not None and can_charge is not None:
        for i in range(len(time_list)):
            if time_list[i] == 0:
                ax.axvline(x=start_time_nominal + time_list[i], color='b', linestyle='-')
            elif can_charge[(i-1) // 2]:
                ax.axvline(x=start_time_nominal + time_list[i], color='g', linestyle='-')
            else:
                ax.axvline(x=start_time_nominal + time_list[i], color='grey', linestyle='-')
                if i % 2 == 0:
                    ax.axvspan(start_time_nominal + time_list[i-1], start_time_nominal + time_list[i], facecolor='red', alpha=0.3)
            if time_list[i] != 0 and i % 2 == 0 and can_charge[(i-1) // 2]:
                ax.axvspan(start_time_nominal + time_list[i-1], start_time_nominal + time_list[i], facecolor='green', alpha=0.3)
    if start_time_lower_bound is not None and start_time_upper_bound is not None:
        ax.axvline(x=start_time_lower_bound, color='b', linestyle='--')
        ax.axvline(x=start_time_upper_bound, color='b', linestyle='--')
    ax.set_ylabel('Electricity Price\n($/kWh)')
    ax.set_xlabel('Time (minutes)')
    ax.set_xlim(0, 1440)
    ax.legend()
    plt.show()
    return fig

def plot_carbon_intensity(time_index, carbon_intensity, time_list=None, start_time_nominal=None, can_charge=None, start_time_lower_bound=None, start_time_upper_bound=None):
    """
    Plot grid carbon intensity, with optional vertical lines for charge windows and time bounds.
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(time_index, carbon_intensity, label='Grid Carbon Intensity')
    if time_list is not None and start_time_nominal is not None and can_charge is not None:
        for i in range(len(time_list)):
            if time_list[i] == 0:
                ax.axvline(x=start_time_nominal + time_list[i], color='b', linestyle='-')
            elif can_charge[(i-1) // 2]:
                ax.axvline(x=start_time_nominal + time_list[i], color='g', linestyle='-')
            else:
                ax.axvline(x=start_time_nominal + time_list[i], color='grey', linestyle='-')
                if i % 2 == 0:
                    ax.axvspan(start_time_nominal + time_list[i-1], start_time_nominal + time_list[i], facecolor='red', alpha=0.3)
            if time_list[i] != 0 and i % 2 == 0 and can_charge[(i-1) // 2]:
                ax.axvspan(start_time_nominal + time_list[i-1], start_time_nominal + time_list[i], facecolor='green', alpha=0.3)
    if start_time_lower_bound is not None and start_time_upper_bound is not None:
        ax.axvline(x=start_time_lower_bound, color='b', linestyle='--')
        ax.axvline(x=start_time_upper_bound, color='b', linestyle='--')
    ax.set_ylabel('Grid Carbon Intensity\n(kgCO$_2$/kWh)')
    ax.set_xlabel('Time (minutes)')
    ax.set_xlim(0, 1440)
    ax.legend()
    plt.show()
    return fig

def plot_remaining_energy(time_list, remaining_energy_list, start_time_nominal, can_charge, start_time_lower_bound, start_time_upper_bound, max_soc, min_soc):
    """
    Plot remaining battery energy with charge windows, SOC bounds, etc.
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot([start_time_nominal + t for t in time_list], remaining_energy_list, label='Remaining Energy')
    for i in range(len(time_list)):
        if time_list[i] == 0:
            ax.axvline(x=start_time_nominal + time_list[i], color='b', linestyle='-')
        elif can_charge[(i-1) // 2]:
            ax.axvline(x=start_time_nominal + time_list[i], color='g', linestyle='-')
        else:
            ax.axvline(x=start_time_nominal + time_list[i], color='grey', linestyle='-')
            if i % 2 == 0:
                ax.axvspan(start_time_nominal + time_list[i-1], start_time_nominal + time_list[i], facecolor='red', alpha=0.3)
        if time_list[i] != 0 and i % 2 == 0 and can_charge[(i-1) // 2]:
            ax.axvspan(start_time_nominal + time_list[i-1], start_time_nominal + time_list[i], facecolor='green', alpha=0.3)
    ax.axvline(x=start_time_lower_bound, color='b', linestyle='--')
    ax.axvline(x=start_time_upper_bound, color='b', linestyle='--')
    ax.axhline(y=max_soc, color='g', linestyle='--', label='Max SOC')
    ax.axhline(y=min_soc, color='orange', linestyle='--', label='Min SOC')
    ax.set_ylabel('Battery\nRemaining Energy (kWh)')
    ax.set_xlabel('Time (minutes)')
    ax.set_xlim(0, 1440)
    ax.legend()
    plt.show()
    return fig
