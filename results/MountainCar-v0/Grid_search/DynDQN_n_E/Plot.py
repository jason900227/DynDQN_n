import pandas as pd
import matplotlib.pyplot as plt

colors = ['black', 'dodgerblue', 'darkorange', 'limegreen', 'yellow', 'magenta', 'indigo', 'aqua', 'green', 'pink', 'brown', 'gray', 'navy']

env = "MountainCar-v0"
n_steps_cycles = [15, 20, 25, 30]
n_steps = [1, 2, 3, 4, 5, 6, 7]

########################################################## (print DynDQN_n_E episode reward)
for n_steps_cycle in n_steps_cycles:

    # Create a figure for the plot
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, n_step in enumerate(n_steps):
        file_path = f"./results/{env}/Reward/Reward_DynDQN_n{n_step}_E{n_steps_cycle}.csv"
        data = pd.read_csv(file_path)

        # Calculate the smoothed Episode reward
        window_size = 10
        data['Smoothed Episode reward'] = data['Episode reward'].rolling(window=window_size, min_periods=1).mean()
        print(f"DynDQN_n{n_step}_E{n_steps_cycle} Last 15 epoch average reward: {data['Smoothed Episode reward'].tail(15).mean()}")
         
        # Plot the smoothed data
        ax.plot(data['Epoch number'], data['Smoothed Episode reward'], label=f'DynDQN_n{n_step}_E{n_steps_cycle}', linestyle='-', color=colors[i], linewidth=2)

    # Set the plot labels and title
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average reward')
    ax.set_ylim(-1100, -600)
    ax.set_yticks(range(-1100, -600, 100))
    ax.set_title(env)
    ax.grid(True)

    fig.subplots_adjust(top=0.8)

    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.4,
        fontsize=9,
        )
    
    print("------------------------------------------------------------------------------")

    # Save the plot
    save_path = f"./results/{env}/Grid_search/DynDQN_n_E/n_steps_cycles_{n_steps_cycle}.png"
    plt.savefig(save_path, bbox_inches='tight')

    plt.show()