import pandas as pd
import matplotlib.pyplot as plt

colors = ['black', 'dodgerblue', 'darkorange', 'limegreen', 'yellow', 'magenta', 'indigo', 'aqua', 'green', 'pink', 'brown', 'gray', 'navy']

env = "CartPole-v0"
n_steps_cycles = [600, 800, 1000, 1200]
n_steps = [1, 2, 3, 4, 5, 6, 7]

########################################################## (print DynDQN_n_T episode reward)
for n_steps_cycle in n_steps_cycles:

    # Create a figure for the plot
    plt.figure(figsize=(8, 5))

    for i, n_step in enumerate(n_steps):
        file_path = f"./results/{env}/Grid_search/DynDQN_n_T/Reward/Reward_DynDQN_n{n_step}_T{n_steps_cycle}.csv"
        data = pd.read_csv(file_path)

        # Calculate the smoothed Episode reward
        window_size = 10
        data['Smoothed Episode reward'] = data['Episode reward'].rolling(window=window_size, min_periods=1).mean()
        print(f"DynDQN_n{n_step}_T{n_steps_cycle} Last 15 epoch average reward: {data['Smoothed Episode reward'].tail(15).mean()}")
            
        # Plot the smoothed data
        plt.plot(data['Epoch number'], data['Smoothed Episode reward'], label=f'DynDQN_n{n_step}_T{n_steps_cycle}', linestyle='-', color=colors[i], linewidth=2)

    print("------------------------------------------------------------------------------")

    # Set the plot labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Average reward')
    plt.ylim(0, 350)
    plt.yticks(range(0, 350, 100))
    plt.title(env)
    plt.legend(loc="upper left")
    plt.grid(True)

    # Save the plot
    save_path = f"./results/{env}/Grid_search/DynDQN_n_T/n_steps_freq_{n_steps_cycle}.png"
    plt.savefig(save_path)

    plt.show()