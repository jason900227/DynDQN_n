import pandas as pd
import matplotlib.pyplot as plt

colors = ['black', 'dodgerblue', 'darkorange', 'limegreen', 'yellow', 'magenta', 'indigo', 'aqua', 'green', 'pink', 'brown', 'gray', 'navy']
env = "Acrobot-v1"
n_steps = [1, 2, 3, 4, 5, 6, 7]
DynDQN_n_E_n_step = 2
DynDQN_n_E_n_steps_cycle = 15
DynDQN_n_T_n_step = 2
DynDQN_n_T_n_steps_cycle = 1200
N_step = 3

########################################################## (print multi steps DQN episode reward)

# Create a figure for the plot
fig, ax = plt.subplots(figsize=(8, 5))

for i, n_step in enumerate(n_steps):
    file_path = f"./results/{env}/Reward/Reward_DQN_n{n_step}.csv"
    data = pd.read_csv(file_path)

    # Calculate the smoothed Episode reward and std
    window_size = 10
    data['Smoothed Episode reward'] = data['Episode reward'].rolling(window=window_size, min_periods=1).mean()
    print(f"DQN_n{n_step} Last 15 epoch average reward: {data['Smoothed Episode reward'].tail(15).mean()}")

    data['Smoothed Episode reward std'] = data['Episode reward std'].rolling(window=window_size, min_periods=1).mean()
    print(f"DQN_n{n_step} Last 15 epoch average reward std: {data['Smoothed Episode reward std'].tail(15).mean()}\n")
        
    # Plot the smoothed data
    ax.plot(data['Epoch number'], data['Smoothed Episode reward'], label=f'DQN_n{n_step}', linestyle='-', color=colors[i], linewidth=2)

########################################################## (print ESDQN episode reward)
file_path = f"./results/{env}/Reward/Reward_ESDQN.csv"
data = pd.read_csv(file_path)

# Calculate the smoothed Episode reward and std
window_size = 10
data['Smoothed Episode reward'] = data['Episode reward'].rolling(window=window_size, min_periods=1).mean()
print(f"ESDQN Last 15 epoch average reward: {data['Smoothed Episode reward'].tail(15).mean()}")

data['Smoothed Episode reward std'] = data['Episode reward std'].rolling(window=window_size, min_periods=1).mean()
print(f"ESDQN Last 15 epoch average reward std: {data['Smoothed Episode reward std'].tail(15).mean()}\n")
    
# Plot the smoothed data
ax.plot(data['Epoch number'], data['Smoothed Episode reward'], label='ESDQN', linestyle='-', color='red', linewidth=2)

########################################################## (print DynDQN_n_E episode reward)
file_path = f"./results/{env}/Reward/Reward_DynDQN_n{DynDQN_n_E_n_step}_E{DynDQN_n_E_n_steps_cycle}.csv"
data = pd.read_csv(file_path)

# Calculate the smoothed Episode reward and std
window_size = 10
data['Smoothed Episode reward'] = data['Episode reward'].rolling(window=window_size, min_periods=1).mean()
print(f"DynDQN_n{DynDQN_n_E_n_step}_E{DynDQN_n_E_n_steps_cycle} Last 15 epoch average reward: {data['Smoothed Episode reward'].tail(15).mean()}")

data['Smoothed Episode reward std'] = data['Episode reward std'].rolling(window=window_size, min_periods=1).mean()
print(f"DynDQN_n{DynDQN_n_E_n_step}_E{DynDQN_n_E_n_steps_cycle} Last 15 epoch average reward std: {data['Smoothed Episode reward std'].tail(15).mean()}\n")
    
# Plot the smoothed data
ax.plot(data['Epoch number'], data['Smoothed Episode reward'], label=f'DynDQN_n{DynDQN_n_E_n_step}_E{DynDQN_n_E_n_steps_cycle}', linestyle='-', color=colors[i+1], linewidth=2)
 
########################################################## (print DynDQN_n_T episode reward)
file_path = f"./results/{env}/Reward/Reward_DynDQN_n{DynDQN_n_T_n_step}_T{DynDQN_n_T_n_steps_cycle}.csv"
data = pd.read_csv(file_path)

# Calculate the smoothed Episode reward and std
window_size = 10
data['Smoothed Episode reward'] = data['Episode reward'].rolling(window=window_size, min_periods=1).mean()
print(f"DynDQN_n{DynDQN_n_T_n_step}_E{DynDQN_n_T_n_steps_cycle} Last 15 epoch average reward: {data['Smoothed Episode reward'].tail(15).mean()}")

data['Smoothed Episode reward std'] = data['Episode reward std'].rolling(window=window_size, min_periods=1).mean()
print(f"DynDQN_n{DynDQN_n_T_n_step}_E{DynDQN_n_T_n_steps_cycle} Last 15 epoch average reward std: {data['Smoothed Episode reward std'].tail(15).mean()}\n")
    
# Plot the smoothed data
ax.plot(data['Epoch number'], data['Smoothed Episode reward'], label=f'DynDQN_n{DynDQN_n_T_n_step}_E{DynDQN_n_T_n_steps_cycle}', linestyle='-', color=colors[i+2], linewidth=2)

########################################################## (print DQN_LNSS episode reward)
file_path = f"./results/{env}/Reward/Reward_DQN_LNSS_N{N_step}.csv"
data = pd.read_csv(file_path)

# Calculate the smoothed Episode reward and std
window_size = 10
data['Smoothed Episode reward'] = data['Episode reward'].rolling(window=window_size, min_periods=1).mean()
print(f"DQN_LNSS Last 15 epoch average reward: {data['Smoothed Episode reward'].tail(15).mean()}")

data['Smoothed Episode reward std'] = data['Episode reward std'].rolling(window=window_size, min_periods=1).mean()
print(f"DQN_LNSS Last 15 epoch average reward std: {data['Smoothed Episode reward std'].tail(15).mean()}\n")
    
# Plot the smoothed data
ax.plot(data['Epoch number'], data['Smoothed Episode reward'], label=f'DQN_LNSS', linestyle='-', color=colors[i+3], linewidth=2)

########################################################## (print MMDQN episode reward)
file_path = f"./results/{env}/Reward/Reward_MMDQN_n{N_step}.csv"
data = pd.read_csv(file_path)

# Calculate the smoothed Episode reward and std
window_size = 10
data['Smoothed Episode reward'] = data['Episode reward'].rolling(window=window_size, min_periods=1).mean()
print(f"MMDQN Last 15 epoch average reward: {data['Smoothed Episode reward'].tail(15).mean()}")

data['Smoothed Episode reward std'] = data['Episode reward std'].rolling(window=window_size, min_periods=1).mean()
print(f"MMDQN Last 15 epoch average reward std: {data['Smoothed Episode reward std'].tail(15).mean()}")
    
# Plot the smoothed data
ax.plot(data['Epoch number'], data['Smoothed Episode reward'], label=f'MMDQN', linestyle='-', color=colors[i+4], linewidth=2)

# Set the plot labels and title
ax.set_xlabel('Epoch')
ax.set_ylabel('Average reward')
ax.set_ylim(-1100, -100)
ax.set_yticks(range(-1100, -100, 100))
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

# Save the plot
save_path = f"./results/{env}/Reward.png"
plt.savefig(save_path, bbox_inches='tight')

plt.show()

########################################################## (print ESDQN Cv Q value)

# Create a figure for the plot
plt.figure(figsize=(8, 5))

file_path = f"./results/{env}/Cv_q_value/Cv_q_value_ESDQN.csv"
data = pd.read_csv(file_path)

# Take the absolute value of Cv Q value
data['Cv Q value'] = data['Cv Q value'].abs()

# Plot the original data
plt.plot(data['Episode time step'], data['Cv Q value'], label='ESDQN', linestyle='-', color='red', linewidth=2)

########################################################## (print DynDQN_n_E Cv Q value)
file_path = f"./results/{env}/Cv_q_value/Cv_q_value_DynDQN_n{DynDQN_n_E_n_step}_E{DynDQN_n_E_n_steps_cycle}.csv"
data = pd.read_csv(file_path)

# Take the absolute value of Cv Q value
data['Cv Q value'] = data['Cv Q value'].abs()

# Plot the original data
plt.plot(data['Episode time step'], data['Cv Q value'], label=f'DynDQN_n{DynDQN_n_E_n_step}_E{DynDQN_n_E_n_steps_cycle}', linestyle='-', color=colors[i+1], linewidth=2)

########################################################## (print DynDQN_n_T Cv Q value)
file_path = f"./results/{env}/Cv_q_value/Cv_q_value_DynDQN_n{DynDQN_n_T_n_step}_T{DynDQN_n_T_n_steps_cycle}.csv"
data = pd.read_csv(file_path)

# Take the absolute value of Cv Q value
data['Cv Q value'] = data['Cv Q value'].abs()

# Plot the original data
plt.plot(data['Episode time step'], data['Cv Q value'], label=f'DynDQN_n{DynDQN_n_T_n_step}_T{DynDQN_n_T_n_steps_cycle}', linestyle='-', color=colors[i+2], linewidth=2)

# Set the plot labels and title
plt.xlabel('Time steps')
plt.ylabel('Cv Q value')
plt.ylim(0, 0.5)
plt.title(env)
plt.legend(loc="upper right")
plt.grid(True)

# Save the plot
save_path = f"./results/{env}/Cv_q_value.png"
plt.savefig(save_path)

plt.show()

########################################################## (print ESDQN Mean silhouette score)

# Create a figure for the plot
plt.figure(figsize=(8, 5))

file_path = f"./results/{env}/Mean_silhouette_score/Mean_silhouette_score_ESDQN.csv"
data = pd.read_csv(file_path)

plt.plot(data['Episode time step'], data['Mean silhouette score'], label='ESDQN', linestyle='-', color="red", linewidth=2)

# Set the plot labels and title
plt.xlabel('Time steps')
plt.ylabel('Mean silhouette score')
plt.title(env)
plt.legend(loc="upper left")
plt.grid(True)

# Save the plot
save_path = f"./results/{env}/Mean_silhouette_score.png"
plt.savefig(save_path)

plt.show()

########################################################## (print DynDQN_n_E n_steps)

# Create a figure for the plot
plt.figure(figsize=(8, 5))

file_path = f"./results/{env}/N_steps/N_steps_DynDQN_n{DynDQN_n_E_n_step}_E{DynDQN_n_E_n_steps_cycle}.csv"
data = pd.read_csv(file_path)

plt.plot(data['Epoch number'], data['N steps'], label=f'DynDQN_n{DynDQN_n_E_n_step}_E{DynDQN_n_E_n_steps_cycle}', linestyle='-', color=colors[i+1], linewidth=2)

########################################################## (print DynDQN_n_T n_steps)
file_path = f"./results/{env}/N_steps/N_steps_DynDQN_n{DynDQN_n_T_n_step}_T{DynDQN_n_T_n_steps_cycle}.csv"
data = pd.read_csv(file_path)

plt.plot(data['Epoch number'], data['N steps'], label=f'DynDQN_n{DynDQN_n_T_n_step}_T{DynDQN_n_T_n_steps_cycle}', linestyle='-', color=colors[i+2], linewidth=2)

# Set the plot labels and title
plt.xlabel('Epoch')
plt.ylabel('n steps')
plt.title(env)
plt.legend(loc="upper left")
plt.grid(True)

# Save the plot
save_path = f"./results/{env}/N_steps.png"
plt.savefig(save_path)

plt.show()