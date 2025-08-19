import os
import pandas as pd

colors = ['red', 'darkorange', 'dodgerblue', 'limegreen', 'yellow', 'magenta', 'indigo', 'gray', 'aqua', 'green', 'black', 'brown', 'pink', 'navy']
env = "CartPole-v0"
seeds = [0, 10, 20, 40, 60, 80, 100]
n_steps = [1, 2, 3, 4, 5, 6, 7]
DynDQN_n_E_n_steps = [1, 2, 3, 4, 5, 6, 7]
DynDQN_n_E_n_steps_cycles = [15, 20, 25, 30]
DynDQN_n_T_n_steps = [1, 2, 3, 4, 5, 6, 7]
DynDQN_n_T_n_steps_cycles = [600, 800, 1000, 1200]
N_step = 3

########################################################## (merged multi steps DQN Episode reward)
for n_step in n_steps:
    episode_reward_data_list = []

    for seed in seeds:
        file_path = f"./results/{env}/DQN_n{n_step}/Evaluate(epoch)/Evaluate(epoch)_DQN_n{n_step}_seed{seed}.csv"
        data = pd.read_csv(file_path)

        # Extract 'Epoch number' and 'Episode reward', then rename the columns
        episode_rewards = data[['Epoch number', 'Episode reward']].rename(columns={'Episode reward': f'Episode reward seed_{seed}'})
        episode_reward_data_list.append(episode_rewards)

    # Initialize merged data with the first set of data
    merged_data = episode_reward_data_list[0]

    # Merge the other data sets based on 'Epoch number'
    for i in range(1, len(episode_reward_data_list)):
        merged_data = pd.merge(merged_data, episode_reward_data_list[i], on='Epoch number')

    # Calculate the average and std for each Epoch number
    merged_data['Episode reward'] = merged_data.iloc[:, 1:].mean(axis=1)
    merged_data['Episode reward std'] = merged_data.iloc[:, 1:].std(axis=1)

    # Save the merged data into a CSV file
    results_dir = "./results/{}/Reward".format(env)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    output_file_path = os.path.join(results_dir, f"Reward_DQN_n{n_step}.csv")
    merged_data.to_csv(output_file_path, index=False)

########################################################## (merged multi steps DQN Cv Q value)
for n_step in n_steps:
    cv_q_value_data_list = []

    for seed in seeds:
        file_path = f"./results/{env}/DQN_n{n_step}/Training(episode)/Training(episode)_DQN_n{n_step}_seed{seed}.csv"
        data = pd.read_csv(file_path)

        # Extract 'Episode time step' and 'Cv Q value', then rename the columns
        cv_q_values = data[['Episode time step', 'Cv Q value']].rename(columns={'Cv Q value': f'Cv Q value seed_{seed}'})
        cv_q_value_data_list.append(cv_q_values)

    # Initialize merged data with the first set of data
    merged_data = cv_q_value_data_list[0]

    # Merge the other data sets based on 'Episode time step'
    for i in range(1, len(cv_q_value_data_list)):
        merged_data = pd.merge(merged_data, cv_q_value_data_list[i], on='Episode time step', how='outer')

    # Calculate the average for each Episode time step
    merged_data['Cv Q value'] = merged_data.iloc[:, 1:].mean(axis=1)

    # Sort Episode time step to ensure the data is in correct order
    merged_data = merged_data.sort_values(by='Episode time step')

    # Save the merged data into a CSV file
    results_dir = "./results/{}/Cv_q_value".format(env)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    output_file_path = os.path.join(results_dir, f"Cv_q_value_DQN_n{n_step}.csv")
    merged_data.to_csv(output_file_path, index=False)

########################################################## (merged DynDQN_n_E Episode reward)
for DynDQN_n_E_n_step in DynDQN_n_E_n_steps:
    for DynDQN_n_E_n_steps_cycle in DynDQN_n_E_n_steps_cycles:
        episode_reward_data_list = []

        for seed in seeds:
            file_path = f"./results/{env}/DynDQN_n_E/DynDQN_n{DynDQN_n_E_n_step}_E{DynDQN_n_E_n_steps_cycle}/Evaluate(epoch)/Evaluate(epoch)_DynDQN_n{DynDQN_n_E_n_step}_E{DynDQN_n_E_n_steps_cycle}_seed{seed}.csv"
            data = pd.read_csv(file_path)

            # Extract 'Epoch number' and 'Episode reward', then rename the columns
            episode_rewards = data[['Epoch number', 'Episode reward']].rename(columns={'Episode reward': f'Episode reward seed_{seed}'})
            episode_reward_data_list.append(episode_rewards)

        # Initialize merged data with the first set of data
        merged_data = episode_reward_data_list[0]

        # Merge the other data sets based on 'Epoch number'
        for i in range(1, len(episode_reward_data_list)):
            merged_data = pd.merge(merged_data, episode_reward_data_list[i], on='Epoch number')

        # Calculate the average and std for each Epoch number
        merged_data['Episode reward'] = merged_data.iloc[:, 1:].mean(axis=1)
        merged_data['Episode reward std'] = merged_data.iloc[:, 1:].std(axis=1)

        # Save the merged data into a CSV file
        results_dir = "./results/{}/Reward".format(env)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        output_file_path = os.path.join(results_dir, f"Reward_DynDQN_n{DynDQN_n_E_n_step}_E{DynDQN_n_E_n_steps_cycle}.csv")
        merged_data.to_csv(output_file_path, index=False)

########################################################## (merged DynDQN_n_E Cv Q value)
for DynDQN_n_E_n_step in DynDQN_n_E_n_steps:
    for DynDQN_n_E_n_steps_cycle in DynDQN_n_E_n_steps_cycles:
        cv_q_value_data_list = []

        for seed in seeds:
            file_path = f"./results/{env}/DynDQN_n_E/DynDQN_n{DynDQN_n_E_n_step}_E{DynDQN_n_E_n_steps_cycle}/Training(episode)/Training(episode)_DynDQN_n{DynDQN_n_E_n_step}_E{DynDQN_n_E_n_steps_cycle}_seed{seed}.csv"
            data = pd.read_csv(file_path)

            # Extract 'Episode time step' and 'Cv Q value', then rename the columns
            cv_q_values = data[['Episode time step', 'Cv Q value']].rename(columns={'Cv Q value': f'Cv Q value seed_{seed}'})
            cv_q_value_data_list.append(cv_q_values)

        # Initialize merged data with the first set of data
        merged_data = cv_q_value_data_list[0]

        # Merge the other data sets based on 'Episode time step'
        for i in range(1, len(cv_q_value_data_list)):
            merged_data = pd.merge(merged_data, cv_q_value_data_list[i], on='Episode time step', how='outer')

        # Calculate the average for each Episode time step
        merged_data['Cv Q value'] = merged_data.iloc[:, 1:].mean(axis=1)

        # Sort Episode time step to ensure the data is in correct order
        merged_data = merged_data.sort_values(by='Episode time step')

        # Save the merged data into a CSV file
        results_dir = "./results/{}/Cv_q_value".format(env)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        output_file_path = os.path.join(results_dir, f"Cv_q_value_DynDQN_n{DynDQN_n_E_n_step}_E{DynDQN_n_E_n_steps_cycle}.csv")
        merged_data.to_csv(output_file_path, index=False)

########################################################## (merged DynDQN_n_E n_steps)
for DynDQN_n_E_n_step in DynDQN_n_E_n_steps:
    for DynDQN_n_E_n_steps_cycle in DynDQN_n_E_n_steps_cycles:
        n_steps_data_list = []

        for seed in seeds:
            file_path = f"./results/{env}/DynDQN_n_E/DynDQN_n{DynDQN_n_E_n_step}_E{DynDQN_n_E_n_steps_cycle}/Evaluate(epoch)/Evaluate(epoch)_DynDQN_n{DynDQN_n_E_n_step}_E{DynDQN_n_E_n_steps_cycle}_seed{seed}.csv"
            data = pd.read_csv(file_path)

            # Extract 'Epoch number' and 'N steps', then rename the columns
            n_steps = data[['Epoch number', 'N steps']].rename(columns={'N steps': f'N steps seed_{seed}'})
            n_steps_data_list.append(n_steps)

        # Initialize merged data with the first set of data
        merged_data = n_steps_data_list[0]

        # Merge the other data sets based on 'Epoch number'
        for i in range(1, len(n_steps_data_list)):
            merged_data = pd.merge(merged_data, n_steps_data_list[i], on='Epoch number', how='outer')

        # Calculate the average for each Epoch number
        merged_data['N steps'] = merged_data.iloc[:, 1:].mean(axis=1)

        # Sort Epoch number to ensure the data is in correct order
        merged_data = merged_data.sort_values(by='Epoch number')

        # Save the merged data into a CSV file
        results_dir = "./results/{}/N_steps".format(env)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        output_file_path = os.path.join(results_dir, f"N_steps_DynDQN_n{DynDQN_n_E_n_step}_E{DynDQN_n_E_n_steps_cycle}.csv")
        merged_data.to_csv(output_file_path, index=False)

########################################################## (merged DynDQN_n_T Episode reward)
for DynDQN_n_T_n_step in DynDQN_n_T_n_steps:
    for DynDQN_n_T_n_steps_cycle in DynDQN_n_T_n_steps_cycles:
        episode_reward_data_list = []

        for seed in seeds:
            file_path = f"./results/{env}/DynDQN_n_T/DynDQN_n{DynDQN_n_T_n_step}_T{DynDQN_n_T_n_steps_cycle}/Evaluate(epoch)/Evaluate(epoch)_DynDQN_n{DynDQN_n_T_n_step}_T{DynDQN_n_T_n_steps_cycle}_seed{seed}.csv"
            data = pd.read_csv(file_path)

            # Extract 'Epoch number' and 'Episode reward', then rename the columns
            episode_rewards = data[['Epoch number', 'Episode reward']].rename(columns={'Episode reward': f'Episode reward seed_{seed}'})
            episode_reward_data_list.append(episode_rewards)

        # Initialize merged data with the first set of data
        merged_data = episode_reward_data_list[0]

        # Merge the other data sets based on 'Epoch number'
        for i in range(1, len(episode_reward_data_list)):
            merged_data = pd.merge(merged_data, episode_reward_data_list[i], on='Epoch number')

        # Calculate the average and std for each Epoch number
        merged_data['Episode reward'] = merged_data.iloc[:, 1:].mean(axis=1)
        merged_data['Episode reward std'] = merged_data.iloc[:, 1:].std(axis=1)

        # Save the merged data into a CSV file
        results_dir = "./results/{}/Reward".format(env)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        output_file_path = os.path.join(results_dir, f"Reward_DynDQN_n{DynDQN_n_T_n_step}_T{DynDQN_n_T_n_steps_cycle}.csv")
        merged_data.to_csv(output_file_path, index=False)

########################################################## (merged DynDQN_n_T Cv Q value)
for DynDQN_n_T_n_step in DynDQN_n_T_n_steps:
    for DynDQN_n_T_n_steps_cycle in DynDQN_n_T_n_steps_cycles:
        cv_q_value_data_list = []

        for seed in seeds:
            file_path = f"./results/{env}/DynDQN_n_T/DynDQN_n{DynDQN_n_T_n_step}_T{DynDQN_n_T_n_steps_cycle}/Training(episode)/Training(episode)_DynDQN_n{DynDQN_n_T_n_step}_T{DynDQN_n_T_n_steps_cycle}_seed{seed}.csv"
            data = pd.read_csv(file_path)

            # Extract 'Episode time step' and 'Cv Q value', then rename the columns
            cv_q_values = data[['Episode time step', 'Cv Q value']].rename(columns={'Cv Q value': f'Cv Q value seed_{seed}'})
            cv_q_value_data_list.append(cv_q_values)

        # Initialize merged data with the first set of data
        merged_data = cv_q_value_data_list[0]

        # Merge the other data sets based on 'Episode time step'
        for i in range(1, len(cv_q_value_data_list)):
            merged_data = pd.merge(merged_data, cv_q_value_data_list[i], on='Episode time step', how='outer')

        # Calculate the average for each Episode time step
        merged_data['Cv Q value'] = merged_data.iloc[:, 1:].mean(axis=1)

        # Sort Episode time step to ensure the data is in correct order
        merged_data = merged_data.sort_values(by='Episode time step')

        # Save the merged data into a CSV file
        results_dir = "./results/{}/Cv_q_value".format(env)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        output_file_path = os.path.join(results_dir, f"Cv_q_value_DynDQN_n{DynDQN_n_T_n_step}_T{DynDQN_n_T_n_steps_cycle}.csv")
        merged_data.to_csv(output_file_path, index=False)

########################################################## (merged DynDQN_n_T n_steps)
for DynDQN_n_T_n_step in DynDQN_n_T_n_steps:
    for DynDQN_n_T_n_steps_cycle in DynDQN_n_T_n_steps_cycles:
        n_steps_data_list = []

        for seed in seeds:
            file_path = f"./results/{env}/DynDQN_n_T/DynDQN_n{DynDQN_n_T_n_step}_T{DynDQN_n_T_n_steps_cycle}/Evaluate(epoch)/Evaluate(epoch)_DynDQN_n{DynDQN_n_T_n_step}_T{DynDQN_n_T_n_steps_cycle}_seed{seed}.csv"
            data = pd.read_csv(file_path)

            # Extract 'Epoch number' and 'N steps', then rename the columns
            n_steps = data[['Epoch number', 'N steps']].rename(columns={'N steps': f'N steps seed_{seed}'})
            n_steps_data_list.append(n_steps)

        # Initialize merged data with the first set of data
        merged_data = n_steps_data_list[0]

        # Merge the other data sets based on 'Epoch number'
        for i in range(1, len(n_steps_data_list)):
            merged_data = pd.merge(merged_data, n_steps_data_list[i], on='Epoch number', how='outer')

        # Calculate the average for each Epoch number
        merged_data['N steps'] = merged_data.iloc[:, 1:].mean(axis=1)

        # Sort Epoch number to ensure the data is in correct order
        merged_data = merged_data.sort_values(by='Epoch number')

        # Save the merged data into a CSV file
        results_dir = "./results/{}/N_steps".format(env)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        output_file_path = os.path.join(results_dir, f"N_steps_DynDQN_n{DynDQN_n_T_n_step}_T{DynDQN_n_T_n_steps_cycle}.csv")
        merged_data.to_csv(output_file_path, index=False)

########################################################## (merged ESDQN Episode reward)
episode_reward_data_list = []

for seed in seeds:
    file_path = f"./results/{env}/ESDQN/Evaluate(epoch)/Evaluate(epoch)_ESDQN_seed{seed}.csv"
    data = pd.read_csv(file_path)

    # Extract 'Epoch number' and 'Episode reward', then rename the columns
    episode_rewards = data[['Epoch number', 'Episode reward']].rename(columns={'Episode reward': f'Episode reward seed_{seed}'})
    episode_reward_data_list.append(episode_rewards)

# Initialize merged data with the first set of data
merged_data = episode_reward_data_list[0]

# Merge the other data sets based on 'Epoch number'
for i in range(1, len(episode_reward_data_list)):
    merged_data = pd.merge(merged_data, episode_reward_data_list[i], on='Epoch number')

# Calculate the average and std for each Epoch number
merged_data['Episode reward'] = merged_data.iloc[:, 1:].mean(axis=1)
merged_data['Episode reward std'] = merged_data.iloc[:, 1:].std(axis=1)

# Save the merged data into a CSV file
results_dir = "./results/{}/Reward".format(env)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

output_file_path = os.path.join(results_dir, "Reward_ESDQN.csv")
merged_data.to_csv(output_file_path, index=False)

########################################################## (merged ESDQN Cv Q value)
cv_q_value_data_list = []

for seed in seeds:
    file_path = f"./results/{env}/ESDQN/Training(episode)/Training(episode)_ESDQN_seed{seed}.csv"
    data = pd.read_csv(file_path)

    # Extract 'Episode time step' and 'Cv Q value', then rename the columns
    cv_q_values = data[['Episode time step', 'Cv Q value']].rename(columns={'Cv Q value': f'Cv Q value seed_{seed}'})
    cv_q_value_data_list.append(cv_q_values)

# Initialize merged data with the first set of data
merged_data = cv_q_value_data_list[0]

# Merge the other data sets based on 'Episode time step'
for i in range(1, len(cv_q_value_data_list)):
    merged_data = pd.merge(merged_data, cv_q_value_data_list[i], on='Episode time step', how='outer')

# Calculate the average for each Episode time step
merged_data['Cv Q value'] = merged_data.iloc[:, 1:].mean(axis=1)

# Sort Episode time step to ensure the data is in correct order
merged_data = merged_data.sort_values(by='Episode time step')

# Save the merged data into a CSV file
results_dir = "./results/{}/Cv_q_value".format(env)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

output_file_path = os.path.join(results_dir, "Cv_q_value_ESDQN.csv")
merged_data.to_csv(output_file_path, index=False)

########################################################## (merged ESDQN Mean silhouette score)
mean_silhouette_score_data_list = []

for seed in seeds:
    file_path = f"./results/{env}/ESDQN/Training(episode)/Training(episode)_ESDQN_seed{seed}.csv"
    data = pd.read_csv(file_path)

    # Extract 'Episode time step' and 'Mean silhouette score', then rename the columns
    mean_silhouette_scores = data[['Episode time step', 'Mean silhouette score']].rename(columns={'Mean silhouette score': f'Mean silhouette score seed_{seed}'})
    mean_silhouette_score_data_list.append(mean_silhouette_scores)

# Initialize merged data with the first set of data
merged_data = mean_silhouette_score_data_list[0]

# Merge the other data sets based on 'Episode time step'
for i in range(1, len(mean_silhouette_score_data_list)):
    merged_data = pd.merge(merged_data, mean_silhouette_score_data_list[i], on='Episode time step', how='outer')

# Calculate the average for each Episode time step
merged_data['Mean silhouette score'] = merged_data.iloc[:, 1:].mean(axis=1)

# Sort Episode time step to ensure the data is in correct order
merged_data = merged_data.sort_values(by='Episode time step')

# Save the merged data into a CSV file
results_dir = "./results/{}/Mean_silhouette_score".format(env)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

output_file_path = os.path.join(results_dir, "Mean_silhouette_score_ESDQN.csv")
merged_data.to_csv(output_file_path, index=False)

########################################################## (merged DQN_LNSS Episode reward)
episode_reward_data_list = []

for seed in seeds:
    file_path = f"./results/{env}/DQN_LNSS_N{N_step}/Evaluate(epoch)/Evaluate(epoch)_DQN_LNSS_N{N_step}_seed{seed}.csv"
    data = pd.read_csv(file_path)

    # Extract 'Epoch number' and 'Episode reward', then rename the columns
    episode_rewards = data[['Epoch number', 'Episode reward']].rename(columns={'Episode reward': f'Episode reward seed_{seed}'})
    episode_reward_data_list.append(episode_rewards)

# Initialize merged data with the first set of data
merged_data = episode_reward_data_list[0]

# Merge the other data sets based on 'Epoch number'
for i in range(1, len(episode_reward_data_list)):
    merged_data = pd.merge(merged_data, episode_reward_data_list[i], on='Epoch number')

# Calculate the average and std for each Epoch number
merged_data['Episode reward'] = merged_data.iloc[:, 1:].mean(axis=1)
merged_data['Episode reward std'] = merged_data.iloc[:, 1:].std(axis=1)

# Save the merged data into a CSV file
results_dir = "./results/{}/Reward".format(env)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

output_file_path = os.path.join(results_dir, f"Reward_DQN_LNSS_N{N_step}.csv")
merged_data.to_csv(output_file_path, index=False)

########################################################## (merged DQN_LNSS Cv Q value)
cv_q_value_data_list = []

for seed in seeds:
    file_path = f"./results/{env}/DQN_LNSS_N{N_step}/Training(episode)/Training(episode)_DQN_LNSS_N{N_step}_seed{seed}.csv"
    data = pd.read_csv(file_path)

    # Extract 'Episode time step' and 'Cv Q value', then rename the columns
    cv_q_values = data[['Episode time step', 'Cv Q value']].rename(columns={'Cv Q value': f'Cv Q value seed_{seed}'})
    cv_q_value_data_list.append(cv_q_values)

# Initialize merged data with the first set of data
merged_data = cv_q_value_data_list[0]

# Merge the other data sets based on 'Episode time step'
for i in range(1, len(cv_q_value_data_list)):
    merged_data = pd.merge(merged_data, cv_q_value_data_list[i], on='Episode time step', how='outer')

# Calculate the average for each Episode time step
merged_data['Cv Q value'] = merged_data.iloc[:, 1:].mean(axis=1)

# Sort Episode time step to ensure the data is in correct order
merged_data = merged_data.sort_values(by='Episode time step')

# Save the merged data into a CSV file
results_dir = "./results/{}/Cv_q_value".format(env)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

output_file_path = os.path.join(results_dir, f"Cv_q_value_DQN_LNSS_N{N_step}.csv")
merged_data.to_csv(output_file_path, index=False)

########################################################## (merged MMDQN Episode reward)
episode_reward_data_list = []

for seed in seeds:
    file_path = f"./results/{env}/MMDQN_n{N_step}/Evaluate(epoch)/Evaluate(epoch)_MMDQN_n{N_step}_seed{seed}.csv"
    data = pd.read_csv(file_path)

    # Extract 'Epoch number' and 'Episode reward', then rename the columns
    episode_rewards = data[['Epoch number', 'Episode reward']].rename(columns={'Episode reward': f'Episode reward seed_{seed}'})
    episode_reward_data_list.append(episode_rewards)

# Initialize merged data with the first set of data
merged_data = episode_reward_data_list[0]

# Merge the other data sets based on 'Epoch number'
for i in range(1, len(episode_reward_data_list)):
    merged_data = pd.merge(merged_data, episode_reward_data_list[i], on='Epoch number')

# Calculate the average and std for each Epoch number
merged_data['Episode reward'] = merged_data.iloc[:, 1:].mean(axis=1)
merged_data['Episode reward std'] = merged_data.iloc[:, 1:].std(axis=1)

# Save the merged data into a CSV file
results_dir = "./results/{}/Reward".format(env)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

output_file_path = os.path.join(results_dir, f"Reward_MMDQN_n{N_step}.csv")
merged_data.to_csv(output_file_path, index=False)

########################################################## (merged MMDQN Cv Q value)
cv_q_value_data_list = []

for seed in seeds:
    file_path = f"./results/{env}/MMDQN_n{N_step}/Training(episode)/Training(episode)_MMDQN_n{N_step}_seed{seed}.csv"
    data = pd.read_csv(file_path)

    # Extract 'Episode time step' and 'Cv Q value', then rename the columns
    cv_q_values = data[['Episode time step', 'Cv Q value']].rename(columns={'Cv Q value': f'Cv Q value seed_{seed}'})
    cv_q_value_data_list.append(cv_q_values)

# Initialize merged data with the first set of data
merged_data = cv_q_value_data_list[0]

# Merge the other data sets based on 'Episode time step'
for i in range(1, len(cv_q_value_data_list)):
    merged_data = pd.merge(merged_data, cv_q_value_data_list[i], on='Episode time step', how='outer')

# Calculate the average for each Episode time step
merged_data['Cv Q value'] = merged_data.iloc[:, 1:].mean(axis=1)

# Sort Episode time step to ensure the data is in correct order
merged_data = merged_data.sort_values(by='Episode time step')

# Save the merged data into a CSV file
results_dir = "./results/{}/Cv_q_value".format(env)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

output_file_path = os.path.join(results_dir, f"Cv_q_value_MMDQN_n{N_step}.csv")
merged_data.to_csv(output_file_path, index=False)