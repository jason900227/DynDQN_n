# Dynamic Multi-step Deep Reinforcement Learning Based on Q-value Coefficient of Variation for Improved Sample Efficiency
![Python](https://img.shields.io/badge/Python-3.8.20-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1-red)
![CUDA](https://img.shields.io/badge/CUDA-11.7-orange)
![License](https://img.shields.io/badge/License-MIT-green)

This repository contains the implementation of **Dynamic Multi-step Deep Reinforcement Learning Based on Q-value Coefficient of Variation for Improved Sample Efficiency**, and compares the following multi-step TD methods: fixed-n baselines (DQN_n1 to DQN_n7), fixed-n extensions (EnDQN, LNSS, MMDQN), and dynamic-n approaches (ESDQN, DynDQN_n_E, DynDQN_n_T), compared across classic control environments in OpenAI Gym to assess sample efficiency and learning stability.

## 1. Environment
* **OS**: Windows 10/11  
* **Python**: 3.8.20  
* **PyTorch**: 1.13.1
* **CUDA**: 11.7
* **Gym**: 0.26.2

## 2. Installation
### Create New Conda Environment
  ```
  # Create environment with Python 3.8.20
  conda create --name DynDQN_n python=3.8.20 -y
  
  # Activate environment
  conda activate DynDQN_n
  
  # Install PyTorch 1.13.1 with CUDA 11.7
  conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
  
  # Install other dependencies
  pip install -r requirements.txt
  ```

## 3. Supported Algorithms
### 3.1 Multi-step TD Method with Fixed n (Baseline)
  * DQN_n1, DQN_n2, ..., DQN_n7: The subscript `n` indicates the number of steps used for the multi-step TD updates.
### 3.2 Multi-step TD Method with Fixed n Extension
  * DQN_LNSS: Using an discounted weighted reward over a fixed n-step as the multi-step TD updates target reward.
  * MMDQN: Using the average of multiple n-step TD targets as the multi-step TD updates target.
### 3.3 Multi-step TD Method with Dynamic n
  * ESDQN: Uses the clustering algorithm (HDBSCAN) to identify state similarity and dynamically adjust the return length `n` in multi-step TD updates.
  * DynDQN_n_E: Uses the coefficient of variation (CV) of Q-values to dynamically adjust the return length `n` in multi-step TD updates every few episodes.
  * DynDQN_n_T: Uses the coefficient of variation (CV) of Q-values to dynamically adjust the return length `n` in multi-step TD updates every few timesteps.

##  4. Usage Examples
### 4.1 Run algorithm
  ```
  # Run
  python Multi_step_DQN.py
  python DQN_LNSS.py
  python MMDQN.py
  python ESDQN.py
  python DynDQN_n_E.py
  python DynDQN_n_T.py
  ```
> Note:
> All hyperparameters are defined inside each algorithm's Python script in the `argparse` section.
> - Common parameters include `hidden_dim`, `lr`, `target_update_freq`, `max_training_steps`, `evaluate_freq`, `epsilon_decay_steps`, `n_steps`, etc.
> - Environment-specific settings (`hidden_dim`, `lr`, `target_update_freq`, `max_training_steps`, `evaluate_freq`, `epsilon_decay_steps`) are provided in comments above the `env_names` list in the script.
> - To modify parameters for a specific method, open the corresponding `.py` file (e.g., `Multi_step_DQN.py`, `EnDQN.py`, `LNSS.py`, etc.) and edit the `parser.add_argument(...)` values directly.
