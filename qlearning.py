import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset (synthetic page replacement data)
df = pd.read_csv("synthetic_page_replacement_data.csv")
pages = df['Current_Page'].tolist()

# Parameters for the simulation
num_frames = 3  # Number of frames in memory
num_states = 2 ** num_frames
num_actions = num_frames

# Initialize Q-table
Q_table = np.zeros((num_states, num_actions))

# Learning parameters
alpha = 0.1
gamma = 0.9
epsilon = 0.8
epsilon_min = 0.1
epsilon_decay = 0.995

def state_to_index(frames):
    frames = [0 if frame is None else 1 for frame in frames]
    return int(''.join(str(f) for f in frames), 2)

def index_to_state(index):
    return [int(x) for x in bin(index)[2:].zfill(num_frames)]

rewards_per_step = []

def q_learning(pages):
    frames = [None] * num_frames
    page_faults = 0
    global epsilon

    for page in pages:
        current_state = state_to_index(frames)

        if random.uniform(0, 1) < epsilon:
            action = random.choice(range(num_actions))
        else:
            action = np.argmax(Q_table[current_state])

        if page not in frames:
            page_faults += 1
            if len(frames) >= num_frames:
                frames[action] = page
            else:
                frames[action] = page
            reward = -1
        else:
            reward = 1

        next_state = state_to_index(frames)
        best_next_action = np.argmax(Q_table[next_state])
        Q_table[current_state, action] = Q_table[current_state, action] + alpha * (reward + gamma * Q_table[next_state, best_next_action] - Q_table[current_state, action])

        rewards_per_step.append(reward)

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    return page_faults

# Simulations for traditional algorithms
def simulate_fifo(pages, num_frames):
    frames = []
    page_faults = 0
    for page in pages:
        if page not in frames:
            page_faults += 1
            if len(frames) >= num_frames:
                frames.pop(0)
            frames.append(page)
    return page_faults

def simulate_lru(pages, num_frames):
    frames = []
    page_faults = 0
    for page in pages:
        if page not in frames:
            page_faults += 1
            if len(frames) >= num_frames:
                frames.pop(frames.index(min(frames, key=lambda x: pages.index(x))))
            frames.append(page)
        else:
            frames.remove(page)
            frames.append(page)
    return page_faults

# Run simulations
q_learning_page_faults = q_learning(pages)
fifo_page_faults = simulate_fifo(pages, num_frames)
lru_page_faults = simulate_lru(pages, num_frames)

# Print Results
print("\nPage Faults Comparison:")
print(f"FIFO: {fifo_page_faults}")
print(f"LRU: {lru_page_faults}")
print(f"ML Model (Q-learning): {q_learning_page_faults}")

# === VISUALIZATIONS ===

# 1. Bar plot of page faults
plt.figure(figsize=(10, 6))
plt.bar(['FIFO', 'LRU', 'ML Model (Q-learning)'], 
        [fifo_page_faults, lru_page_faults, q_learning_page_faults], 
        color=['lightcoral', 'lightseagreen', 'lightskyblue'])
plt.xlabel('Page Replacement Algorithm')
plt.ylabel('Number of Page Faults')
plt.title('Page Fault Comparison')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 2. Line plot of cumulative rewards over time
cumulative_rewards = np.cumsum(rewards_per_step)

plt.figure(figsize=(10, 6))
plt.plot(cumulative_rewards, color='purple')
plt.xlabel('Steps')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Rewards During Q-learning Training')
plt.grid(True)
plt.show()

# 3. Exploration rate (epsilon) decay over time
epsilons = [0.8 * (epsilon_decay ** i) for i in range(len(pages))]
epsilons = [max(e, epsilon_min) for e in epsilons]

plt.figure(figsize=(10, 6))
plt.plot(epsilons, color='orange')
plt.xlabel('Steps')
plt.ylabel('Epsilon (Exploration Rate)')
plt.title('Exploration Rate Decay')
plt.grid(True)
plt.show()

# 4. Heatmap of the final Q-Table (just for fun)
# 4. Heatmap of the final Q-Table using Matplotlib
plt.figure(figsize=(8, 6))
plt.imshow(Q_table, cmap="coolwarm", interpolation='nearest', aspect='auto')
plt.colorbar(label='Q-Value')
plt.xticks(ticks=range(num_actions), labels=[f'Action {i}' for i in range(num_actions)])
plt.yticks(ticks=range(num_states), labels=[f'State {i}' for i in range(num_states)])
plt.title('Final Q-Table (State-Action Values)')
plt.xlabel('Actions (Frame Index to Evict)')
plt.ylabel('States (Memory Frame Status)')
plt.grid(False)
plt.show()
