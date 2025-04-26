import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset (synthetic page replacement data)
df = pd.read_csv("synthetic_page_replacement_data.csv")
pages = df['Current_Page'].tolist()

# Parameters for the simulation
num_frames = 3  # Number of frames in memory
num_states = 2 ** num_frames  # Number of possible states (2^num_frames, binary states)
num_actions = num_frames  # Number of possible actions (evict any one of the pages)

# Initialize Q-table with zeros
Q_table = np.zeros((num_states, num_actions))

# Learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.8  # Initial Exploration rate
epsilon_min = 0.1  # Minimum exploration rate
epsilon_decay = 0.995  # Epsilon decay rate

# Convert the memory frames (state) into a binary representation
def state_to_index(frames):
    """Convert a set of frames into a binary state index."""
    # Handle None values by treating them as 0 (empty frame)
    frames = [0 if frame is None else 1 for frame in frames]
    return int(''.join(str(f) for f in frames), 2)

def index_to_state(index):
    """Convert a state index back into a set of frames."""
    return [int(x) for x in bin(index)[2:].zfill(num_frames)]

# Simulate Q-learning for page replacement
def q_learning(pages):
    frames = [None] * num_frames  # Initial empty frames
    page_faults = 0
    rewards = []

    global epsilon  # Use global epsilon to apply decay after each episode

    for page in pages:
        # Convert frames to a state index
        current_state = state_to_index(frames)

        # Explore or exploit (epsilon-greedy)
        if random.uniform(0, 1) < epsilon:
            action = random.choice(range(num_actions))  # Explore (choose a random action)
        else:
            action = np.argmax(Q_table[current_state])  # Exploit (choose the best action from Q-table)

        # If the page is not in memory, it's a page fault
        if page not in frames:
            page_faults += 1
            if len(frames) >= num_frames:
                frames[action] = page  # Evict a page (based on action)
            else:
                frames[action] = page  # Add new page if there is space
            reward = -1  # Page fault penalty
        else:
            reward = 1  # No page fault (page hit)

        # Update the Q-value using the Q-learning formula
        next_state = state_to_index(frames)
        best_next_action = np.argmax(Q_table[next_state])
        Q_table[current_state, action] = Q_table[current_state, action] + alpha * (reward + gamma * Q_table[next_state, best_next_action] - Q_table[current_state, action])

        rewards.append(reward)

        # Decay epsilon after each step
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    return page_faults

# Run Q-learning on the page reference string
q_learning_page_faults = q_learning(pages)

# Now, simulate FIFO and LRU for comparison

def simulate_fifo(pages, num_frames):
    frames = []
    page_faults = 0
    for page in pages:
        if page not in frames:
            page_faults += 1
            if len(frames) >= num_frames:
                frames.pop(0)  # Remove the oldest page (FIFO)
            frames.append(page)
    return page_faults

def simulate_lru(pages, num_frames):
    frames = []
    page_faults = 0
    for page in pages:
        if page not in frames:
            page_faults += 1
            if len(frames) >= num_frames:
                frames.pop(frames.index(min(frames, key=lambda x: pages.index(x))))  # Remove the least recently used page
            frames.append(page)
        else:
            frames.remove(page)  # Move to the end to mark as recently used
            frames.append(page)
    return page_faults

# Now, simulate FIFO and LRU for comparison
fifo_page_faults = simulate_fifo(pages, num_frames)
lru_page_faults = simulate_lru(pages, num_frames)

# Display the results
print("\nPage Faults Comparison:")
print(f"FIFO: {fifo_page_faults}")
print(f"LRU: {lru_page_faults}")
print(f"ML Model (Q-learning): {q_learning_page_faults}")

# Create a bar plot to show the page faults comparison
plt.figure(figsize=(10, 6))
plt.bar(['FIFO', 'LRU', 'ML Model (Q-learning)'], 
        [fifo_page_faults, lru_page_faults, q_learning_page_faults], 
        color='skyblue')
plt.xlabel('Page Replacement Algorithm')
plt.ylabel('Page Faults')
plt.title('Page Fault Comparison')
plt.show()
