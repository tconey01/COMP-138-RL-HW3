import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import random
import time

# Set global seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# GridWorld Environment Definition
class GridWorld:
    def __init__(self):
        self.rows = 6
        self.cols = 9
        self.grid = [[" " for _ in range(self.cols)] for _ in range(self.rows)]

        # Wall positions
        wall_positions = [(1, 3), (2, 3), (3, 3), (5, 5)]
        for r, c in wall_positions:
            self.grid[r][c] = "X"

        self.start_pos = (3, 0)
        self.goal_pos = (0, 8)
        self.agent_pos = self.start_pos

    def step(self, action):
        row, col = self.agent_pos

        if action == "up":
            next_pos = (row - 1, col)
        elif action == "down":
            next_pos = (row + 1, col)
        elif action == "left":
            next_pos = (row, col - 1)
        elif action == "right":
            next_pos = (row, col + 1)
        else:
            return -1, False  # Invalid action

        # Stay in place if hitting wall or border
        if not (0 <= next_pos[0] < self.rows and 0 <= next_pos[1] < self.cols):
            next_pos = self.agent_pos
        if self.grid[next_pos[0]][next_pos[1]] == "X":
            next_pos = self.agent_pos

        self.agent_pos = next_pos

        # Reward only upon reaching the goal
        if self.agent_pos == self.goal_pos:
            return 1, True

        return 0, False

    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos

    def get_state(self):
        return self.agent_pos

# Q-Learning and Dyna-Q components
actions = ["up", "down", "left", "right"]

def q_learning_update(Q, state, action, reward, next_state, alpha=0.1, gamma=0.95):
    max_q_next = max(Q[next_state].values())
    Q[state][action] += alpha * (reward + gamma * max_q_next - Q[state][action])

def choose_action(Q, state, epsilon=0.1):
    if random.random() < epsilon:
        return random.choice(actions)
    return max(Q[state], key=Q[state].get)

def smooth(data, window=10):
    return [np.mean(data[max(0, i - window + 1):i + 1]) for i in range(len(data))]

# Run one trial of Dyna-Q with n planning steps
def run_dyna_q(label, planning_steps=0, max_updates=100000, seed=None):
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random(RANDOM_SEED)

    env = GridWorld()
    Q = defaultdict(lambda: {a: 0.0 for a in actions})
    Model = dict()

    log_interval = 1000
    next_log = log_interval
    total_updates = 0
    q_values, update_times = [], []

    while total_updates < max_updates:
        state = env.reset()
        done = False

        while not done and total_updates < max_updates:
            action = choose_action(Q, state, epsilon=0.1)
            reward, done = env.step(action)
            next_state = env.get_state()

            q_learning_update(Q, state, action, reward, next_state)
            total_updates += 1
            Model[(state, action)] = (next_state, reward)

            for _ in range(planning_steps):
                s_a = rng.choice(list(Model.keys()))
                s, a = s_a
                s_next, r = Model[s_a]
                q_learning_update(Q, s, a, r, s_next)
                total_updates += 1

            state = next_state

            if total_updates >= next_log:
                start_state = env.start_pos
                greedy_value = max(Q[start_state].values())
                q_values.append(greedy_value)
                update_times.append(total_updates)
                next_log += log_interval

    return q_values, update_times

# Run multiple trials and collect statistics
def run_trials(label, planning_steps, max_updates, n_trials=5):
    all_vals, all_times = [], []
    for t in range(n_trials):
        vals, times = run_dyna_q(f"{label} Trial {t+1}", planning_steps, max_updates, seed=RANDOM_SEED + t * 100)
        all_vals.append(vals)
        all_times.append(times)

    min_len = min(len(vals) for vals in all_vals)
    aligned_vals = [v[:min_len] for v in all_vals]
    aligned_times = all_times[0][:min_len]

    mean_vals = np.mean(aligned_vals, axis=0)
    std_vals = np.std(aligned_vals, axis=0)
    return mean_vals, std_vals, aligned_times

# Run and plot experiments
if __name__ == "__main__":
    configs = [0, 5, 10, 20]
    results = {}
    max_updates = 100000

    print("Starting planning depth experiments...")
    for n in configs:
        print(f"Running for planning steps: {n}")
        mean, std, times = run_trials(f"n={n}", n, max_updates)
        results[n] = (mean, std, times)

    # Plot learning curves with confidence intervals - SINGLE PLOT
    plt.figure(figsize=(10, 7), dpi=100)
    plt.style.use('seaborn-v0_8-whitegrid')
    ax = plt.gca()
    ax.set_facecolor('#f8f9fa')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    styles = ['-', '--', ':', '-.']

    for idx, n in enumerate(configs):
        mean, std, times = results[n]
        label = f"n={n}"
        smoothed = smooth(mean)
        plt.plot(times, smoothed, linestyle=styles[idx], color=colors[idx], label=label, linewidth=2.5)
        plt.fill_between(times, smoothed - std, smoothed + std, color=colors[idx], alpha=0.2)

    plt.xlabel("Computation Time (Updates)", fontsize=14, labelpad=16, color='#343a40')
    plt.ylabel("Value of Start State (Greedy Policy)", fontsize=14, color='#343a40')
    plt.title("Learning Efficiency by Planning Step Count (Dyna-Q)", 
            fontsize=16, fontweight='bold', pad=20, color='#343a40')

    plt.legend(title="Planning Steps", loc='lower right', fontsize=12, 
            frameon=True, fancybox=True, shadow=True)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(colors='#343a40')
    plt.tight_layout()
    plt.savefig("secondary_experiment_learning_curves.png", dpi=300, bbox_inches='tight')
    plt.show()