import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import random

# Environment: GridWorld with goal, walls, and deterministic movement
class GridWorld:
    def __init__(self):
        self.rows = 6
        self.cols = 9
        self.grid = [[" " for _ in range(self.cols)] for _ in range(self.rows)]
        for r, c in [(1, 3), (2, 3), (3, 3), (5, 5)]:
            self.grid[r][c] = "X"
        self.start_pos = (3, 0)
        self.goal_pos = (0, 8)
        self.agent_pos = self.start_pos

    def step(self, action):
        row, col = self.agent_pos
        moves = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
        dr, dc = moves.get(action, (0, 0))
        next_pos = (row + dr, col + dc)
        if not (0 <= next_pos[0] < self.rows and 0 <= next_pos[1] < self.cols) or self.grid[next_pos[0]][next_pos[1]] == "X":
            next_pos = self.agent_pos
        self.agent_pos = next_pos
        return (1, True) if self.agent_pos == self.goal_pos else (0, False)

    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos

    def get_state(self):
        return self.agent_pos

# Q-learning components
actions = ["up", "down", "left", "right"]

def q_learning_update(Q, state, action, reward, next_state, alpha=0.1, gamma=0.95):
    Q[state][action] += alpha * (reward + gamma * max(Q[next_state].values()) - Q[state][action])

def choose_action(Q, state, epsilon=0.1):
    return random.choice(actions) if random.random() < epsilon else max(Q[state], key=Q[state].get)

def smooth(data, window=10):
    return [np.mean(data[max(0, i-window+1):i+1]) for i in range(len(data))]

# Dyna-Q experiment with replay bias (b)
def run_dyna_q_variant(label, planning_steps=5, b=0, max_updates=200000, return_q=False):
    env = GridWorld()
    Q = defaultdict(lambda: {a: 0.0 for a in actions})
    Model = {}
    history, q_values, update_times = [], [], []
    total_updates, next_log = 0, 1000

    while total_updates < max_updates:
        state, done = env.reset(), False
        while not done and total_updates < max_updates:
            action = choose_action(Q, state)
            reward, done = env.step(action)
            next_state = env.get_state()

            q_learning_update(Q, state, action, reward, next_state)
            total_updates += 1
            Model[(state, action)] = (next_state, reward)
            if b > 0:
                history.append((state, action))
                if len(history) > 1000:
                    history.pop(0)

            if b == 0:
                for _ in range(planning_steps):
                    s_a = random.choice(list(Model)) if Model else None
                    if s_a:
                        s, a = s_a
                        s_next, r = Model[s_a]
                        q_learning_update(Q, s, a, r, s_next)
                        total_updates += 1
            else:
                for s_a in history[-b:]:
                    if s_a in Model:
                        s_next, r = Model[s_a]
                        q_learning_update(Q, s_a[0], s_a[1], r, s_next)
                        total_updates += 1
                for _ in range(max(0, planning_steps - b)):
                    s_a = random.choice(list(Model)) if Model else None
                    if s_a:
                        s, a = s_a
                        s_next, r = Model[s_a]
                        q_learning_update(Q, s, a, r, s_next)
                        total_updates += 1

            state = next_state
            if total_updates >= next_log:
                q_values.append(max(Q[env.start_pos].values()))
                update_times.append(total_updates)
                next_log += 1000

    return (q_values, update_times, Q) if return_q else (q_values, update_times)

# Run and plot experiments
print("Starting experiments...")
vals_b0, times_b0 = run_dyna_q_variant("Uniform (b=0)", planning_steps=5, b=0)
vals_b1, times_b1 = run_dyna_q_variant("Biased (b=1)", planning_steps=5, b=1)
vals_b3, times_b3 = run_dyna_q_variant("Biased (b=3)", planning_steps=5, b=3)
print("Experiments complete.")

# Add initial zero point for visualization
for vals, times in [(vals_b0, times_b0), (vals_b1, times_b1), (vals_b3, times_b3)]:
    vals.insert(0, 0.0)
    times.insert(0, 0)

# Plot learning curves
plt.figure(figsize=(12, 8))
plt.style.use('seaborn-v0_8-darkgrid')

plt.plot(times_b0, smooth(vals_b0), label="Uniform (b=0)", linewidth=2.5, linestyle='-')
plt.plot(times_b1, smooth(vals_b1), label="Biased (b=1)", linewidth=2.5, linestyle='--')
plt.plot(times_b3, smooth(vals_b3), label="Biased (b=3)", linewidth=2.5, linestyle=':')

plt.xlabel("Computation Time (Updates)", fontsize=14)
plt.ylabel("Value of Start State", fontsize=14)
plt.title("Effect of Replay Bias in Dyna-Q", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("dyna_q_replay_bias_comparison.png", dpi=300)
plt.show()
