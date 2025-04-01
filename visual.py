import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def create_gridworld_visualizations():
    """
    Creates two visualizations of the GridWorld environment:
    1. Basic layout showing start, goal, and walls
    2. Same layout with optimal policy directions
    """
    rows, cols = 6, 9
    start_pos = (3, 0)
    goal_pos = (0, 8)
    wall_pos = [(1, 3), (2, 3), (3, 3), (5, 5)]

    # --- Visualization 1: Basic Grid Layout ---
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    for i in range(rows + 1):
        ax1.axhline(i, color='black', linewidth=1)
    for j in range(cols + 1):
        ax1.axvline(j, color='black', linewidth=1)

    for wall in wall_pos:
        ax1.add_patch(plt.Rectangle((wall[1], wall[0]), 1, 1, facecolor='gray'))

    ax1.add_patch(plt.Rectangle((start_pos[1], start_pos[0]), 1, 1, facecolor='lightgreen'))
    ax1.text(start_pos[1] + 0.5, start_pos[0] + 0.5, 'S', ha='center', va='center', fontsize=20, fontweight='bold')

    ax1.add_patch(plt.Rectangle((goal_pos[1], goal_pos[0]), 1, 1, facecolor='gold'))
    ax1.text(goal_pos[1] + 0.5, goal_pos[0] + 0.5, 'G', ha='center', va='center', fontsize=20, fontweight='bold')

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in wall_pos and (r, c) != start_pos and (r, c) != goal_pos:
                ax1.text(c + 0.5, r + 0.5, f'({r},{c})', ha='center', va='center', fontsize=10, color='gray')

    ax1.invert_yaxis()
    ax1.set_aspect('equal')
    ax1.set_xlim(0, cols)
    ax1.set_ylim(rows, 0)
    ax1.set_xticks(np.arange(0.5, cols, 1))
    ax1.set_xticklabels(range(cols))
    ax1.set_yticks(np.arange(0.5, rows, 1))
    ax1.set_yticklabels(range(rows))
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    ax1.set_title('GridWorld Environment (6×9)\nWalls at (1,3), (2,3), (3,3), and (5,5)', fontsize=14)

    ax1.legend(
        handles=[
            Patch(facecolor='lightgreen', edgecolor='black', label='Start (3,0)'),
            Patch(facecolor='gold', edgecolor='black', label='Goal (0,8)'),
            Patch(facecolor='gray', edgecolor='black', label='Wall')
        ],
        loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=True
    )

    # --- Visualization 2: With Optimal Policy ---
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    for i in range(rows + 1):
        ax2.axhline(i, color='black', linewidth=1)
    for j in range(cols + 1):
        ax2.axvline(j, color='black', linewidth=1)

    for wall in wall_pos:
        ax2.add_patch(plt.Rectangle((wall[1], wall[0]), 1, 1, facecolor='gray'))

    ax2.add_patch(plt.Rectangle((start_pos[1], start_pos[0]), 1, 1, facecolor='lightgreen'))
    ax2.text(start_pos[1] + 0.5, start_pos[0] + 0.5, 'S', ha='center', va='center', fontsize=20, fontweight='bold')

    ax2.add_patch(plt.Rectangle((goal_pos[1], goal_pos[0]), 1, 1, facecolor='gold'))
    ax2.text(goal_pos[1] + 0.5, goal_pos[0] + 0.5, 'G', ha='center', va='center', fontsize=20, fontweight='bold')

    policy = {}
    for r in range(rows):
        for c in range(4):
            if r <= 3:
                policy[(r, c)] = 1 if c < 2 else 0
            else:
                policy[(r, c)] = 1
    for r in range(1, rows):
        for c in range(4, cols):
            policy[(r, c)] = 0
    for c in range(cols - 1):
        policy[(0, c)] = 1

    arrow_dirs = [(0, -0.4), (0.4, 0), (0, 0.4), (-0.4, 0)]

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in wall_pos and (r, c) != goal_pos and (r, c) in policy:
                dx, dy = arrow_dirs[policy[(r, c)]]
                ax2.arrow(c + 0.5, r + 0.5, dx, dy,
                          head_width=0.15, head_length=0.15, fc='blue', ec='blue', width=0.05,
                          length_includes_head=True)

    ax2.invert_yaxis()
    ax2.set_aspect('equal')
    ax2.set_xlim(0, cols)
    ax2.set_ylim(rows, 0)
    ax2.set_xticks(np.arange(0.5, cols, 1))
    ax2.set_xticklabels(range(cols))
    ax2.set_yticks(np.arange(0.5, rows, 1))
    ax2.set_yticklabels(range(rows))
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    ax2.set_title('GridWorld with Optimal Policy\nBlue arrows indicate the optimal action in each state', fontsize=14)

    ax2.legend(
        handles=[
            Patch(facecolor='lightgreen', edgecolor='black', label='Start (3,0)'),
            Patch(facecolor='gold', edgecolor='black', label='Goal (0,8)'),
            Patch(facecolor='gray', edgecolor='black', label='Wall'),
            Line2D([0], [0], marker=r'$\uparrow$', color='w', markerfacecolor='blue',
                   markersize=15, label='Policy Direction')
        ],
        loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=True
    )

    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig('gridworld_layout.png', dpi=300, bbox_inches='tight')
    fig2.savefig('gridworld_with_policy.png', dpi=300, bbox_inches='tight')

    print("Both GridWorld visualizations have been created and saved.")
    return fig1, fig2

# Run visualization
if __name__ == "__main__":
    create_gridworld_visualizations()
