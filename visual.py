import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
import matplotlib as mpl

def create_professional_gridworld_visualizations():
    """
    Creates two professional visualizations of the GridWorld environment:
    1. Basic layout showing start, goal, and walls
    2. Same layout with optimal policy directions
    
    Uses a consistent, professional color scheme matching the main codebase.
    """
    # Set style for a more professional look
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Professional color scheme matching main code
    # Based on matplotlib default colors but with more professional tones
    colors = {
        'background': '#f8f9fa',
        'grid': '#e9ecef',
        'wall': '#adb5bd',
        'start': '#52b788',  # Professional green
        'goal': '#ffb703',   # Professional gold
        'arrow': '#1f77b4',  # Standard blue from the main plots
        'text': '#343a40'
    }
    
    # Set font properties for a consistent, professional look
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    mpl.rcParams['font.size'] = 12
    
    # GridWorld parameters
    rows, cols = 6, 9
    start_pos = (3, 0)
    goal_pos = (0, 8)
    wall_pos = [(1, 3), (2, 3), (3, 3), (5, 5)]
    
    # --- VISUALIZATION 1: BASIC LAYOUT ---
    fig1, ax1 = plt.subplots(figsize=(10, 7), dpi=100)
    ax1.set_facecolor(colors['background'])
    fig1.patch.set_facecolor(colors['background'])
    
    # Create the grid
    for i in range(rows + 1):
        ax1.axhline(i, color=colors['grid'], linewidth=1)
    for j in range(cols + 1):
        ax1.axvline(j, color=colors['grid'], linewidth=1)
    
    # Draw walls
    for wall in wall_pos:
        ax1.add_patch(Rectangle((wall[1], wall[0]), 1, 1, 
                               facecolor=colors['wall'], 
                               edgecolor='white', 
                               linewidth=1.5))
    
    # Draw start position
    ax1.add_patch(Rectangle((start_pos[1], start_pos[0]), 1, 1, 
                           facecolor=colors['start'], 
                           edgecolor='white', 
                           linewidth=1.5))
    ax1.text(start_pos[1] + 0.5, start_pos[0] + 0.5, 'S', 
            ha='center', va='center', 
            fontsize=18, fontweight='bold', color='white')
    
    # Draw goal position
    ax1.add_patch(Rectangle((goal_pos[1], goal_pos[0]), 1, 1, 
                           facecolor=colors['goal'], 
                           edgecolor='white', 
                           linewidth=1.5))
    ax1.text(goal_pos[1] + 0.5, goal_pos[0] + 0.5, 'G', 
            ha='center', va='center', 
            fontsize=18, fontweight='bold', color='white')
    
    # Add coordinates (subtler)
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in wall_pos and (r, c) != start_pos and (r, c) != goal_pos:
                ax1.text(c + 0.5, r + 0.5, f'({r},{c})', 
                        ha='center', va='center', 
                        fontsize=8, color='#6c757d', alpha=0.7)
    
    # Configure axes
    ax1.invert_yaxis()
    ax1.set_aspect('equal')
    ax1.set_xlim(0, cols)
    ax1.set_ylim(rows, 0)
    
    # Customize ticks for a cleaner look
    ax1.set_xticks(np.arange(0.5, cols, 1))
    ax1.set_xticklabels(range(cols))
    ax1.set_yticks(np.arange(0.5, rows, 1))
    ax1.set_yticklabels(range(rows))
    ax1.tick_params(axis='both', which='both', length=0)  # Hide tick marks
    
    # Add labels and title with improved spacing
    ax1.set_xlabel('Column', fontsize=14, labelpad=16, color=colors['text'])
    ax1.set_ylabel('Row', fontsize=14, color=colors['text'])
    ax1.set_title('GridWorld Environment (6×9)', 
                 fontsize=16, fontweight='bold', pad=20, color=colors['text'])
    
    # Create legend with a more professional look
    legend_handles = [
        Patch(facecolor=colors['start'], edgecolor='white', label='Start (3,0)'),
        Patch(facecolor=colors['goal'], edgecolor='white', label='Goal (0,8)'),
        Patch(facecolor=colors['wall'], edgecolor='white', label='Wall')
    ]
    ax1.legend(handles=legend_handles, 
              loc='upper center', 
              bbox_to_anchor=(0.5, -0.12), 
              ncol=3, 
              frameon=True, 
              fancybox=True, 
              shadow=True,
              fontsize=11)
    
    # --- VISUALIZATION 2: WITH OPTIMAL POLICY ---
    fig2, ax2 = plt.subplots(figsize=(10, 7), dpi=100)
    ax2.set_facecolor(colors['background'])
    fig2.patch.set_facecolor(colors['background'])
    
    # Create the grid
    for i in range(rows + 1):
        ax2.axhline(i, color=colors['grid'], linewidth=1)
    for j in range(cols + 1):
        ax2.axvline(j, color=colors['grid'], linewidth=1)
    
    # Draw walls
    for wall in wall_pos:
        ax2.add_patch(Rectangle((wall[1], wall[0]), 1, 1, 
                               facecolor=colors['wall'], 
                               edgecolor='white', 
                               linewidth=1.5))
    
    # Draw start position
    ax2.add_patch(Rectangle((start_pos[1], start_pos[0]), 1, 1, 
                           facecolor=colors['start'], 
                           edgecolor='white', 
                           linewidth=1.5))
    ax2.text(start_pos[1] + 0.5, start_pos[0] + 0.5, 'S', 
            ha='center', va='center', 
            fontsize=18, fontweight='bold', color='white')
    
    # Draw goal position
    ax2.add_patch(Rectangle((goal_pos[1], goal_pos[0]), 1, 1, 
                           facecolor=colors['goal'], 
                           edgecolor='white', 
                           linewidth=1.5))
    ax2.text(goal_pos[1] + 0.5, goal_pos[0] + 0.5, 'G', 
            ha='center', va='center', 
            fontsize=18, fontweight='bold', color='white')
    
    # Define policy (mapping of (r,c) to action directions)
    policy = {}
    
    # Left section (need to go right and then up)
    for r in range(rows):
        for c in range(4):
            if r <= 3:  # Upper part
                if c < 2:  # First two columns
                    policy[(r, c)] = 1  # go right
                else:  # Next two columns before the wall
                    policy[(r, c)] = 0  # go up
            else:  # Lower part (rows 4-5)
                policy[(r, c)] = 1  # go right to go around the wall
    
    # Right section (mostly go up)
    for r in range(1, rows):
        for c in range(4, cols):
            policy[(r, c)] = 0  # go up
    
    # Top row (go right to reach the goal)
    for c in range(cols-1):
        policy[(0, c)] = 1  # go right
    
    # Define arrow directions and properties
    arrow_dirs = [
        (0, -0.4),  # up: dx=0, dy=-0.4
        (0.4, 0),   # right: dx=0.4, dy=0
        (0, 0.4),   # down: dx=0, dy=0.4
        (-0.4, 0)   # left: dx=-0.4, dy=0
    ]
    
    arrow_props = {
        'head_width': 0.18,
        'head_length': 0.18,
        'fc': colors['arrow'],
        'ec': colors['arrow'],
        'width': 0.06,
        'length_includes_head': True,
        'zorder': 10  # Ensure arrows appear above the grid
    }
    
    # Draw policy arrows (except at start position)
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in wall_pos and (r, c) != goal_pos and (r, c) in policy:
                # Skip drawing arrow at the start position
                if (r, c) != start_pos:
                    dx, dy = arrow_dirs[policy[(r, c)]]
                    ax2.arrow(c + 0.5, r + 0.5, dx, dy, **arrow_props)
    
    # Configure axes
    ax2.invert_yaxis()
    ax2.set_aspect('equal')
    ax2.set_xlim(0, cols)
    ax2.set_ylim(rows, 0)
    
    # Customize ticks
    ax2.set_xticks(np.arange(0.5, cols, 1))
    ax2.set_xticklabels(range(cols))
    ax2.set_yticks(np.arange(0.5, rows, 1))
    ax2.set_yticklabels(range(rows))
    ax2.tick_params(axis='both', which='both', length=0)
    
    # Add labels and title
    ax2.set_xlabel('Column', fontsize=14, labelpad=16, color=colors['text'])
    ax2.set_ylabel('Row', fontsize=14, color=colors['text'])
    ax2.set_title('GridWorld with Optimal Policy', 
                 fontsize=16, fontweight='bold', pad=20, color=colors['text'])
    
    # Create legend with a more professional look
    legend_handles = [
        Patch(facecolor=colors['start'], edgecolor='white', label='Start (3,0)'),
        Patch(facecolor=colors['goal'], edgecolor='white', label='Goal (0,8)'),
        Patch(facecolor=colors['wall'], edgecolor='white', label='Wall'),
        Line2D([0], [0], marker=r'$\uparrow$', color='w', markerfacecolor=colors['arrow'],
               markersize=15, label='Policy Direction')
    ]
    ax2.legend(handles=legend_handles, 
              loc='upper center', 
              bbox_to_anchor=(0.5, -0.12), 
              ncol=4, 
              frameon=True, 
              fancybox=True, 
              shadow=True,
              fontsize=11)
    
    # Adjust layout to accommodate the legend
    fig1.subplots_adjust(bottom=0.18)
    fig2.subplots_adjust(bottom=0.18)
    
    # Save high-resolution images with tight layouts
    fig1.savefig('gridworld_layout.png', dpi=300, bbox_inches='tight', facecolor=fig1.get_facecolor())
    fig2.savefig('gridworld_with_policy.png', dpi=300, bbox_inches='tight', facecolor=fig2.get_facecolor())
    
    print("Professional GridWorld visualizations have been created and saved.")
    return fig1, fig2

# Run visualization
if __name__ == "__main__":
    create_professional_gridworld_visualizations()