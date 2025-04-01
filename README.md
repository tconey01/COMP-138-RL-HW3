# Dyna-Q Planning Experiment

This project investigates the impact of planning depth and experience replay strategies on reinforcement learning efficiency using the Dyna-Q algorithm.

## Project Structure

- `visual.py`: Creates visualizations of the GridWorld environment and optimal policy.
- `dynaq_exp.py`: Evaluates different replay biases (b=0, 1, 3) with fixed planning steps (n=5).
- `dynaq_exp2.py`: Analyzes the effect of planning depth (n=0, 5, 10, 20) on learning performance.
- `/images`: Directory containing generated visualizations and experiment results.

## Requirements

- Python 3.6+
- NumPy
- Matplotlib

## Running the Experiments

```bash
# Install dependencies
pip install matplotlib numpy

# Run replay strategy comparison
python dynaq_exp.py

# Run planning depth analysis
python dynaq_exp2.py
```

## Outputs

The experiments generate several visualizations:

- `gridworld_layout.png` & `gridworld_with_policy.png`: Environment visualizations
- `plot_learning_curve_vs_planning_steps.png`: Learning curves for different planning depths
- `plot_time_to_thresholds_by_planning_steps.png`: Updates required to reach performance benchmarks
- `plot_speedup_vs_planning_steps.png`: Efficiency gains compared to standard Q-learning

All visualizations use consistent formatting with colorblind-friendly elements and are saved as high-resolution PNG files (300 DPI).

## Reproducibility

All experiments use a fixed random seed (42) and average results across multiple trials to ensure consistent, reliable findings.