"""
Utility script to plot training rewards from saved training data.

This script loads reward histories from checkpoints directories and 
generates visualization plots.
"""

import os
import json
import argparse
from typing import Optional, List
import matplotlib.pyplot as plt
import numpy as np


def load_rewards_from_metadata(checkpoint_dir: str) -> Optional[List[float]]:
    """
    Try to load rewards from training metadata.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        
    Returns:
        List of episode rewards if available, None otherwise
    """
    meta_path = os.path.join(checkpoint_dir, "training_meta.json")
    if not os.path.exists(meta_path):
        return None
    
    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        # The metadata doesn't store full reward history, just best performance
        return None
    except Exception as e:
        print(f"Error loading metadata from {checkpoint_dir}: {e}")
        return None


def load_rewards_from_csv(checkpoint_dir: str) -> Optional[List[float]]:
    """
    Load rewards from CSV file if available.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        
    Returns:
        List of episode rewards if CSV exists, None otherwise
    """
    csv_path = os.path.join(checkpoint_dir, "training_rewards.csv")
    if not os.path.exists(csv_path):
        return None
    
    try:
        rewards = []
        with open(csv_path, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    try:
                        reward = float(parts[1])
                        rewards.append(reward)
                    except ValueError:
                        continue
        return rewards if rewards else None
    except Exception as e:
        print(f"Error loading CSV from {checkpoint_dir}: {e}")
        return None


def plot_training_rewards(
    checkpoint_dir: str, 
    output_dir: Optional[str] = None,
    window_size: int = 100,
    title_suffix: str = ""
):
    """
    Plot training rewards from a checkpoint directory.
    
    Args:
        checkpoint_dir: Path to checkpoint directory containing training data
        output_dir: Directory to save plot (defaults to checkpoint_dir)
        window_size: Size of moving average window
        title_suffix: Additional text for the plot title
    """
    if output_dir is None:
        output_dir = checkpoint_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Try to load rewards
    rewards = load_rewards_from_csv(checkpoint_dir)
    
    if rewards is None:
        print(f"No training rewards found in {checkpoint_dir}")
        return False
    
    print(f"Loaded {len(rewards)} episodes from {checkpoint_dir}")
    
    # Calculate moving average
    rewards_array = np.array(rewards)
    if len(rewards_array) < window_size:
        print(f"Warning: Only {len(rewards_array)} episodes, window_size={window_size}")
        window_size = max(1, len(rewards_array) // 2)
    
    moving_avg = np.convolve(rewards_array, np.ones(window_size)/window_size, mode='valid')
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Episode rewards with moving average
    ax1.plot(rewards, alpha=0.3, label='Episode Reward')
    ax1.plot(range(window_size-1, len(rewards)), moving_avg, 
             color='orange', linewidth=2, label=f'Moving Avg (window={window_size})')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Total Episode Reward', fontsize=12)
    ax1.set_title(f'Training Rewards Over Time {title_suffix}', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Moving average only (zoomed)
    ax2.plot(range(window_size-1, len(rewards)), moving_avg, 
             color='green', linewidth=2)
    ax2.fill_between(range(window_size-1, len(rewards)), moving_avg, alpha=0.3, color='green')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Average Reward', fontsize=12)
    ax2.set_title(f'Moving Average {title_suffix}', fontsize=13)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, 'training_rewards_plot.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {plot_file}")
    
    # Print statistics
    print(f"\nReward Statistics:")
    print(f"  Min: {np.min(rewards):.2f}")
    print(f"  Max: {np.max(rewards):.2f}")
    print(f"  Mean: {np.mean(rewards):.2f}")
    print(f"  Std Dev: {np.std(rewards):.2f}")
    print(f"  Final Moving Avg: {moving_avg[-1]:.2f}")
    
    plt.close()
    return True


def compare_training_runs(
    checkpoint_dirs: List[str],
    labels: Optional[List[str]] = None,
    output_dir: str = ".",
    window_size: int = 100
):
    """
    Plot multiple training runs for comparison.
    
    Args:
        checkpoint_dirs: List of checkpoint directories to compare
        labels: Labels for each run (defaults to directory names)
        output_dir: Directory to save comparison plot
        window_size: Size of moving average window
    """
    if labels is None:
        labels = [os.path.basename(d) for d in checkpoint_dirs]
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for checkpoint_dir, label in zip(checkpoint_dirs, labels):
        rewards = load_rewards_from_csv(checkpoint_dir)
        
        if rewards is None:
            print(f"Skipping {checkpoint_dir} - no data found")
            continue
        
        # Calculate moving average
        rewards_array = np.array(rewards)
        if len(rewards_array) < window_size:
            window_size = max(1, len(rewards_array) // 2)
        
        moving_avg = np.convolve(rewards_array, np.ones(window_size)/window_size, mode='valid')
        
        ax.plot(range(window_size-1, len(rewards)), moving_avg, 
                linewidth=2, label=label, alpha=0.8)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title('Training Comparison', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, 'comparison.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot: {plot_file}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot training rewards from checkpoint directories"
    )
    parser.add_argument(
        "checkpoint_dirs",
        nargs="+",
        help="Path(s) to checkpoint directory/directories"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for plots (defaults to checkpoint dir)"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Moving average window size (default: 100)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Create comparison plot for multiple directories"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Labels for comparison plot"
    )
    
    args = parser.parse_args()
    
    if len(args.checkpoint_dirs) == 1 and not args.compare:
        # Single plot
        plot_training_rewards(
            args.checkpoint_dirs[0],
            output_dir=args.output,
            window_size=args.window
        )
    else:
        # Comparison plot
        compare_training_runs(
            args.checkpoint_dirs,
            labels=args.labels,
            output_dir=args.output or ".",
            window_size=args.window
        )
