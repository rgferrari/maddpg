#!/usr/bin/env python
"""
Visualize training curves from MADDPG experiments.
Usage:
    python plot_results.py --exp-name adversary
"""
import pickle
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def moving_average(data, window_size=100):
    """Compute moving average of data."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_results(exp_name, plots_dir="./learning_curves/"):
    """Plot training curves for an experiment."""
    
    rewards_file = os.path.join(plots_dir, "{}_rewards.pkl".format(exp_name))
    agrewards_file = os.path.join(plots_dir, "{}_agrewards.pkl".format(exp_name))
    
    if not os.path.exists(rewards_file):
        print("ERROR: Rewards file not found: {}".format(rewards_file))
        return
    
    # Load training data
    with open(rewards_file, 'rb') as fp:
        final_ep_rewards = pickle.load(fp)
    
    agent_rewards = None
    if os.path.exists(agrewards_file):
        with open(agrewards_file, 'rb') as fp:
            agent_rewards = pickle.load(fp)
    
    print("Loaded {} episodes of reward data".format(len(final_ep_rewards)))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("MADDPG Training Results: {}".format(exp_name), fontsize=16, fontweight='bold')
    
    # Plot 1: Mean episode rewards
    ax = axes[0]
    episodes = np.arange(len(final_ep_rewards)) * 1000  # save_rate defaults to 1000
    ax.plot(episodes, final_ep_rewards, 'o-', alpha=0.6, label='Raw rewards', markersize=4)
    
    # Add moving average
    if len(final_ep_rewards) > 100:
        ma_100 = moving_average(final_ep_rewards, window_size=10)
        ma_episodes = episodes[:len(ma_100)]
        ax.plot(ma_episodes, ma_100, 'r-', linewidth=2, label='Moving average (window=10)')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Mean Episode Reward', fontsize=12)
    ax.set_title('Total Reward per Episode', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Plot 2: Individual agent rewards (if available)
    if agent_rewards is not None and len(agent_rewards) > 0:
        ax = axes[1]
        first_item = agent_rewards[0]

        # Some runs save per-agent rewards as a flat numeric list instead of list-of-lists.
        if isinstance(first_item, (list, tuple, np.ndarray)):
            num_agents = len(agent_rewards)
            colors = plt.cm.tab10(np.linspace(0, 1, num_agents))

            for i, rewards in enumerate(agent_rewards):
                rewards_array = np.asarray(rewards)
                if rewards_array.size > 0:
                    agent_episodes = np.arange(rewards_array.size) * 1000
                    ax.plot(agent_episodes, rewards_array, 'o-', alpha=0.6,
                            label='Agent {}'.format(i), color=colors[i], markersize=4)

            ax.set_title('Individual Agent Rewards', fontsize=13)
        else:
            flat_rewards = np.asarray(agent_rewards, dtype=np.float64)
            flat_episodes = np.arange(flat_rewards.size) * 1000
            ax.plot(flat_episodes, flat_rewards, 'o-', alpha=0.7,
                    label='Agent reward samples (flattened)', color='tab:green', markersize=4)
            ax.set_title('Agent Rewards (Flattened Log Format)', fontsize=13)

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Agent Reward', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    else:
        axes[1].text(0.5, 0.5, 'No individual agent reward data available',
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(plots_dir, "{}_training_curves.png".format(exp_name))
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print("\n✓ Saved plot: {}".format(output_file))
    
    # Display statistics
    print("\n" + "="*60)
    print("TRAINING STATISTICS")
    print("="*60)
    print("Total episodes recorded:     {}".format(len(final_ep_rewards)))
    print("Mean episode reward (full):  {:.4f}".format(np.mean(final_ep_rewards)))
    print("Std dev:                     {:.4f}".format(np.std(final_ep_rewards)))
    print("Max episode reward:          {:.4f}".format(np.max(final_ep_rewards)))
    print("Min episode reward:          {:.4f}".format(np.min(final_ep_rewards)))
    
    if len(final_ep_rewards) > 100:
        recent = final_ep_rewards[-10:]
        print("\nLast 10 episodes:")
        print("  Mean:  {:.4f}".format(np.mean(recent)))
        print("  Std:   {:.4f}".format(np.std(recent)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Plot MADDPG training results")
    parser.add_argument("--exp-name", type=str, required=True, help="Experiment name")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="Directory with pickle files")
    args = parser.parse_args()
    
    plot_results(args.exp_name, args.plots_dir)
