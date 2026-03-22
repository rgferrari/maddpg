#!/usr/bin/env python
"""
Run trained MADDPG agents with visual display.
Usage:
    python play_trained_agents.py --exp-name adversary --scenario simple_adversary --num-episodes 10
"""
import argparse
import os
import sys
import numpy as np
import tensorflow as tf

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
from experiments.train import make_env, get_trainers, mlp_model

def play_trained_agents(arglist):
    """Load trained model and run episodes with display."""
    use_display = bool(arglist.display)
    if use_display and not os.environ.get("DISPLAY"):
        print("WARNING: DISPLAY is not set. Running headless (no render window).")
        use_display = False
    
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, benchmark=False)
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))
        
        # Initialize
        U.initialize()
        
        # Load trained model
        print("\nLoading trained model from: {}".format(arglist.load_dir))
        saver = tf.train.Saver()
        try:
            U.load_state(arglist.load_dir)
            print("✓ Model loaded successfully!")
        except Exception as e:
            print("ERROR: Failed to load model: {}".format(e))
            return
        
        mode_text = "display" if use_display else "headless"
        print("\nRunning {} episodes in {} mode...\n".format(arglist.num_episodes, mode_text))
        
        episode_rewards = []
        
        for episode in range(arglist.num_episodes):
            obs_n = env.reset()
            episode_reward = 0
            episode_agent_rewards = [0.0] * env.n
            
            for step in range(arglist.max_episode_len):
                # Get actions from agents
                action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
                
                # Take step in environment
                new_obs_n, reward_n, done_n, info_n = env.step(action_n)
                
                # Accumulate rewards
                episode_reward += sum(reward_n)
                for i, reward in enumerate(reward_n):
                    episode_agent_rewards[i] += reward
                
                # Render
                if use_display:
                    env.render()
                
                obs_n = new_obs_n
                
                # Check if episode is done
                if all(done_n):
                    break
            
            episode_rewards.append(episode_reward)
            
            # Print progress
            if (episode + 1) % max(1, arglist.num_episodes // 5) == 0 or episode == 0:
                print("Episode {}/{}".format(episode + 1, arglist.num_episodes))
                print("  Total reward:     {:.2f}".format(episode_reward))
                agent_rewards_str = "[" + ", ".join("{:.2f}".format(r) for r in episode_agent_rewards) + "]"
                print("  Agent rewards:    {}".format(agent_rewards_str))
                print()
        
        # Final statistics
        print("="*60)
        print("EVALUATION STATISTICS")
        print("="*60)
        print("Episodes run:        {}".format(arglist.num_episodes))
        print("Mean episode reward: {:.4f}".format(np.mean(episode_rewards)))
        print("Std dev:             {:.4f}".format(np.std(episode_rewards)))
        print("Max episode reward:  {:.4f}".format(np.max(episode_rewards)))
        print("Min episode reward:  {:.4f}".format(np.min(episode_rewards)))
        print("="*60)

def parse_args():
    parser = argparse.ArgumentParser("Play trained MADDPG agents")
    
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=10, help="number of episodes to play")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    
    # Model parameters (must match training)
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="batch size")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in mlp")
    
    # Checkpointing
    parser.add_argument("--exp-name", type=str, required=True, help="experiment name")
    parser.add_argument("--load-dir", type=str, default="/tmp/policy/", help="directory to load trained model from")
    
    # Display
    parser.add_argument("--display", action="store_true", default=False, help="enable visual display")
    
    return parser.parse_args()

if __name__ == "__main__":
    arglist = parse_args()
    play_trained_agents(arglist)
