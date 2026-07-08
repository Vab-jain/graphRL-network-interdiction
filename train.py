import numpy as np
import torch as th
import time
import argparse
import os

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from graph_nip.env import NetworkInterdictionEnv
from graph_nip.policy import MaskableGraphActorCriticPolicy
from graph_nip.utils import get_clean_kwargs, change_obs_action_space
from graph_nip.config import ExperimentConfig

def train_ppo(train_env, val_env, config: ExperimentConfig, run_id: int):
    # Setup PPO Kwargs
    ppo_kwargs = {
        "learning_rate": config.train_cfg.learning_rate,
        "gamma": config.train_cfg.gamma,
        "n_steps": config.train_cfg.n_steps,
        "batch_size": config.train_cfg.batch_size,
        "seed": config.train_cfg.seed,
    }
    ppo_kwargs = get_clean_kwargs(MaskablePPO.__init__, warn=False, kwargs=ppo_kwargs)

    policy_kwargs = {
        "pooling_type": config.policy_cfg.pooling_type,
        "embed_dim": config.policy_cfg.embed_dim,
        "network_kwargs": {"network": config.policy_cfg.network, "num_layers": config.policy_cfg.num_layers},
        "node_dim": train_env.observation_space["node_features"].shape[1]
    }

    os.makedirs(f"models/{run_id}", exist_ok=True)
    os.makedirs(f"runs/{run_id}", exist_ok=True)

    print(f"Creating PPO model with {config.policy_cfg.network}...")
    model = MaskablePPO(
        MaskableGraphActorCriticPolicy,
        train_env,
        **ppo_kwargs,
        policy_kwargs=policy_kwargs,
        tensorboard_log=f"runs/{run_id}",
        verbose=1
    )

    eval_callback = MaskableEvalCallback(
        eval_env=val_env,
        n_eval_episodes=config.train_cfg.n_eval_episodes,
        eval_freq=config.train_cfg.eval_freq,
        best_model_save_path=f"models/{run_id}",
        deterministic=True,
        render=False,
        use_masking=True,
    )

    print(f"Starting training for {config.train_cfg.timesteps} timesteps...")
    model.learn(
        total_timesteps=config.train_cfg.timesteps,
        callback=eval_callback,
        use_masking=True,
        progress_bar=True
    )
    return model

def evaluate(run_id: int, config: ExperimentConfig, eval_graph_sizes: tuple):
    print("\n--- Evaluating Models on different graph sizes ---")
    model_path = f"models/{run_id}/best_model.zip"
    if not os.path.exists(model_path):
        print("Model not found. Skipping evaluation.")
        return

    model = MaskablePPO.load(model_path)
    
    results = {}
    for size in eval_graph_sizes:
        print(f"\nEvaluating on graphs of ~{size} nodes...")
        # Derive rows/cols for the grid approximation. E.g. size=12 -> 4 cols, 3 nodes
        num_cols = max(3, size // 3)
        nodes_per_col = max(2, size // num_cols)
        
        eval_cfg = config.train_graph_cfg
        eval_cfg.num_cols = num_cols
        eval_cfg.nodes_per_col = nodes_per_col
        
        test_env = VecMonitor(DummyVecEnv([lambda: NetworkInterdictionEnv(eval_cfg, seed=100+size, is_eval=True)]))
        model.policy = change_obs_action_space(model.policy, test_env)
        
        ep_rewards, ep_lengths = evaluate_policy(
            model,
            test_env,
            n_eval_episodes=config.train_cfg.n_eval_episodes,
            deterministic=True,
            render=False,
            use_masking=True,
            return_episode_rewards=True,
        )
        
        mean_rew = np.mean(ep_rewards)
        std_rew = np.std(ep_rewards)
        print(f"Size {size} | Mean Reward: {mean_rew:.3f} +/- {std_rew:.3f}")
        results[size] = mean_rew
        
    return results


def main():
    parser = argparse.ArgumentParser(description="Train Graph RL Network Interdiction")
    parser.add_argument("--network", type=str, default="GAT", choices=["GCN", "GAT", "GraphSAGE"], help="GNN Architecture")
    parser.add_argument("--timesteps", type=int, default=50000, help="Training timesteps")
    args = parser.parse_args()

    config = ExperimentConfig()
    config.policy_cfg.network = args.network
    config.train_cfg.timesteps = args.timesteps
    
    th.manual_seed(config.train_cfg.seed)
    np.random.seed(config.train_cfg.seed)
    
    run_id = int(time.time())

    def make_env(is_eval=False, seed_offset=0):
        def _init():
            return NetworkInterdictionEnv(config.train_graph_cfg, seed=config.train_cfg.seed + seed_offset, is_eval=is_eval)
        return _init

    train_env = VecMonitor(DummyVecEnv([make_env(is_eval=False, seed_offset=0)]))
    val_env = VecMonitor(DummyVecEnv([make_env(is_eval=True, seed_offset=10)]))
    
    train_ppo(train_env, val_env, config, run_id)
    evaluate(run_id, config, config.eval_sizes)

if __name__ == "__main__":
    main()
