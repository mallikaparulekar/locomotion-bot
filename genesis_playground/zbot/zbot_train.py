""" ZBot training

Run:
    python genesis_playground/zbot/zbot_train.py --num_envs 4096 --max_iterations 200 --device mps 
    (or cuda)
"""

import argparse
import os
import pickle
import shutil
import wandb
from datetime import datetime
from collections import deque

from zbot_env import ZbotEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 48,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "OnPolicyRunner",
            "save_interval": 100,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 10,
        # joint/link names
        # NOTE: hip roll/yaw flipped between sim & real robot FIXME
        # old
        # "default_joint_angles": {  # [rad]
        #     "R_Hip_Pitch": 0.0,
        #     "L_Hip_Pitch": 0.0,
        #     "R_Hip_Yaw": 0.0,
        #     "L_Hip_Yaw": 0.0,
        #     "R_Hip_Roll": 0.0,
        #     "L_Hip_Roll": 0.0,
        #     "R_Knee_Pitch": 0.0,
        #     "L_Knee_Pitch": 0.0,
        #     "R_Ankle_Pitch": 0.0,
        #     "L_Ankle_Pitch": 0.0,
        # },
        # "dof_names": [
        #     "R_Hip_Pitch",
        #     "L_Hip_Pitch",
        #     "R_Hip_Yaw",
        #     "L_Hip_Yaw",
        #     "R_Hip_Roll",
        #     "L_Hip_Roll",
        #     "R_Knee_Pitch",
        #     "L_Knee_Pitch",
        #     "R_Ankle_Pitch",
        #     "L_Ankle_Pitch",
        # ],
         #New
        "default_joint_angles": {
             "right_hip_pitch": 0.0,
             "left_hip_pitch": 0.0,
             "right_hip_yaw": 0.0,
             "left_hip_yaw": 0.0,
             "right_hip_roll": 0.0,
             "left_hip_roll": 0.0,
             "right_knee": 0.0,
             "left_knee": 0.0,
             "right_ankle": 0.0,
             "left_ankle": 0.0,
        },
        "dof_names": [
             "right_hip_pitch",
             "left_hip_pitch",
             "right_hip_yaw",
             "left_hip_yaw",
             "right_hip_roll",
             "left_hip_roll",
             "right_knee",
             "left_knee",
             "right_ankle",
             "left_ankle",
         ],
        # friction
        "env_friction_range": {
            "start": [0.9, 1.1],
            "end": [0.9, 1.1], # Original [0.9, 1.1], Other [0.8, 1.2]
        },
        # link mass
        # varying this too much collapses the training (IMPORTANT TO DISCUSS)
        "link_mass_multipliers": {
            "start": [1.0, 1.0],
            "end": [1.0, 1.0], # Original [1.0, 1.0], Other [0.9, 1.1], Other [0.95, 1.05]
        },
        # RFI
        "rfi_scale": 0.1,
        # PD
        "kp": 17.8, # original 20.0
        "kd": 0.0, # original 0.5
        "kp_multipliers": [0.75, 1.25],
        "kd_multipliers": [0.75, 1.25],
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.41],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
        "max_torque": 10.0,
    }
    obs_cfg = {
        # need to update whenever a minimal policy is used
        "num_obs": 36, # original 39
        # FIXME: IMU mounting orientation is different between sim & real robot
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0, # original 1.0 
            "tracking_ang_vel": 0.2, # original 0.2
            "lin_vel_z": -1.0, # original -1.0
            "base_height": -50.0, # original -50.0
            "action_rate": -0.005, # original -0.005
            "similar_to_default": -0.1, # original -0.1
            "feet_air_time": 5.0, # original 5.0
            # "lateral_stability": 5.0,  # Penalizes side-to-side movement (Function outputs a negative)
            # "step_phase_coherence": 1.0,  # Rewards proper alternating gait (new)
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_y_range": [0.0, 0.0],
        "lin_vel_x_range": [-0.2, 0.4],
        "ang_vel_range": [-0.4, 0.4],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg

class WandbOnPolicyRunner(OnPolicyRunner):
    def log(self, info):
        super().log(info)
        
        # Skip logging if this is just initialization (no training metrics yet)
        if 'it' not in info or info['it'] == 0:
            return
        
        # Basic metrics that are always available
        metrics = {
            'train/iteration': info['it']
        }
        
        # Convert deque objects to numpy arrays for mean calculation
        if 'rewbuffer' in info:
            if hasattr(info['rewbuffer'], 'mean'):  # If it's a tensor or similar
                metrics['train/mean_reward'] = info['rewbuffer'].mean().item()
            else:  # If it's a deque
                rewards = list(info['rewbuffer'])
                if rewards:
                    metrics['train/mean_reward'] = sum(rewards) / len(rewards)
        
        if 'lenbuffer' in info:
            if hasattr(info['lenbuffer'], 'mean'):  # If it's a tensor or similar
                metrics['train/mean_episode_length'] = info['lenbuffer'].mean().item()
            else:  # If it's a deque
                lengths = list(info['lenbuffer'])
                if lengths:
                    metrics['train/mean_episode_length'] = sum(lengths) / len(lengths)
        
        # Add policy training metrics if available
        if 'mean_value_loss' in info:
            metrics['train/value_loss'] = info['mean_value_loss']
        if 'mean_surrogate_loss' in info:
            metrics['train/policy_loss'] = info['mean_surrogate_loss']
            
        # Add learning rate
        metrics['train/learning_rate'] = self.alg.learning_rate
        
        # Add curriculum metrics if available in extras
        if hasattr(self.env, 'extras') and 'curriculum' in self.env.extras:
            curriculum_data = self.env.extras['curriculum']
            if 'mean_reward' in curriculum_data:
                metrics['curriculum/mean_reward'] = curriculum_data['mean_reward']
            if 'current_friction' in curriculum_data:
                metrics['curriculum/friction'] = curriculum_data['current_friction']
            if 'current_mass_mult' in curriculum_data:
                metrics['curriculum/mass_mult'] = curriculum_data['current_mass_mult']
        
        # Print and log the metrics
        print(f"Logging metrics to wandb: {metrics}")
        wandb.log(metrics)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="zbot-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=10)
    parser.add_argument("--max_iterations", type=int, default=300)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--show_viewer", type=bool, default=False)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--use_wandb", type=bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="zbot-walking", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name. If None, will use exp_name")
    parser.add_argument("--from_checkpoint", type=bool, default=False)
    args = parser.parse_args()
    
    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if not args.from_checkpoint:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)

    # Resume training from checkpoint
    if args.from_checkpoint:
        exp_log_dir = f"logs/{args.exp_name}"  # Ensure it's per experiment
        checkpoint_files = [
            f for f in os.listdir(exp_log_dir) if f.startswith("model_") and f.endswith(".pt")
        ]
        
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda f: int(f.split("_")[1].split(".")[0]))
            train_cfg["runner"]["resume"] = True
            train_cfg["runner"]["resume_path"] = os.path.join(exp_log_dir, latest_checkpoint)
            print(f"Resuming training from {train_cfg['runner']['resume_path']}")
        else:
            print(f"No checkpoint found for experiment '{args.exp_name}'! Starting training from scratch.")


    env = ZbotEnv(
        num_envs=args.num_envs, 
        env_cfg=env_cfg, 
        obs_cfg=obs_cfg, 
        reward_cfg=reward_cfg, 
        command_cfg=command_cfg, 
        device=args.device,
        show_viewer=args.show_viewer,
    )

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )
    
    if args.use_wandb:
        run = wandb.init(
                project=args.wandb_project,  # Use wandb_project instead of exp_name for project
                entity=args.wandb_entity,
                name=args.wandb_run_name if args.wandb_run_name else f"{args.exp_name}",  # Get rid of timestamp
                group=args.exp_name,  # Use exp_name as group name
                config={
                    "num_envs": args.num_envs,
                    "max_iterations": args.max_iterations,
                    "device": args.device,
                    "env_cfg": env_cfg,
                    "obs_cfg": obs_cfg,
                    "reward_cfg": reward_cfg,
                    "command_cfg": command_cfg,
                    "train_cfg": train_cfg,
                }
        )
        runner = WandbOnPolicyRunner(env, train_cfg, log_dir, device=args.device)
    else:
        runner = OnPolicyRunner(env, train_cfg, log_dir, device=args.device)
    
    # ✅ Load the latest checkpoint if resuming training
    if args.from_checkpoint and train_cfg["runner"]["resume_path"] is not None:
        print(f"Loading model weights from: {train_cfg['runner']['resume_path']}")
        
        # ✅ Load checkpoint (model + optimizer + iteration number)
        infos = runner.load(train_cfg["runner"]["resume_path"])
        print(f"Checkpoint successfully loaded from {train_cfg['runner']['resume_path']}")
        
        # ✅ Restore episode reward and length buffers (to continue logging properly)
        if infos and "rewbuffer" in infos and "lenbuffer" in infos:
            runner.rewbuffer = deque(infos["rewbuffer"], maxlen=100)
            runner.lenbuffer = deque(infos["lenbuffer"], maxlen=100)
            print("Restored reward and length buffers.")


    try:
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    finally:
        if args.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
