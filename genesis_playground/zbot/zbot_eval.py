# """ ZBot evaluation

# Run:
#     python genesis_playground/zbot/zbot_eval.py -e zbot-walking --show_viewer True --num_envs 1 --device mps
# """
# import argparse
# import os
# import pickle

# import torch
# from zbot_env import ZbotEnv
# from rsl_rl.runners import OnPolicyRunner

# import genesis as gs


# def run_sim(env, policy, obs):
#     timesteps = 0
#     while timesteps < 1000:
#         actions = policy(obs)
#         obs, _, rews, dones, infos = env.step(actions)
#         timesteps += 1
#         if timesteps > 1000:
#             exit()


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-e", "--exp_name", type=str, default="zeroth-walking")
#     parser.add_argument("--ckpt", type=int, default=100)
#     parser.add_argument("--show_viewer", type=bool, default=False)
#     parser.add_argument("--num_envs", type=int, default=1)
#     parser.add_argument("--device", type=str, default="mps")
#     args = parser.parse_args()

#     gs.init()

#     log_dir = f"logs/{args.exp_name}"
#     env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
#     # reward_cfg["reward_scales"] = {}

#     env = ZbotEnv(
#         num_envs=args.num_envs,
#         env_cfg=env_cfg,
#         obs_cfg=obs_cfg,
#         reward_cfg=reward_cfg,
#         command_cfg=command_cfg,
#         show_viewer=args.show_viewer,
#         device=args.device,
#     )

#     runner = OnPolicyRunner(env, train_cfg, log_dir, device=args.device)
#     resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
#     runner.load(resume_path)
#     policy = runner.get_inference_policy(device=args.device)

#     obs, _ = env.reset()
#     with torch.no_grad():
#         # if on mac
#         if torch.backends.mps.is_available():
#             gs.tools.run_in_another_thread(fn=run_sim, args=(env, policy, obs)) # start the simulation in another thread
#         else:
#             run_sim(env, policy, obs)
        
#         env.scene.viewer.start() # start the viewer in the main thread (the render thread)


# if __name__ == "__main__":
#     main()

""" ZBot evaluation with keyboard controls using pygame. """

import argparse
import os
import pickle
import threading
import time
import numpy as np
import torch
import pygame  # pip install pygame
from zbot_env import ZbotEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

# Global dictionary to hold user commands for x, y, yaw.
USER_CMD = {"x": 0.0, "y": 0.0, "yaw": 0.0}
INCREMENT = 0.6

def init_pygame_window():
    """Initialize a small pygame window so that we can capture keyboard events."""
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("Keyboard Control")
    return screen

def handle_pygame_events():
    """
    Checks the pygame event queue and updates USER_CMD accordingly.
    W/S: forward/back
    A/D: left/right
    Q/E: yaw left/right
    """
    global USER_CMD
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # If user closes the pygame window, exit
            pygame.quit()
            exit(0)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                USER_CMD["x"] += INCREMENT
            elif event.key == pygame.K_s:
                USER_CMD["x"] -= INCREMENT
            elif event.key == pygame.K_a:
                USER_CMD["y"] += INCREMENT
            elif event.key == pygame.K_d:
                USER_CMD["y"] -= INCREMENT
            elif event.key == pygame.K_q:
                USER_CMD["yaw"] += INCREMENT
            elif event.key == pygame.K_e:
                USER_CMD["yaw"] -= INCREMENT
        elif event.type == pygame.KEYUP:
            # Optionally revert the increment to zero out the command if desired
            if event.key == pygame.K_w:
                USER_CMD["x"] -= INCREMENT
            elif event.key == pygame.K_s:
                USER_CMD["x"] += INCREMENT
            elif event.key == pygame.K_a:
                USER_CMD["y"] -= INCREMENT
            elif event.key == pygame.K_d:
                USER_CMD["y"] += INCREMENT
            elif event.key == pygame.K_q:
                USER_CMD["yaw"] -= INCREMENT
            elif event.key == pygame.K_e:
                USER_CMD["yaw"] += INCREMENT

def keyboard_control_policy(obs: torch.Tensor) -> torch.Tensor:
    """
    Returns actions for the robot based on USER_CMD (x, y, yaw).
    Adjust to fit your robot’s DOF layout and indexing.
    """

    x_cmd = USER_CMD["x"]
    y_cmd = USER_CMD["y"]
    yaw_cmd = USER_CMD["yaw"]

    # For forward/back (pitch)
    obs[:, 6] = x_cmd
    obs[:, 7] = y_cmd
    obs[:, 8] = yaw_cmd

    return obs

def run_sim(env, policy, obs, use_keyboard=False):
    """
    Runs the simulation loop for a fixed timespan.
    If use_keyboard=True, we override the policy with keyboard_control_policy().
    """
    screen = None
    if use_keyboard:
        screen = init_pygame_window()

    timesteps = 0
    max_timesteps = 2000

    while timesteps < max_timesteps:
        # If using keyboard, poll pygame events
        if use_keyboard:
            handle_pygame_events()
            obs = keyboard_control_policy(obs)
        
        print(obs[:, 6:9])
        actions = policy(obs)

        obs, _, rews, dones, infos = env.step(actions)
        timesteps += 1

        # If using pygame, maybe fill the screen black each frame:
        if use_keyboard and screen is not None:
            screen.fill((0, 0, 0))
            pygame.display.flip()

        if timesteps >= max_timesteps:
            if use_keyboard and screen is not None:
                pygame.quit()
            exit(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="zeroth-walking")
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument("--show_viewer", action='store_true')
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--keyboard_control", action='store_true',
                        help="Enable manual keyboard control, overriding policy.")
    args = parser.parse_args()

    gs.init()
    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(
        open(f"{log_dir}/cfgs.pkl", "rb")
    )

    # Create the environment
    env = ZbotEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.show_viewer,
        device=args.device,
    )

    # Load a trained policy
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=args.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=args.device)

    obs, _ = env.reset()

    with torch.no_grad():
        # If you’re using Apple M1’s MPS backend, we can optionally run sim in another thread via gs.tools.run_in_another_thread. 
        # But if we want to handle pygame events in the main thread, we can just do run_sim directly:
        run_sim(env, policy, obs, args.keyboard_control)

if __name__ == "__main__":
    main()