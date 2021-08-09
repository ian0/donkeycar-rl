import argparse

import gym
#import tensorflow as tf
from loguru import logger
from stable_baselines3 import HerReplayBuffer, SAC
from stable_baselines3.sac import MlpPolicy
import torch
import yaml

import gym_donkeycar
from environment.utility import seed, load_ae_controller
from environment.wrappers import make_wrappers
from environment.command import common_args, parse_args
import logging

from callbacks import TensorboardCallback

from stable_baselines3.common.envs import BitFlippingEnv

#tf.logging.set_verbosity(tf.logging.DEBUG)


def main(args: dict):

    logging.basicConfig(level=logging.INFO)
    logging.info('Started')

    vae = load_ae_controller(args["ae_path"])

    env = gym.make(args["environment_id"])
    try:
        env = make_wrappers(env, vae)

        env.metadata["video.frames_per_second"] = 10
        env = gym.wrappers.monitor.Monitor(
            env,
            directory=args["monitoring_dir"],
            force=True,
            video_callable=lambda episode: episode % 5,  # Dump every 5 episodes
        )

        with open("hyperparams/sac.yaml", "r") as f:
            hyperparams_dict = yaml.safe_load(f)
            hyperparams = hyperparams_dict[args["environment_id"]]
            logger.debug(f"Policy hyperparameters: {hyperparams}")

        test_callback = TensorboardCallback()

        policy = dict(activation_fn=torch.nn.ReLU, net_arch=[32, 32], use_sde=True)


        # model = SAC(env=env,
        #             policy=MlpPolicy,
        #             policy_kwargs=policy,
        #             buffer_size=30000,
        #             learning_starts=0,
        #             train_freq=(1, "episode"),
        #             batch_size=32,
        #             verbose=2,
        #             gradient_steps=2,
        #             learning_rate=0.0003,
        #             tensorboard_log=str(args["tensorboard_dir"]),
        #             ent_coef='auto_0.1',
        #             gamma=0.9,
        #             tau=0.001
        #             )
        model = SAC(MlpPolicy,
                    policy_kwargs=policy,
                    env=env,
                    verbose=1,
                    batch_size=64,
                    buffer_size=30000,
                    learning_starts=0,
                    gradient_steps=600,
                    train_freq=(1, "episode"),
                    ent_coef='auto_0.1',
                    learning_rate=3e-4,
                    tensorboard_log=str(args["tensorboard_dir"]),
                    gamma=0.99,
                    tau=0.02,
                    use_sde_at_warmup=True,
                    use_sde=True,
                    sde_sample_freq=64,
                    )








        #     SAC(
        #     "MlpPolicy",
        #     env,
        #     tensorboard_log=str(args["tensorboard_dir"]),
        #     verbose=1,
        #     learning_starts=0,  # default 100
        #     seed=42,
        #     **hyperparams,
        # )

        seed(42, env)

        logger.info(f"Learning. CTRL+C to quit.")
        model.learn(total_timesteps=30000, log_interval=1, callback=test_callback)
        #model.learn(total_timesteps=30000, log_interval=1)
    except KeyboardInterrupt as e:
        logging.info('Finished early')
        pass
    finally:
        # logger.info(f'Trained for {env.get_total_steps()}')
        logger.info(f'Saving model to {args["model_path"]}, don\'t quit!')
        model.save(args["model_path"])
        env.close()
        logging.info('Finished')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the SAC algorithm on the DonkeyCar environment with a VAE."
    )
    parser = common_args(parser)
    main(parse_args(parser))
