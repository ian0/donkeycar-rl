import argparse
from pathlib import Path

from loguru import logger

env_list = [
    "donkey-warehouse-v0",
    "donkey-generated-roads-v0",
    "donkey-avc-sparkfun-v0",
    "donkey-generated-track-v0",
]


def common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Common arguments for all files."""
    parser.add_argument(
        "--environment_id",
        type=str,
        default="donkey-generated-roads-v0",
        choices=env_list,
        help="Gym Environment ID (e.g. donkey-generated-roads-v0).",
    )
    parser.add_argument(
        "--ae_path",
        help="Path to VAE model (pkl)",
        type=str,
        default="pretrained-models/autoencoder/vae-donkey-generated-roads-v0-32.pkl",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default="pretrained-models/policy/policy-donkey-generated-roads-v0-32.zip",
        help="Path to the policy model (zip file).",
    )
    parser.add_argument(
        "--tensorboard_dir",
        help="Tensorboard log dir (dir).",
        default="./log/",
        type=Path,
    )
    parser.add_argument(
        "--monitoring_dir",
        type=Path,
        default="./monitoring/",
        help="Output directory (dir). WILL BE OVERWRITTEN!",
    )
    return parser


def parse_args(parser) -> dict:
    args = vars(parser.parse_args())
    logger.debug(
        f'Training SAC with the following configuration: \
--environment_id={args["environment_id"]} \
--ae_path={args["ae_path"]} \
--model_path={args["model_path"]} \
--tensorboard_dir={args["tensorboard_dir"]} \
--monitoring_dir={args["monitoring_dir"]} \
    '
    )
    return args
