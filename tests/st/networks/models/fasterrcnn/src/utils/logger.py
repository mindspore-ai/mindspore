# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Custom Logger."""
import os
import sys
import logging
from datetime import datetime

__all__ = ["get_logger"]

GLOBAL_LOGGER = None


class Logger(logging.Logger):
    """
    Logger classes and functions, support print information on console and files.

    Args:
         logger_name(str): The name of Logger. In most cases, it can be the name of the network.
    """

    def __init__(self, logger_name="fasterrcnn"):
        super(Logger, self).__init__(logger_name)
        self.log_level = "INFO"
        self.rank_id = _get_rank_id()
        self.device_per_servers = 8
        self.formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")


def setup_logging(logger_name="fasterrcnn", log_level="INFO", rank_id=None, device_per_servers=8):
    """Setup logging file."""
    logger = get_logger()
    logger.name = logger_name
    logger.log_level = log_level
    if rank_id is not None:
        logger.rank_id = rank_id
    logger.device_per_servers = device_per_servers

    if logger.log_level not in ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]:
        raise ValueError(
            f"Not support log_level: {logger.log_level}, "
            f"the log_level should be in ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']"
        )

    # In the distributed scenario, only one card is printed on the console.
    if logger.rank_id % logger.device_per_servers == 0:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logger.log_level)
        console.setFormatter(logger.formatter)
        logger.addHandler(console)


def setup_logging_file(log_dir="./logs"):
    """Setup logging file."""
    logger = get_logger()
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Generate a file stream based on the log generation time and rank_id
    log_name = f"{logger.name}_{datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S')}_rank_{logger.rank_id}.log"
    log_path = os.path.join(log_dir, log_name)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logger.log_level)
    file_handler.setFormatter(logger.formatter)
    logger.addHandler(file_handler)


def print_args(args):
    """Print hyper-parameter"""
    get_logger().info("Args:")
    args_dict = vars(args)
    for key in args_dict.keys():
        get_logger().info("--> %s: %s", key, args_dict[key])
    get_logger().info("")


def important_info(msg, *args, **kwargs):
    """For information that needs to be focused on, add special printing format."""
    line_width = 2
    important_msg = "\n"
    important_msg += ("*" * 70 + "\n") * line_width
    important_msg += ("*" * line_width + "\n") * 2
    important_msg += "*" * line_width + " " * 8 + msg + "\n"
    important_msg += ("*" * line_width + "\n") * 2
    important_msg += ("*" * 70 + "\n") * line_width
    get_logger().info(important_msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    """
    Log a message with severity 'INFO' on the logger.

    Examples:
        >>> logger.setup_logging(logger_name="fasterrcnn", log_level="INFO", rank_id=0, device_per_servers=8)
        >>> logger.setup_logging_file(log_dir="./logs")
        >>> logger.info("test info")
    """
    get_logger().info(msg, *args, **kwargs)


def debug(msg, *args, **kwargs):
    """Log a message with severity 'DEBUG' on the logger."""
    get_logger().debug(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    """Log a message with severity 'ERROR' on the logger."""
    get_logger().error(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    """Log a message with severity 'WARNING' on the logger."""
    get_logger().warning(msg, *args, **kwargs)


def critical(msg, *args, **kwargs):
    """Log a message with severity 'CRITICAL' on the logger."""
    get_logger().critical(msg, *args, **kwargs)


def get_level():
    """
    Get the logger level.

    Returns:
        str, the Log level includes 'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'.
    """
    # level and glog level mapping dictionary

    return get_logger().log_level


def _get_rank_id():
    """Get rank id."""
    rank_id = os.getenv("RANK_ID")
    gpu_rank_id = os.getenv("OMPI_COMM_WORLD_RANK")
    rank = "0"
    if rank_id and gpu_rank_id and rank_id != gpu_rank_id:
        print(
            f"Environment variables RANK_ID and OMPI_COMM_WORLD_RANK set by different values, RANK_ID={rank_id}, "
            f"OMPI_COMM_WORLD_RANK={gpu_rank_id}. We will use RANK_ID to get rank id by default.",
            flush=True,
        )
    if rank_id:
        rank = rank_id
    elif gpu_rank_id:
        rank = gpu_rank_id
    return int(rank)


def get_logger():
    """Get logger instance."""
    global GLOBAL_LOGGER
    if GLOBAL_LOGGER:
        return GLOBAL_LOGGER
    GLOBAL_LOGGER = Logger()
    return GLOBAL_LOGGER
