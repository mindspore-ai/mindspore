# Copyright 2021 Huawei Technologies Co., Ltd
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
"""
Package initialization for common utils.
"""

import logging
logger = logging.getLogger('utils')


def init_utils(config):
    """
    Initialize common utils.

    Args:
        config (Config): Contains configurable parameters throughout the
                         project.
    """
    init_logging(config.log_level)
    for k, v in vars(config).items():
        logger.info('%s=%s', k, v)


def init_logging(level=logging.INFO):
    """
    Initialize logging formats.

    Args:
        level (logging level constant): Set to mute logs with lower levels.
    """
    FMT = r'[%(asctime)s][%(name)s][%(levelname)8s] %(message)s'
    DATE_FMT = r'%Y-%m-%d %H:%M:%S'
    logging.basicConfig(format=FMT, datefmt=DATE_FMT, level=level)
