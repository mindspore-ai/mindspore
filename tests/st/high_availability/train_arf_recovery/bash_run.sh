#!/bin/bash
# Copyright 2025 Huawei Technologies Co., Ltd
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

export MS_ENABLE_RECOVERY=1
export MS_DEV_RUNTIME_CONF="async_init_comm:False"
export MS_ENABLE_TFT="{RSC:1}"
export MS_ENABLE_THM="{HCCL_WATCHDOG:0}"

msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 --master_port=10971 --join=True --log_dir=./logs "resuming_interface.py"

