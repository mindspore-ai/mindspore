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
""" Context management base module. """
from mindspore import context
from mindspore.communication.management import init
from mindspore.communication.management import release
from mindspore.communication.management import get_rank
import os
from mindspore import log as logger
from ._utils import clean_all_ir_files


class ContextBase:
    def __init__(self):
        self.save_graphs_flag = 0
        self.save_graphs_path = '.'

    def case_prepare(self):
        logger.info("MindSporeTest::call case-prepare !!!")
        self.set_default_context()
        self.set_context_from_env()
        self.init_parallel()


    def case_cleanup(self):
        logger.info("MindSporeTest::call case-cleanup !!!")
        self.release_parallel()
        context.reset_ps_context()
        context.reset_auto_parallel_context()
        if self.save_graphs_flag:
            clean_all_ir_files(self.save_graphs_path)

    def set_context_from_env(self):
        mode_dict = {
            'GRAPH': context.GRAPH_MODE,
            'GRAPH_MODE': context.GRAPH_MODE,
            'CONTEXT.GRAPH_MODE': context.GRAPH_MODE,
            'PYNATIVE': context.PYNATIVE_MODE,
            'PYNATIVE_MODE': context.PYNATIVE_MODE,
            'CONTEXT.PYNATIVE_MODE': context.PYNATIVE_MODE
        }
        if 'CONTEXT_MODE' in os.environ:
            mode_key = os.environ['CONTEXT_MODE']
            if mode_key in mode_dict:
                context.set_context(mode=mode_dict[mode_key])
        if 'CONTEXT_DEVICE_TARGET' in os.environ:
            context.set_context(device_target=os.environ['CONTEXT_DEVICE_TARGET'])
        if 'CONTEXT_ENABLE_SPARSE' in os.environ:
            context.set_context(enable_sparse=True)
        if "CONTEXT_JIT_LEVEL" in os.environ:
            jit_level = os.environ["CONTEXT_JIT_LEVEL"]
            if jit_level in ("O0", "O1", "O2"):
                context.set_context(jit_config={"jit_level": jit_level})
            else:
                raise ValueError(f"CONTEXT_JIT_LEVEL must be O0/O1/O2, got {jit_level}")
        ssl_enable = os.environ.get('ENABLE_SSL', 'false').lower()
        if ssl_enable == "true":
            security_ctx = {
                "config_file_path": os.path.join(os.path.dirname(__file__), "config.json"),
                "enable_ssl": True,
                "client_password": "",
                "server_password": ""
            }
            context.set_ps_context(**security_ctx)
        else:
            context.set_ps_context(enable_ssl=False)

        logger.info("MindSporeTest::set context from env success!!!")

    def set_default_context(self):
        context.reset_auto_parallel_context()
        context.set_context(mode=context.GRAPH_MODE)
        if 'CONTEXT_DEVICE_TARGET' not in os.environ:
            os.environ['CONTEXT_DEVICE_TARGET'] = 'Ascend'
        context.set_context(device_target=os.environ['CONTEXT_DEVICE_TARGET'])
        logger.info("MindSporeTest::set context from default success!!!")

    def init_parallel(self):
        if 'RANK_SIZE' in os.environ and int(os.environ['RANK_SIZE']) > 1:
            device_target = context.get_context("device_target")
            if device_target == 'Ascend':
                init(backend_name='hccl')
            elif device_target == 'GPU':
                init(backend_name='nccl')
            else:
                init()
            logger.info("MindSporeTest::init parallel success!!!")
        else:
            logger.info("MindSporeTest::single device, skip parallel init")

    def release_parallel(self):
        if 'RANK_SIZE' in os.environ and int(os.environ['RANK_SIZE']) > 1:
            release()
            logger.info("MindSporeTest::release parallel success!!!")

    def get_parallel_variable_from_env(self, attr_key):
        if attr_key in os.environ:
            return int(os.environ[attr_key])
        if attr_key == "RANK_ID":
            return get_rank() if 'RANK_SIZE' in os.environ and int(os.environ['RANK_SIZE']) > 1 else 0
        if attr_key == "DEVICE_ID":
            device_target = context.get_context("device_target")
            return get_rank() if device_target == "GPU" else os.environ.get("DEVICE_ID", 0)
        return -1

    def set_parallel_context(self, parallel_mode, dataset_strategy="data_parallel",
                           search_mode="recursive_programming", device_num=None,
                           strategy_ckpt_config=None,
                           enable_parallel_optimizer=False, **kwargs):
        if strategy_ckpt_config is None:
            strategy_ckpt_config = {"save_file": "", "load_file": "", "only_trainable_params": True}
        context.reset_auto_parallel_context()
        if device_num is None:
            context.set_auto_parallel_context(
                parallel_mode=parallel_mode,
                strategy_ckpt_config=strategy_ckpt_config,
                search_mode=search_mode,
                dataset_strategy=dataset_strategy,
                enable_parallel_optimizer=enable_parallel_optimizer,
                **kwargs
            )
        else:
            context.set_auto_parallel_context(
                parallel_mode=parallel_mode,
                device_num=device_num,
                strategy_ckpt_config=strategy_ckpt_config,
                search_mode=search_mode,
                dataset_strategy=dataset_strategy,
                enable_parallel_optimizer=enable_parallel_optimizer,
                **kwargs
            )
        logger.info("MindSporeTest::set parallel context success!!!")


contextbase = ContextBase()
