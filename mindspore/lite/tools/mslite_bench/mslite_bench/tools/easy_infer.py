# Copyright 2023 Huawei Technologies Co., Ltd
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
functions for model easy infer
"""
import time
import random

import numpy as np

from mslite_bench.common.config import ModelConfig
from mslite_bench.utils import InferLogger
from mslite_bench.infer_base import InferSessionFactory
from mslite_bench.common.task_common_func import CommonFunc
from mslite_bench.common.model_info_enum import SaveFileType


class EasyInfer:
    """
    functions for model easy infer
    """
    @staticmethod
    def easy_infer(args, logger=None):
        """model easy infer"""
        if logger is None:
            logger = InferLogger(args.log_path).logger
        output_data_dir = None
        model_path = args.model_file
        param_path = args.params_file

        cfg = CommonFunc.get_framework_config(model_path,
                                              args)
        if args.input_data_file is not None:
            if args.input_data_file.endswith(SaveFileType.NPY.value):
                input_data_map = np.load(args.input_data_file, allow_pickle=True).item()
            else:
                input_data_map = np.fromfile(args.input_data_file)
            output_data_dir = f'{args.input_data_file}_output'

            cfg.input_tensor_shapes = {
                key: value.shape for key, value in input_data_map.items()
            }
        else:
            input_data_map = CommonFunc.create_numpy_data_map(args)
            input_data_file = f'{args.model_file}_input'
            output_data_dir = f'{args.model_file}_output'
            if args.save_file_type == SaveFileType.NPY.value:
                np.save(f'{input_data_file}.npy', input_data_map)
            elif args.save_file_type == SaveFileType.BIN.value:
                for key, value in input_data_map.items():
                    value.tofile(f'{input_data_file}_{"".join(key.split("/"))}.bin')
            else:
                output_data_dir = None

        model_session = InferSessionFactory.create_infer_session(model_path,
                                                                 cfg,
                                                                 params_file=param_path)
        logger.debug('Create model session success')

        for _ in range(args.warmup_times):
            outputs = model_session(input_data_map)

        start = time.time()
        for _ in range(args.loop_infer_times):
            outputs = model_session(input_data_map)
        end = time.time()
        if args.loop_infer_times != 0:
            logger.info('Model Infer %s times, '
                        'Avg infer time is %s ms',
                        args.loop_infer_times,
                        round((end - start) / args.loop_infer_times * 1000, 3))

        if output_data_dir is not None:
            if args.save_file_type == SaveFileType.NPY.value:
                np.save(output_data_dir, outputs)
            else:
                CommonFunc.save_output_as_benchmark_txt(output_data_dir,
                                                        outputs)
        return outputs

    @staticmethod
    def ms_dynamic_input_infer(args, logger=None):
        """conduct dynamic shape mindspore lite model infer"""
        if logger is None:
            logger = InferLogger(args.log_path).logger

        cfg = ModelConfig(device=args.device)
        ms_session = InferSessionFactory.create_infer_session(args.model_path,
                                                              cfg)
        model_inputs = ms_session.get_input()

        for _ in range(args.dynamic_infer_times):
            input_tensor_infos = {}
            random_batch_size = random.randint(args.min_random_batch_size,
                                               args.max_random_batch_size)
            for input_tensor in model_inputs:
                input_shape = input_tensor.shape
                input_shape[0] = random_batch_size
                tensor_name = input_tensor.name.rstrip()
                input_tensor_infos[tensor_name] = (input_shape, input_tensor.dtype)
            input_tensor_map = CommonFunc.create_numpy_data_map(input_tensor_infos)
            _ = ms_session(input_tensor_map)

        logger.debug('All dynamic input passed successfully')
