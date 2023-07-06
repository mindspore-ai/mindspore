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
            logger = InferLogger(args.log_path)
        output_data_dir = None
        model_path = args.model_file
        param_path = args.params_file

        cfg = CommonFunc.get_framework_config(model_path,
                                              args)
        if args.input_data_file is not None:
            input_data_map = np.load(args.input_data_file, allow_pickle=True).item()
            output_data_dir = f'{args.input_data_file}.bin'

            cfg.input_tensor_shapes = {
                key: value.shape for key, value in input_data_map.items()
            }
        else:
            input_data_map = CommonFunc.create_numpy_data_map(args)

        model_session = InferSessionFactory.create_infer_session(model_path,
                                                                 cfg,
                                                                 params_file=param_path)
        logger.info('[MODEL INFER] Create model session success')

        for _ in range(args.warmup_times):
            outputs = model_session(input_data_map)

        start = time.time()
        for _ in range(args.loop_infer_times):
            outputs = model_session(input_data_map)
        end = time.time()
        if args.loop_infer_times != 0:
            logger.info(f'Model Infer {args.loop_infer_times} times, '
                        f'Avg infer time is '
                        f'{round((end - start) / args.loop_infer_times * 1000, 3)} ms')
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
            logger = InferLogger(args.log_path)

        cfg = ModelConfig(device=args.device)
        ms_session = InferSessionFactory.create_infer_session(args.model_path,
                                                              cfg)
        model_inputs = ms_session.get_input()

        logger.info(f'Start Dynamic model infer: {args.model_path},'
                    f'infer times is {args.dynamic_infer_times}')

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

        logger.info('All dynamic input passed successfully')
