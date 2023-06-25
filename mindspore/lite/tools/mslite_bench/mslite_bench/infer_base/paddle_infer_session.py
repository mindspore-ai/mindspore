"""
for paddle infer session
"""
from abc import ABC
from typing import Dict

import paddle.inference as paddle_infer
import numpy as np

from mslite_bench.infer_base.abs_infer_session import AbcInferSession
from mslite_bench.common.model_info_enum import (
    DeviceType,
)


class PaddleSession(AbcInferSession, ABC):
    """paddle infer session"""
    def __init__(self,
                 model_file,
                 cfg,
                 params_file=None):
        super(PaddleSession, self).__init__(model_file, cfg)
        self.place = None
        self.param_file = params_file
        self.model_session = self._create_infer_session()
        self.input_names = self.model_session.get_input_names()
        self.output_names = self.model_session.get_output_names()
        self.model_inputs = self._get_input_tensor()

    def infer(self, input_data_map: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """model infer"""
        for name, input_tensor in self.model_inputs.items():
            input_data = input_data_map.get(name, None)
            input_tensor.copy_from_cpu(input_data)
        self.model_session.run()
        predict_results = {
            name: self.model_session.get_output_handle(name)
            for name in self.output_names
        }
        return predict_results

    def destroy(self):
        """model destroy"""
        self.model_session.clear_intermediate_tensor()
        self.model_session.try_shrink_memory()

    def _get_input_tensor(self):
        """get input tensor"""
        input_tensor_map = {}
        for name in self.input_names:
            input_tensor_map[name] = self.model_session.get_input_handle(name)

        return input_tensor_map

    def _create_infer_session(self):
        """create infer session"""
        config = paddle_infer.Config(self.model_file,
                                     self.param_file)
        if self.cfg.device == DeviceType.CPU.value:
            config.set_cpu_math_library_num_threads(self.cfg.thread_num)
            self.logger.info(f'Paddle infer on CPU device, thread num is '
                             f'{self.cfg.thread_num}')
        elif self.cfg.device == DeviceType.GPU.value:
            config.enable_use_gpu(self.cfg.gpu_memory_size,
                                  self.cfg.device_id)
            self.logger.info(f'Paddle infer on GPU device: {self.cfg.device_id}')
            if self.cfg.is_enable_tensorrt:
                precision_type = paddle_infer.PrecisionType.Float32
                if self.cfg.is_fp16:
                    precision_type = paddle_infer.PrecisionType.Half
                elif self.cfg.is_int8:
                    precision_type = paddle_infer.PrecisionType.Int8
                config.set_trt_dynamic_shape_info(
                    optim_input_shape=self.cfg.tensorrt_optim_input_shape,
                    min_input_shape=self.cfg.tensorrt_min_input_shape,
                    max_input_shape=self.cfg.tensorrt_max_input_shape
                )
                config.enable_tensorrt_engine(workspace_size=1 << 28,
                                              max_batch_size=self.cfg.batch_size,
                                              min_subgraph_size=1,
                                              precision_mode=precision_type,
                                              use_static=False,
                                              use_calib_mode=True)
                self.logger.info(f'Enable TensorRT is {config.tensorrt_engine_enabled()}')

        else:
            raise ValueError(f'paddle do not work on device type {self.cfg.device}')

        model_session = paddle_infer.create_predictor(config)
        return model_session
