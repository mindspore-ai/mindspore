"""
for mslite infer session
"""
from abc import ABC
from typing import Dict

import mindspore_lite as mslite
import numpy as np

from mslite_bench.infer_base.abs_infer_session import AbcInferSession


class MsliteSession(AbcInferSession, ABC):
    """
    mindspore lite infer session
    """
    def __init__(self,
                 model_file,
                 cfg=None):
        super(MsliteSession, self).__init__(model_file, cfg)
        self.thread_num = cfg.thread_num
        mslite_model_type = self._set_ms_model_type()
        self.model_type = mslite.ModelType(mslite_model_type)
        self.device = cfg.device
        self.thread_affinity_mode = cfg.thread_affinity_mode
        self.context = self._init_context()
        self.model_session = self._create_infer_session()
        self.model_inputs = self.model_session.get_inputs()

    def infer(self, input_data_map: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """model infer"""
        self._check_and_resize_input_tensor(input_data_map)
        for model_input in self.model_inputs:
            tensor_name = model_input.name.rstrip()
            input_data = input_data_map.get(tensor_name, None)
            model_input.set_data_from_numpy(input_data)
        outputs = self.model_session.predict(self.model_inputs)
        predict_results = {
            tensor.name.rstrip(): tensor.get_data_to_numpy()
            for tensor in outputs
        }
        return predict_results

    def _check_and_resize_input_tensor(self, input_data_map):
        """check and resize input tensor"""
        is_need_reshape = False
        input_shape_list = []

        for model_input in self.model_inputs:
            tensor_name = model_input.name.rstrip()
            input_data = input_data_map.get(tensor_name, None)
            if input_data is None:
                raise ValueError(f'{tensor_name} is not in model inputs')
            if model_input.shape != list(input_data.shape):
                self.logger.warning(f'model input shape: {model_input.shape} is not equal'
                                    f'with input data shape: {input_data.shape}, model input shape'
                                    f'would be reshaped')
                is_need_reshape = True
            input_shape_list.append(list(input_data.shape))

        if is_need_reshape:
            self.model_session.resize(self.model_inputs, input_shape_list)
            self.model_inputs = self.model_session.get_inputs()

    def _create_infer_session(self):
        """create mslite infer session"""
        model_session = mslite.Model()
        model_session.build_from_file(self.model_file,
                                      self.model_type,
                                      self.context)
        return model_session

    def _get_input_tensor_infos(self):
        """get infos about input tensors"""
        input_tensor_infos = {}
        tensor_shape_list = []
        resize_tensor_list = []
        for input_tensor in self.model_inputs:
            tensor_name = input_tensor.name.rstrip()
            dtype = input_tensor.dtype
            shape = input_tensor.shape
            if -1 in shape or not shape:
                resize_tensor_list.append(input_tensor)
                shape = self.input_tensor_shapes.get(tensor_name, None)
                tensor_shape_list.append(list(shape))
            input_tensor_infos[tensor_name] = (shape, dtype)

        if not resize_tensor_list:
            self.logger.info(f'resize model input tensor '
                             f'shape {resize_tensor_list} to {tensor_shape_list}')
            self.model_session.resize(resize_tensor_list, tensor_shape_list)
            self.model_inputs = self.model_session.get_inputs()

        return input_tensor_infos

    def _init_context(self):
        """init mslite context"""
        context = mslite.Context()
        context.target = [self.device]
        if self.device == 'ascend':
            context.ascend.device_id = 0
            context.provider = self.cfg.ascend_provider
        context.cpu.thread_num = self.thread_num
        context.cpu.thread_affinity_mode = self.thread_affinity_mode
        return context

    def _set_ms_model_type(self):
        """set mslite model type"""
        if self.model_file.endswith('ms'):
            mslite_model_type = 4
        else:
            self.mslite_model_type = 0
        return mslite_model_type
