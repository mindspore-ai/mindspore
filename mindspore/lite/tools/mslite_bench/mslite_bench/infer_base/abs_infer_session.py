"""abstract infer session for mslite bench"""
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from mslite_bench.utils import InferLogger


class AbcInferSession(ABC):
    """
    abstract infer session
    """
    def __init__(self,
                 model_file: str,
                 cfg=None):
        self.model_file = model_file
        self.cfg = cfg
        self.input_tensor_shapes = cfg.input_tensor_shapes
        self.output_tensor_names = cfg.output_tensor_names
        self.batch_size = cfg.batch_size
        self.logger = InferLogger(file_path=cfg.log_path)
        self.data_type_class = None
        self.input_tensor_infos = None

    def __call__(self, *args, **kwargs):
        return self.infer(*args, **kwargs)

    @property
    def input_infos(self):
        return self.input_tensor_infos

    @property
    def dtype_class(self):
        return self.data_type_class

    @abstractmethod
    def infer(self, input_data_map: Dict[str, np.ndarray]):
        raise NotImplementedError

    @abstractmethod
    def _create_infer_session(self):
        raise NotImplementedError

    def _nchw2nhwc(self, data: np.ndarray) -> np.ndarray:
        shape_dim = 4
        ret_data = data
        if len(data.shape) == shape_dim:
            ret_data = np.transpose(data, (0, 2, 3, 1))

        return ret_data

    def _reshape_nchw2nhwc(self, shape):
        out_shape = shape
        nchw_dim_num = 4
        if len(shape) == nchw_dim_num:
            out_shape = (
                out_shape[0], out_shape[2], out_shape[3], out_shape[1]
            )
        return out_shape
