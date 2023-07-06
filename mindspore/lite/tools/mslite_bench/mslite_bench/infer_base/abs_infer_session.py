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
        """property input infos"""
        return self.input_tensor_infos

    @property
    def dtype_class(self):
        """property dtype class"""
        return self.data_type_class

    @abstractmethod
    def infer(self, input_data_map: Dict[str, np.ndarray]):
        """start model infer"""
        raise NotImplementedError

    @abstractmethod
    def _create_infer_session(self):
        """create model infer"""
        raise NotImplementedError
