"""
configs for mslite bench
"""
from dataclasses import dataclass
from typing import List, Tuple, Dict


from mslite_bench.common.model_info_enum import FrameworkType


@dataclass
class Config:
    """base config"""
    device: str = 'cpu'
    device_id: int = 0
    log_path: str = None
    batch_size: int = 1


class ModelConfig(Config):
    """model config"""
    infer_framework: FrameworkType = FrameworkType.MSLITE.value
    thread_num: int = 1
    input_tensor_shapes: Dict[str, Tuple] = None
    input_tensor_dtypes: Dict[str, str] = None
    output_tensor_names: List[str] = None


@dataclass
class MsliteConfig(ModelConfig):
    """mslite config"""
    thread_affinity_mode: int = 2

    ascend_provider: str = ''


@dataclass
class PaddleConfig(ModelConfig):
    """paddle config"""
    infer_framework = FrameworkType.PADDLE.value
    is_fp16: bool = False
    is_int8: bool = False

    # for paddle infer
    is_enable_tensorrt: bool = False
    gpu_memory_size: int = 100
    tensorrt_optim_input_shape: Dict[str, List[int]] = None
    tensorrt_min_input_shape: Dict[str, List[int]] = None
    tensorrt_max_input_shape: Dict[str, List[int]] = None


@dataclass
class OnnxConfig(ModelConfig):
    """onnx config"""
    # for onnx export
    infer_framework = FrameworkType.ONNX.value


@dataclass
class TFConfig(ModelConfig):
    """tensorflow config"""
    infer_framework = FrameworkType.TF.value


@dataclass
class BenchConfig(Config):
    """benchmark config"""
    eps: float = 1e-5
    random_input_flag: bool = False
    cmp_model_file: str = None
    input_data_file: str = None
