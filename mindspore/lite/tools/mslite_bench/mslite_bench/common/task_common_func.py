"""
common functions
"""
import os
import stat
from typing import Dict, Tuple

import numpy as np

from mslite_bench.common.model_info_enum import FrameworkType
from mslite_bench.common.enum_class import NumpyDtype
from mslite_bench.common.config import (
    MsliteConfig, TFConfig, PaddleConfig, OnnxConfig
)


class CommonFunc:
    """common functions"""
    @classmethod
    def get_framework_config(cls,
                             model_path,
                             args):
        """
        get framework config by model type and args
        params:
        model_path: path to model file
        args: input arguments
        return: model config
        """
        if not os.path.exists(model_path):
            raise ValueError(f'Create model session failed: {model_path} does not exist')

        if model_path.endswith('pb'):
            cfg = cls.init_tf_cfg()
        elif model_path.endswith('onnx'):
            cfg = cls.init_onnx_cfg()
        elif model_path.endswith('ms') or model_path.endswith('mindir'):
            cfg = cls.init_mslite_cfg(args, model_path)
        elif model_path.endswith('pdmodel'):
            cfg = cls.init_paddle_cfg(args)
        else:
            raise ValueError(f'model {model_path} is not supported yet')

        cfg.input_tensor_shapes = cls.get_tensor_shapes(args.input_tensor_shapes)
        cfg.device = args.device
        cfg.device_id = args.device_id
        cfg.batch_size = args.batch_size
        cfg.output_tensor_names = args.output_tensor_names
        cfg.thread_num = args.thread_num

        if cfg.input_tensor_shapes is None and args.input_data_file is not None:
            input_data_map = cls.get_input_data_map_from_file(args.input_data_file)
            cfg.input_tensor_shapes = {
                key: val.shape for key, val in input_data_map.items()
            }

        return cfg

    @classmethod
    def create_numpy_data_map(cls,
                              args):
        """
        create input tensor map, with key input tensor name,
        value its numpy value
        """
        if args.input_data_file is not None:
            input_data_map = np.load(args.input_data_file, allow_pickle=True).item()
            return input_data_map

        input_tensor_dtypes = CommonFunc.parse_dtype_infos(args.input_tensor_dtypes)
        input_tensor_shapes = CommonFunc.get_tensor_shapes(args.input_tensor_shapes)
        input_tensor_infos = {
            key: (shape, input_tensor_dtypes.get(key))
            for key, shape in input_tensor_shapes.items()
        }
        try:
            input_tensor_map = cls.create_numpy_data_map_out(input_tensor_infos)
        except ValueError as e:
            raise e

        return input_tensor_map

    @classmethod
    def init_onnx_cfg(cls):
        """init onnx config"""
        cfg = OnnxConfig()
        return cfg

    @classmethod
    def init_mslite_cfg(cls, args, model_path):
        """init mslite config"""
        cfg = MsliteConfig()
        cfg.infer_framework = FrameworkType.MSLITE.value
        cfg.mslite_model_type = 4 if model_path.endswith('ms') else 0
        cfg.thread_affinity_mode = args.thread_affinity_mode
        cfg.ascend_provider = args.ascend_provider
        return cfg

    @classmethod
    def init_paddle_cfg(cls, args):
        """init paddle config"""
        cfg = PaddleConfig()
        cfg.infer_framework = FrameworkType.PADDLE.value
        cfg.is_fp16 = args.is_fp16
        cfg.is_int8 = args.is_int8
        cfg.is_enable_tensorrt = args.is_enable_tensorrt
        tmp_func = lambda x: None if x is None else cls.get_tensor_shapes(x)
        cfg.tensorrt_optim_input_shape = tmp_func(args.tensorrt_optim_input_shape)
        cfg.tensorrt_min_input_shape = tmp_func(args.tensorrt_min_input_shape)
        cfg.tensorrt_max_input_shape = tmp_func(args.tensorrt_max_input_shape)
        if cfg.tensorrt_min_input_shape is None:
            cfg.tensorrt_min_input_shape = cfg.tensorrt_optim_input_shape
        if cfg.tensorrt_max_input_shape is None:
            cfg.tensorrt_max_input_shape = cfg.tensorrt_optim_input_shape
        return cfg

    @staticmethod
    def get_tensor_shapes(tensor_shapes: str) -> Dict[str, Tuple[int]]:
        """parse tensor shapes string into dict"""
        if tensor_shapes is None:
            return None

        input_tensor_shape = {}
        shape_list = tensor_shapes.split(';')

        for shapes in shape_list:
            name, shape = shapes.split(':')
            shape = tuple([int(i) for i in shape.split(',')])
            input_tensor_shape[name] = shape

        return input_tensor_shape

    @staticmethod
    def get_input_data_map_from_file(input_data_file):
        """get input data map from file"""
        return np.load(input_data_file, allow_pickle=True).item()

    @staticmethod
    def create_numpy_data_map_out(tensor_infos):
        """create numpy data dict"""
        np_data_map = {}
        for tensor_name, infos in tensor_infos.items():
            if not isinstance(infos, tuple):
                raise ValueError('input info shall contain tensor shape and tensor dtype')
            shape, dtype = infos
            np_dtype = getattr(NumpyDtype, dtype).value
            tensor_data = np.random.rand(*shape).astype(np_dtype)
            np_data_map[tensor_name] = tensor_data

        return np_data_map

    @staticmethod
    def save_output_as_benchmark_txt(save_dir,
                                     output_tensor):
        """save output tensor as benchmark type text"""
        for key, value in output_tensor.items():
            save_path = f'{save_dir}_{key}.txt'
            shape = value.shape
            shape_str = ''
            for val in shape:
                shape_str += f'{val} '
            dim = len(shape)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            modes = stat.S_IWUSR | stat.S_IRUSR
            with os.fdopen(os.open(save_path, flags, modes), 'a') as fi:
                fi.write(f'{key} {dim} {shape_str}\n')
                np.savetxt(fi, value.flatten(), newline=' ')

    @staticmethod
    def init_tf_cfg():
        """init tensorflow config"""
        cfg = TFConfig()
        return cfg

    @staticmethod
    def parse_dtype_infos(dtype_infos):
        """
        parse input dtype infos string to dict, key is input tensor,
        value is tensor dtype
        params:
        model_path: path to model file
        args: input arguments
        return: model config
        """
        infos = dtype_infos.split(';')
        ret = {}
        for info in infos:
            key, dtype = info.split(':')
            ret[key] = dtype.strip()

        return ret
