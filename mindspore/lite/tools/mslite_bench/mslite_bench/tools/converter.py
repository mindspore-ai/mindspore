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
""" converter for mslite """

import copy
from enum import Enum
import os
from dataclasses import  dataclass

import mindspore_lite as mslite

from mslite_bench.utils import InferLogger
from mslite_bench.common.model_info_enum import DeviceType
from mslite_bench.common.task_common_func import CommonFunc
from mslite_bench.tools.cross_framework_accuracy import CrossFrameworkAccSummary


class InputModelType(Enum):
    """ enum model type class for model to be convterted"""
    PB = mslite.FmkType.TF
    CAFFE = mslite.FmkType.CAFFE
    ONNX = mslite.FmkType.ONNX
    MINDIR = mslite.FmkType.MINDIR
    TFLITE = mslite.FmkType.TFLITE
    PTH = mslite.FmkType.PYTORCH


class MsliteModelType(Enum):
    """ enum model type in mslite"""
    MINDIR = mslite.ModelType.MINDIR
    MINDIR_LITE = mslite.ModelType.MINDIR_LITE


class MsliteDataType(Enum):
    """ enum data type for quant """
    FLOAT32 = mslite.DataType.FLOAT32
    INT8 = mslite.DataType.INT8
    UINT8 = mslite.DataType.UINT8
    UNKNOWN = mslite.DataType.UNKNOWN


class MsliteTensorFormat(Enum):
    """ enum input data format"""
    NCHW = mslite.Format.NCHW
    NHWC = mslite.Format.NHWC


@dataclass
class ConverterParams:
    """data class for converter input params"""
    input_shape: str = None
    input_data_type: str = "FLOAT32"
    output_data_type: str = "FLOAT32"
    input_format: str = "NCHW"
    weight_fp16: bool = False
    save_type: str = "MINDIR"
    decrypt_key: str = None
    decrypt_mode: str = None
    enable_encryption: bool = False
    encrypt_key: str = None
    infer: bool = False
    optimize: str = None
    device: str = DeviceType.CPU.value
    converter_output_file: str = None
    model_file: str = None
    params_file: str = None


class MsliteConverter:
    """model converter for mindspore lite"""
    @classmethod
    def convert(cls,
                args,
                logger=None,
                is_delete_ms_model=False):
        """convert third party model to mslite model"""
        if logger is None:
            logger = InferLogger(args.log_path).logger

        logger.debug('Start to convert model')
        cls.model_convert(args, logger)

        if not args.converter_is_analysis:
            return

        if cls.enum_value(args.converter_save_type, MsliteModelType) == MsliteModelType.MINDIR.value:
            output_file_name = f'{args.converter_output_file}.mindir'
        else:
            output_file_name = f'{args.converter_output_file}.ms'

        if os.path.exists(output_file_name):
            args_copy = copy.deepcopy(args)
            args_copy.cmp_model_file = args.model_file
            args_copy.model_file = output_file_name
            logger.info('Start accuracy compare procedure')
            CrossFrameworkAccSummary.accuracy_compare_func(args_copy, logger)

            if is_delete_ms_model:
                os.remove(output_file_name)

    @classmethod
    def model_convert(cls, args, logger=None):
        """model convert for mslite"""
        if logger is None:
            logger = InferLogger(args.log_path).logger
        converter = cls._init_converter(args)
        try:
            extension = os.path.splitext(args.model_file)[1][1:].upper()
            model_type = cls.enum_value(extension, InputModelType)
        except NotImplementedError as e:
            logger.error('Input model type error: %s', e)
            return

        converter.convert(model_type,
                          args.model_file,
                          args.converter_output_file,
                          weight_file=args.params_file,
                          config_file=args.converter_config_file)


    @classmethod
    def _init_converter(cls, args):
        """init mslite model converter with args"""
        converter = mslite.Converter()

        converter.input_shape = CommonFunc.get_tensor_shapes(args.converter_input_shape)
        converter.input_data_type = cls.enum_value(args.quant_input_data_type.upper(),
                                                   MsliteDataType)
        converter.output_data_type = cls.enum_value(args.quant_output_data_type.upper(),
                                                    MsliteDataType)
        converter.input_format = cls.enum_value(args.converter_input_format,
                                                MsliteTensorFormat)
        converter.weight_fp16 = args.converter_weight_fp16
        converter.save_type = cls.enum_value(args.converter_save_type, MsliteModelType)
        converter.decrypt_key = args.converter_decrypt_key
        converter.decrypt_mode = args.converter_decrypt_mode
        converter.enable_encryption = args.converter_enable_encryption
        converter.encrypt_key = args.converter_encrypt_key
        converter.infer = False
        converter.optimize = args.converter_optimize
        if args.device.lower() == DeviceType.ASCEND.value:
            converter.device = "Ascend"

        return converter

    @staticmethod
    def enum_value(enum_key, enum_class):
        """get value from key in enum class"""
        if enum_key in enum_class.__members__:
            return enum_class[enum_key].value
        raise NotImplementedError(f"{enum_key} is not in class {enum_class}")
