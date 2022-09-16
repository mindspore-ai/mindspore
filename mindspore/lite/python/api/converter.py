# Copyright 2022 Huawei Technologies Co., Ltd
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
Converter API.
"""

import os
from enum import Enum

from ._checkparam import check_isinstance, check_input_shape, check_config_info
from .lib import _c_lite_wrapper
from .tensor import DataType, Format, data_type_py_cxx_map, data_type_cxx_py_map, format_py_cxx_map, format_cxx_py_map
from .model import ModelType, model_type_py_cxx_map, model_type_cxx_py_map

__all__ = ['FmkType', 'Converter']


class FmkType(Enum):
    """
    The FmkType is used to define Input model framework type.
    """
    TF = 0
    CAFFE = 1
    ONNX = 2
    MINDIR = 3
    TFLITE = 4
    PYTORCH = 5


class Converter:
    r"""
    Converter is used to convert third-party models.

    Note:
        If the default value of the parameter is none, it means the parameter is not set.

    Args:
        fmk_type (FmkType): Input model framework type. Options: FmkType.TF | FmkType.CAFFE | FmkType.ONNX |
            FmkType.MINDIR | FmkType.TFLITE | FmkType.PYTORCH.
        model_file (str): Path of the input model. e.g. "/home/user/model.prototxt". Options:
            TF: "\*.pb" | CAFFE: "\*.prototxt" | ONNX: "\*.onnx" | MINDIR: "\*.mindir" | TFLITE: "\*.tflite" |
            PYTORCH: "\*.pt or \*.pth".
        output_file (str): Path of the output model. The suffix .ms can be automatically generated.
            e.g. "/home/user/model.prototxt", it will generate the model named model.prototxt.ms in /home/user/
        weight_file (str, optional): Input model weight file. Required only when fmk_type is FmkType.CAFFE.
            e.g. "/home/user/model.caffemodel". Default: "".
        config_file (str, optional): Configuration for post-training, offline split op to parallel,
            disable op fusion ability and set plugin so path. e.g. "/home/user/model.cfg". Default: "".
        weight_fp16 (bool, optional): Serialize const tensor in Float16 data type,
            only effective for const tensor in Float32 data type. Default: False.
        input_shape (dict{str, list[int]}, optional): Set the dimension of the model input,
            the order of input dimensions is consistent with the original model. For some models, the model structure
            can be further optimized, but the transformed model may lose the characteristics of dynamic shape.
            e.g. {"inTensor1": [1, 32, 32, 32], "inTensor2": [1, 1, 32, 32]}. Default: {}.
        input_format (Format, optional): Assign the input format of exported model. Only Valid for 4-dimensional input.
            Options: Format.NHWC | Format.NCHW. Default: Format.NHWC.
        input_data_type (DataType, optional): Data type of input tensors. The default type is same with the type
            defined in model. Default: DataType.FLOAT32.
        output_data_type (DataType, optional): Data type of output tensors.
            The default type is same with the type defined in model. Default: DataType.FLOAT32.
        export_mindir (ModelType, optional): Which model type need to be export. Default: ModelType.MINDIR_LITE.
        decrypt_key (str, optional): The key used to decrypt the file, expressed in hexadecimal characters.
            Only valid when fmk_type is FmkType.MINDIR. Default: "".
        decrypt_mode (str, optional): Decryption method for the MindIR file. Only valid when dec_key is set.
            Options: "AES-GCM" | "AES-CBC". Default: "AES-GCM".
        enable_encryption (bool, optional): Whether to export the encryption model. Default: False.
        encrypt_key (str, optional): The key used to encrypt the file, expressed in hexadecimal characters.
            Only support decrypt_mode is "AES-GCM", the key length is 16. Default: "".
        infer (bool, optional): Whether to do pre-inference after convert. Default: False.
        train_model (bool, optional): whether the model is going to be trained on device. Default: False.
        no_fusion(bool, optional): Avoid fusion optimization, fusion optimization is allowed by default. Default: False.

    Raises:
        TypeError: `fmk_type` is not a FmkType.
        TypeError: `model_file` is not a str.
        TypeError: `output_file` is not a str.
        TypeError: `weight_file` is not a str.
        TypeError: `config_file` is not a str.
        TypeError: `weight_fp16` is not a bool.
        TypeError: `input_shape` is neither a dict nor None.
        TypeError: `input_shape` is a dict, but the keys are not str.
        TypeError: `input_shape` is a dict, the keys are str, but the values are not list.
        TypeError: `input_shape` is a dict, the keys are str, the values are list, but the value's elements are not int.
        TypeError: `input_format` is not a Format.
        TypeError: `input_data_type` is not a DataType.
        TypeError: `output_data_type` is not a DataType.
        TypeError: `export_mindir` is not a ModelType.
        TypeError: `decrypt_key` is not a str.
        TypeError: `decrypt_mode` is not a str.
        TypeError: `enable_encryption` is not a bool.
        TypeError: `encrypt_key` is not a str.
        TypeError: `infer` is not a bool.
        TypeError: `train_model` is not a bool.
        TypeError: `no_fusion` is not a bool.
        ValueError: `input_format` is neither Format.NCHW nor Format.NHWC when it is a Format.
        ValueError: `decrypt_mode` is neither "AES-GCM" nor "AES-CBC" when it is a str.
        RuntimeError: `model_file` does not exist.
        RuntimeError: `weight_file` is not "", but `weight_file` does not exist.
        RuntimeError: `config_file` is not "", but `config_file` does not exist.

    Examples:
        >>> # Download the model package and extract it, model download link:
        >>> # https://download.mindspore.cn/model_zoo/official/lite/quick_start/micro/mobilenetv2.tar.gz
        >>> import mindspore_lite as mslite
        >>> converter = mslite.Converter(mslite.FmkType.kFmkTypeTflite, "./mobilenetv2/mobilenet_v2_1.0_224.tflite",
        ...                              "mobilenet_v2_1.0_224.tflite")
        >>> print(converter)
        config_file: ,
        config_info: {},
        weight_fp16: False,
        input_shape: {},
        input_format: Format.NHWC,
        input_data_type: DataType.FLOAT32,
        output_data_type: DataType.FLOAT32,
        export_mindir: ModelType.MINDIR_LITE,
        decrypt_key: ,
        decrypt_mode: AES-GCM,
        enable_encryption: False,
        encrypt_key: ,
        infer: False,
        train_model: False,
        no_fusion: False.
    """

    def __init__(self, fmk_type, model_file, output_file, weight_file="", config_file="", weight_fp16=False,
                 input_shape=None, input_format=Format.NHWC, input_data_type=DataType.FLOAT32,
                 output_data_type=DataType.FLOAT32, export_mindir=ModelType.MINDIR_LITE, decrypt_key="",
                 decrypt_mode="AES-GCM", enable_encryption=False, encrypt_key="", infer=False, train_model=False,
                 no_fusion=False):
        check_isinstance("fmk_type", fmk_type, FmkType)
        check_isinstance("model_file", model_file, str)
        check_isinstance("output_file", output_file, str)
        check_isinstance("weight_file", weight_file, str)
        check_isinstance("config_file", config_file, str)
        check_isinstance("weight_fp16", weight_fp16, bool)
        check_input_shape("input_shape", input_shape, enable_none=True)
        check_isinstance("input_format", input_format, Format)
        check_isinstance("input_data_type", input_data_type, DataType)
        check_isinstance("output_data_type", output_data_type, DataType)
        check_isinstance("export_mindir", export_mindir, ModelType)
        check_isinstance("decrypt_key", decrypt_key, str)
        check_isinstance("decrypt_mode", decrypt_mode, str)
        check_isinstance("enable_encryption", enable_encryption, bool)
        check_isinstance("encrypt_key", encrypt_key, str)
        check_isinstance("infer", infer, bool)
        check_isinstance("train_model", train_model, bool)
        check_isinstance("no_fusion", no_fusion, bool)
        if not os.path.exists(model_file):
            raise RuntimeError(f"Converter's init failed, model_file does not exist!")
        if weight_file != "":
            if not os.path.exists(weight_file):
                raise RuntimeError(f"Converter's init failed, weight_file does not exist!")
        if config_file != "":
            if not os.path.exists(config_file):
                raise RuntimeError(f"Converter's init failed, config_file does not exist!")
        if input_format not in [Format.NCHW, Format.NHWC]:
            raise ValueError(f"Converter's init failed, input_format must be NCHW or NHWC.")
        if decrypt_mode not in ["AES-GCM", "AES-CBC"]:
            raise ValueError(f"Converter's init failed, decrypt_mode must be AES-GCM or AES-CBC.")
        input_shape_ = {} if input_shape is None else input_shape

        fmk_type_py_cxx_map = {
            FmkType.TF: _c_lite_wrapper.FmkType.kFmkTypeTf,
            FmkType.CAFFE: _c_lite_wrapper.FmkType.kFmkTypeCaffe,
            FmkType.ONNX: _c_lite_wrapper.FmkType.kFmkTypeOnnx,
            FmkType.MINDIR: _c_lite_wrapper.FmkType.kFmkTypeMs,
            FmkType.TFLITE: _c_lite_wrapper.FmkType.kFmkTypeTflite,
            FmkType.PYTORCH: _c_lite_wrapper.FmkType.kFmkTypePytorch,
        }
        self._converter = _c_lite_wrapper.ConverterBind(fmk_type_py_cxx_map.get(fmk_type), model_file, output_file,
                                                        weight_file)
        if config_file != "":
            self._converter.set_config_file(config_file)
        if weight_fp16:
            self._converter.set_weight_fp16(weight_fp16)
        if input_shape is not None:
            self._converter.set_input_shape(input_shape_)
        if input_format != Format.NHWC:
            self._converter.set_input_format(format_py_cxx_map.get(input_format))
        if input_data_type != DataType.FLOAT32:
            self._converter.set_input_data_type(data_type_py_cxx_map.get(input_data_type))
        if output_data_type != DataType.FLOAT32:
            self._converter.set_output_data_type(data_type_py_cxx_map.get(output_data_type))
        if export_mindir != ModelType.MINDIR_LITE:
            self._converter.set_export_mindir(model_type_py_cxx_map.get(export_mindir))
        if decrypt_key != "":
            self._converter.set_decrypt_key(decrypt_key)
        self._converter.set_decrypt_mode(decrypt_mode)
        if enable_encryption:
            self._converter.set_enable_encryption(enable_encryption)
        if encrypt_key != "":
            self._converter.set_encrypt_key(encrypt_key)
        if infer:
            self._converter.set_infer(infer)
        if train_model:
            self._converter.set_train_model(train_model)
        if no_fusion:
            self._converter.set_no_fusion(no_fusion)

    def __str__(self):
        res = f"config_file: {self._converter.get_config_file()},\n" \
              f"config_info: {self._converter.get_config_info()},\n" \
              f"weight_fp16: {self._converter.get_weight_fp16()},\n" \
              f"input_shape: {self._converter.get_input_shape()},\n" \
              f"input_format: {format_cxx_py_map.get(self._converter.get_input_format())},\n" \
              f"input_data_type: {data_type_cxx_py_map.get(self._converter.get_input_data_type())},\n" \
              f"output_data_type: {data_type_cxx_py_map.get(self._converter.get_output_data_type())},\n" \
              f"export_mindir: {model_type_cxx_py_map.get(self._converter.get_export_mindir())},\n" \
              f"decrypt_key: {self._converter.get_decrypt_key()},\n" \
              f"decrypt_mode: {self._converter.get_decrypt_mode()},\n" \
              f"enable_encryption: {self._converter.get_enable_encryption()},\n" \
              f"encrypt_key: {self._converter.get_encrypt_key()},\n" \
              f"infer: {self._converter.get_infer()},\n" \
              f"train_model: {self._converter.get_train_model()},\n" \
              f"no_fusion: {self._converter.get_no_fusion()}."
        return res

    def set_config_info(self, section, config_info):
        """
        Set config info for converter.It is used together with get_config_info method for online converter.

        Args:
            section (str): The category of the configuration parameter.
                Set the individual parameters of the configFile together with config_info.
                e.g. for section = "common_quant_param", config_info = {"quant_type":"WEIGHT_QUANT"}. Default: None.
                For the configuration parameters related to post training quantization, please refer to
                `quantization <https://www.mindspore.cn/lite/docs/en/master/use/post_training_quantization.html>`_.
                For the configuration parameters related to extension, please refer to
                `extension  <https://www.mindspore.cn/lite/docs/en/master/use/nnie.html#extension-configuration>`_.

                - "common_quant_param": Common quantization parameter. One of configuration parameters for quantization.
                - "mixed_bit_weight_quant_param": Mixed bit weight quantization parameter.
                  One of configuration parameters for quantization.
                - "full_quant_param": Full quantization parameter. One of configuration parameters for quantization.
                - "data_preprocess_param": Data preprocess parameter. One of configuration parameters for quantization.
                - "registry": Extension configuration parameter. One of configuration parameters for extension.

            config_info (dict{str, str}): List of configuration parameters.
                Set the individual parameters of the configFile together with section.
                e.g. for section = "common_quant_param", config_info = {"quant_type":"WEIGHT_QUANT"}. Default: None.
                For the configuration parameters related to post training quantization, please refer to
                `quantization <https://www.mindspore.cn/lite/docs/en/master/use/post_training_quantization.html>`_.
                For the configuration parameters related to extension, please refer to
                `extension  <https://www.mindspore.cn/lite/docs/en/master/use/nnie.html#extension-configuration>`_.

        Raises:
            TypeError: `section` is not a str.
            TypeError: `config_info` is not a dict.
            TypeError: `config_info` is a dict, but the keys are not str.
            TypeError: `config_info` is a dict, the keys are str, but the values are not str.

        Examples:
            >>> # Download the model package and extract it, model download link:
            >>> # https://download.mindspore.cn/model_zoo/official/lite/quick_start/micro/mobilenetv2.tar.gz
            >>> import mindspore_lite as mslite
            >>> converter = mslite.Converter(mslite.FmkType.kFmkTypeTflite, "./mobilenetv2/mobilenet_v2_1.0_224.tflite",
            ...                              "mobilenet_v2_1.0_224.tflite")
            >>> section = "common_quant_param"
            >>> config_info = {"quant_type":"WEIGHT_QUANT"}
            >>> converter.set_config_info(section, config_info)
        """
        check_isinstance("section", section, str)
        check_config_info("config_info", config_info, enable_none=True)
        if section != "" and config_info is not None:
            self._converter.set_config_info(section, config_info)

    def get_config_info(self):
        """
        Get config info of converter.It is used together with set_config_info method for online converter.
        Please use set_config_info method before get_config_info.

        Returns:
            dict{str, dict{str, str}}, the config info which has been set in converter.

        Examples:
            >>> # Download the model package and extract it, model download link:
            >>> # https://download.mindspore.cn/model_zoo/official/lite/quick_start/micro/mobilenetv2.tar.gz
            >>> import mindspore_lite as mslite
            >>> converter = mslite.Converter(mslite.FmkType.kFmkTypeTflite, "./mobilenetv2/mobilenet_v2_1.0_224.tflite",
            ...                              "mobilenet_v2_1.0_224.tflite")
            >>> section = "common_quant_param"
            >>> config_info_in = {"quant_type":"WEIGHT_QUANT"}
            >>> converter.set_config_info(section, config_info_in)
            >>> config_info_out = converter.get_config_info()
            >>> print(config_info_out)
            {'common_quant_param': {'quant_type': 'WEIGHT_QUANT'}}
        """
        return self._converter.get_config_info()

    def converter(self):
        """
        Perform conversion, and convert the third-party model to the mindspire model.

        Raises:
            RuntimeError: converter model failed.

        Examples:
            >>> # Download the model package and extract it, model download link:
            >>> # https://download.mindspore.cn/model_zoo/official/lite/quick_start/micro/mobilenetv2.tar.gz
            >>> import mindspore_lite as mslite
            >>> converter = mslite.Converter(mslite.FmkType.kFmkTypeTflite, "./mobilenetv2/mobilenet_v2_1.0_224.tflite",
            ...                              "mobilenet_v2_1.0_224.tflite")
            >>> converter.converter()
            CONVERT RESULT SUCCESS:0
        """
        ret = self._converter.converter()
        if not ret.IsOk():
            raise RuntimeError(f"Converter model failed! Error is {ret.ToString()}")
