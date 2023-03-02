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
    When Converter, the `FmkType` is used to define Input model framework type.

    Currently, the following model framework types are supported:

    ===========================  ============================================================================
    Definition                    Description
    ===========================  ============================================================================
    `FmkType.TF`                 TensorFlow model's framework type, and the model uses .pb as suffix.
    `FmkType.CAFFE`              Caffe model's framework type, and the model uses .prototxt as suffix.
    `FmkType.ONNX`               ONNX model's framework type, and the model uses .onnx as suffix.
    `FmkType.MINDIR`             MindSpore model's framework type, and the model uses .mindir as suffix.
    `FmkType.TFLITE`             TensorFlow Lite model's framework type, and the model uses .tflite as suffix.
    `FmkType.PYTORCH`            PyTorch model's framework type, and the model uses .pt or .pth as suffix.
    ===========================  ============================================================================

    Examples:
        >>> # Method 1: Import mindspore_lite package
        >>> import mindspore_lite as mslite
        >>> print(mslite.FmkType.TF)
        FmkType.TF
        >>> # Method 2: from mindspore_lite package import FmkType
        >>> from mindspore_lite import FmkType
        >>> print(FmkType.TF)
        FmkType.TF
    """

    TF = 0
    CAFFE = 1
    ONNX = 2
    MINDIR = 3
    TFLITE = 4
    PYTORCH = 5


class Converter:
    r"""
    Constructs a `Converter` class. The usage scenarios are: 1. Convert the third-party model into MindSpore model or
    MindSpore Lite model; 2. Convert MindSpore model into MindSpore Lite model.

    Note:
        Please construct the `Converter` class first, and then generate the model by executing the Converter.converter()
        method.

        The encryption and decryption function is only valid when it is set to `MSLITE_ENABLE_MODEL_ENCRYPTION=on` at
        the compile time, and only supports Linux x86 platforms. `decrypt_key` and `encrypt_key` are string expressed in
        hexadecimal. For example, if the key is defined as '(b)0123456789ABCDEF' , the corresponding hexadecimal
        expression is '30313233343637383939414243444546' . Linux platform users can use the' xxd 'tool to convert the
        key expressed in bytes into hexadecimal expressions. It should be noted that the encryption and decryption
        algorithm has been updated in version 1.7, resulting in the new python interface does not support the conversion
        of MindSpore Lite's encryption exported models in version 1.6 and earlier.

    Args:
        fmk_type (FmkType): Input model framework type. Options: FmkType.TF | FmkType.CAFFE |
            FmkType.ONNX | FmkType.MINDIR | FmkType.TFLITE | FmkType.PYTORCH. For details, see
            `FmkType <https://mindspore.cn/lite/api/en/master/mindspore_lite/mindspore_lite.FmkType.html>`_ .
        model_file (str): Set the path of the input model when converter. For example, "/home/user/model.prototxt".
            Options:TF: "model.pb" | CAFFE: "model.prototxt" | ONNX: "model.onnx" | MINDIR: "model.mindir" |
            TFLITE: "model.tflite" | PYTORCH: "model.pt or model.pth".
        output_file (str): Set the path of the output model. The suffix .ms or .mindir can be automatically generated.
            If set `save_type` to ModelType.MINDIR, then MindSpore's model will be generated, which uses .mindir as
            suffix. If set `save_type` to ModelType.MINDIR_LITE, then MindSpore Lite's model will be generated,
            which uses .ms as suffix. For example, the input model is "/home/user/model.prototxt", it will generate the
            model named model.prototxt.ms in /home/user/.
        weight_file (str, optional): Set the path of input model weight file. Required only when fmk_type is
            FmkType.CAFFE. The Caffe model is generally divided into two files: 'model.prototxt' is model structure,
            corresponding to `model_file` parameter; 'model.Caffemodel' is model weight value file, corresponding to
            `weight_file` parameter. For example, "/home/user/model.caffemodel". Default: "".
        config_file (str, optional): Set the path of the configuration file of Converter can be used to post-training,
            offline split op to parallel, disable op fusion ability and set plugin so path. `config_file` uses the
            `key = value` method to define the related parameters.
            For the configuration parameters related to post training quantization, please refer to
            `quantization <https://www.mindspore.cn/lite/docs/en/master/use/post_training_quantization.html>`_ .
            For the configuration parameters related to extension, please refer to
            `extension  <https://www.mindspore.cn/lite/docs/en/master/use/nnie.html#extension-configuration>`_ .
            For example, "/home/user/model.cfg". Default: "".
        weight_fp16 (bool, optional): If it is True, the const Tensor of the Float32 in the model will be saved as the
            Float16 data type during Converter, and the generated model size will be compressed. Then, according to
            `DeviceInfo`'s `enable_fp16` parameter determines the inputs' data type to perform inference. The priority
            of `weight_fp16` is very low. For example, if quantization is enabled, for the weight of the quantized,
            `weight_fp16` will not take effect again. `weight_fp16` only effective for the const Tensor in Float32 data
            type. Default: False.
        input_shape (dict{str, list[int]}, optional): Set the dimension of the model input. The order of input
            dimensions is consistent with the original model. In the following scenarios, users may need to set the
            parameter. For example, {"inTensor1": [1, 32, 32, 32], "inTensor2": [1, 1, 32, 32]}. Default: None, None is
            equivalent to {}.

            - Usage 1:The input of the model to be converted is dynamic shape, but prepare to use fixed shape for
              inference, then set the parameter to fixed shape. After setting, when inferring on the converted
              model, the default input shape is the same as the parameter setting, no need to resize.
            - Usage 2: No matter whether the original input of the model to be converted is dynamic shape or not,
              but prepare to use fixed shape for inference, and the performance of the model is
              expected to be optimized as much as possible, then set the parameter to fixed shape. After
              setting, the model structure will be further optimized, but the converted model may lose the
              characteristics of dynamic shape(some operators strongly related to shape will be merged).
            - Usage 3: When using the converter function to generate code for Micro inference execution, it is
              recommended to set the parameter to reduce the probability of errors during deployment.
              When the model contains a Shape ops or the input of the model to be converted is a dynamic
              shape, you must set the parameter to fixed shape to support the relevant shape optimization and
              code generation.

        input_format (Format, optional): Set the input format of exported model. Only Valid for 4-dimensional input. The
            following 2 input formats are supported: Format.NCHW | Format.NHWC. Default: Format.NHWC. For details, see
            `Format <https://mindspore.cn/lite/api/en/master/mindspore_lite/mindspore_lite.Format.html>`_ .

            - Format.NCHW: Store tensor data in the order of batch N, channel C, height H and width W.
            - Format.NHWC: Store tensor data in the order of batch N, height H, width W and channel C.

        input_data_type (DataType, optional): Set the data type of the quantization model input Tensor. It is only valid
            when the quantization parameters ( `scale` and `zero point` ) of the model input tensor are available.
            The following 4 DataTypes are supported: DataType.FLOAT32 | DataType.INT8 | DataType.UINT8 |
            DataType.UNKNOWN. Default: DataType.FLOAT32. For details, see
            `DataType <https://mindspore.cn/lite/api/en/master/mindspore_lite/mindspore_lite.DataType.html>`_ .

            - DataType.FLOAT32: 32-bit floating-point number.
            - DataType.INT8:    8-bit integer.
            - DataType.UINT8:   unsigned 8-bit integer.
            - DataType.UNKNOWN: Set the Same DataType as the model input Tensor.

        output_data_type (DataType, optional): Set the data type of the quantization model output Tensor. It is only
            valid when the quantization parameters ( `scale` and `zero point` ) of the model output tensor are
            available. The following 4 DataTypes are supported: DataType.FLOAT32 | DataType.INT8 | DataType.UINT8 |
            DataType.UNKNOWN. Default: DataType.FLOAT32. For details, see
            `DataType <https://mindspore.cn/lite/api/en/master/mindspore_lite/mindspore_lite.DataType.html>`_ .

            - DataType.FLOAT32: 32-bit floating-point number.
            - DataType.INT8:    8-bit integer.
            - DataType.UINT8:   unsigned 8-bit integer.
            - DataType.UNKNOWN: Set the Same DataType as the model output Tensor.

        save_type (ModelType, optional): Set the model type needs to be export. Options: ModelType.MINDIR |
            ModelType.MINDIR_LITE. Default: None. For details, see
            `ModelType <https://mindspore.cn/lite/api/en/master/mindspore_lite/mindspore_lite.ModelType.html>`_ .
        decrypt_key (str, optional): Set the key used to decrypt the encrypted MindIR file, expressed in hexadecimal
            characters. Only valid when fmk_type is FmkType.MINDIR. Default: "".
        decrypt_mode (str, optional): Set decryption mode for the encrypted MindIR file. Only valid when dec_key is
            set. Options: "AES-GCM" | "AES-CBC". Default: "AES-GCM".
        enable_encryption (bool, optional): Whether to encrypt the model when exporting. Export encryption can protect
            the integrity of the model, but it will increase the initialization time at runtime. Default: False.
        encrypt_key (str, optional): Set the key used to encrypt the model when exporting, expressed in hexadecimal
            characters. Only support when `decrypt_mode` is "AES-GCM", the key length is 16. Default: "".
        infer (bool, optional): Whether to do pre-inference after Converter. Default: False.
        train_model (bool, optional):   Whether the model is going to be trained on device. Default: False.
        optimize(str, optional): Whether avoid fusion optimization. Default: general,
            which means fusion optimization is allowed.
        device (str, optional): Set target device when converter model. Only valid for Ascend. The use case is when on
            the Ascend device, if you need to the converted model to have the ability to use Ascend backend to perform
            inference, you can set the parameter. If it is not set, the converted model will use CPU backend to perform
            inference by default. Options: "Ascend". Default: "".
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
        TypeError: `save_type` is not a ModelType.
        TypeError: `decrypt_key` is not a str.
        TypeError: `decrypt_mode` is not a str.
        TypeError: `enable_encryption` is not a bool.
        TypeError: `encrypt_key` is not a str.
        TypeError: `infer` is not a bool.
        TypeError: `train_model` is not a bool.
        TypeError: `general` is not a str.
        TypeError: `device` is not a str.
        ValueError: `input_format` is neither Format.NCHW nor Format.NHWC when it is a Format.
        ValueError: `decrypt_mode` is neither "AES-GCM" nor "AES-CBC" when it is a str.
        ValueError: `device` is not "Ascend" when it is a str.
        RuntimeError: `model_file` does not exist.
        RuntimeError: `weight_file` is not "", but `weight_file` does not exist.
        RuntimeError: `config_file` is not "", but `config_file` does not exist.

    Examples:
        >>> import mindspore_lite as mslite
        >>> converter = mslite.Converter(mslite.FmkType.TFLITE, "./mobilenetv2/mobilenet_v2_1.0_224.tflite",
        ...                              "mobilenet_v2_1.0_224.tflite")
        >>> # The ms model may be generated only after converter.converter() is executed after the class is constructed.
        >>> print(converter)
        config_file: ,
        config_info: {},
        weight_fp16: False,
        input_shape: {},
        input_format: Format.NHWC,
        input_data_type: DataType.FLOAT32,
        output_data_type: DataType.FLOAT32,
        save_type: None,
        decrypt_key: ,
        decrypt_mode: AES-GCM,
        enable_encryption: False,
        encrypt_key: ,
        infer: False,
        train_model: False,
        optimize: general,
        device: .
    """

    def __init__(self, fmk_type, model_file, output_file, weight_file="", config_file="", weight_fp16=False,
                 input_shape=None, input_format=Format.NHWC, input_data_type=DataType.FLOAT32,
                 output_data_type=DataType.FLOAT32, save_type=None, decrypt_key="",
                 decrypt_mode="AES-GCM", enable_encryption=False, encrypt_key="", infer=False, train_model=False,
                 optimize="general", device=""):
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
        check_isinstance("decrypt_key", decrypt_key, str)
        check_isinstance("decrypt_mode", decrypt_mode, str)
        check_isinstance("enable_encryption", enable_encryption, bool)
        check_isinstance("encrypt_key", encrypt_key, str)
        check_isinstance("infer", infer, bool)
        check_isinstance("train_model", train_model, bool)
        check_isinstance("device", device, str)
        check_isinstance("optimize", optimize, str)
        if save_type is not None:
            check_isinstance("save_type", save_type, ModelType)

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
        if device != "":
            if device not in ["Ascend"]:
                raise ValueError(f"Converter's init failed, device must be Ascend.")
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
        if save_type is not None:
            self._converter.set_save_type(model_type_py_cxx_map.get(save_type))
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

        no_fusion = True
        if optimize == "none":
            no_fusion = True
        elif optimize == "general":
            no_fusion = False
        elif optimize == "ascend_oriented":
            no_fusion = False
        else:
            raise ValueError(f"Converter's init failed, optimize must be 'general', 'none' or 'ascend_oriented'.")
        self._converter.set_no_fusion(no_fusion)
        if device != "":
            self._converter.set_device(device)

    def __str__(self):
        res = f"config_file: {self._converter.get_config_file()},\n" \
              f"config_info: {self._converter.get_config_info()},\n" \
              f"weight_fp16: {self._converter.get_weight_fp16()},\n" \
              f"input_shape: {self._converter.get_input_shape()},\n" \
              f"input_format: {format_cxx_py_map.get(self._converter.get_input_format())},\n" \
              f"input_data_type: {data_type_cxx_py_map.get(self._converter.get_input_data_type())},\n" \
              f"output_data_type: {data_type_cxx_py_map.get(self._converter.get_output_data_type())},\n" \
              f"save_type: {model_type_cxx_py_map.get(self._converter.get_save_type())},\n" \
              f"decrypt_key: {self._converter.get_decrypt_key()},\n" \
              f"decrypt_mode: {self._converter.get_decrypt_mode()},\n" \
              f"enable_encryption: {self._converter.get_enable_encryption()},\n" \
              f"encrypt_key: {self._converter.get_encrypt_key()},\n" \
              f"infer: {self._converter.get_infer()},\n" \
              f"train_model: {self._converter.get_train_model()},\n" \
              f"optimize: {self._converter.get_no_fusion()},\n" \
              f"device: {self._converter.get_device()}."
        return res

    def set_config_info(self, section="", config_info=None):
        """
        Set config info for Converter.It is used together with `get_config_info` method for online converter.

        Args:
            section (str, optional): The category of the configuration parameter.
                Set the individual parameters of the configfile together with `config_info` .
                For example, for `section` = "common_quant_param", `config_info` = {"quant_type":"WEIGHT_QUANT"}.
                Default: "".

                For the configuration parameters related to post training quantization, please refer to
                `quantization <https://www.mindspore.cn/lite/docs/en/master/use/post_training_quantization.html>`_ .

                For the configuration parameters related to extension, please refer to
                `extension  <https://www.mindspore.cn/lite/docs/en/master/use/nnie.html#extension-configuration>`_ .

                - "common_quant_param": Common quantization parameter.
                - "mixed_bit_weight_quant_param": Mixed bit weight quantization parameter.
                - "full_quant_param": Full quantization parameter.
                - "data_preprocess_param": Data preprocess quantization parameter.
                - "registry": Extension configuration parameter.

            config_info (dict{str, str}, optional): List of configuration parameters.
                Set the individual parameters of the configfile together with `section` .
                For example, for `section` = "common_quant_param", `config_info` = {"quant_type":"WEIGHT_QUANT"}.
                Default: None, None is equivalent to {}.

                For the configuration parameters related to post training quantization, please refer to
                `quantization <https://www.mindspore.cn/lite/docs/en/master/use/post_training_quantization.html>`_ .

                For the configuration parameters related to extension, please refer to
                `extension  <https://www.mindspore.cn/lite/docs/en/master/use/nnie.html#extension-configuration>`_ .

        Raises:
            TypeError: `section` is not a str.
            TypeError: `config_info` is not a dict .
            TypeError: `config_info` is a dict, but the keys are not str.
            TypeError: `config_info` is a dict, the keys are str, but the values are not str.

        Examples:
            >>> import mindspore_lite as mslite
            >>> converter = mslite.Converter(mslite.FmkType.TFLITE, "./mobilenetv2/mobilenet_v2_1.0_224.tflite",
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
        Get config info of converter.It is used together with `set_config_info` method for online converter.
        Please use `set_config_info` method before `get_config_info` .

        Returns:
            dict{str, dict{str, str}}, the config info which has been set in converter.

        Examples:
            >>> import mindspore_lite as mslite
            >>> converter = mslite.Converter(mslite.FmkType.TFLITE, "./mobilenetv2/mobilenet_v2_1.0_224.tflite",
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
            >>> import mindspore_lite as mslite
            >>> converter = mslite.Converter(mslite.FmkType.TFLITE, "./mobilenetv2/mobilenet_v2_1.0_224.tflite",
            ...                              "mobilenet_v2_1.0_224.tflite")
            >>> converter.converter()
            CONVERT RESULT SUCCESS:0
            >>> # mobilenet_v2_1.0_224.tflite.ms model will be generated.
        """
        ret = self._converter.converter()
        if not ret.IsOk():
            raise RuntimeError(f"Converter model failed! Error is {ret.ToString()}")
