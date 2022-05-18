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
from .lib import _c_lite_wrapper


class Converter:
    """
    Converter is used to convert third-party models.

    Args:
        fmk_type(Enum, optional): Input model framework type. TF | TFLITE | CAFFE | MINDIR | ONNX.
        model_file (str, optional): Input model file.
                                    TF: *.pb | TFLITE: *.tflite | CAFFE: *.prototxt | MINDIR: *.mindir | ONNX: *.onnx.
        output_file (list, optional): Output model file path. Will add .ms automatically.
        weight_file (str, optional): Input model weight file. Needed when fmk is CAFFE. CAFFE: *.caffemodel,
        config_file (str, optional): Configuration for post-training, offline split op to parallel,
                                     disable op fusion ability and set plugin so path.
        weight_fp16 (list, optional): Serialize const tensor in Float16 data type,
                                      only effective for const tensor in Float32 data type. on | off.
        input_shape (list, optional): Set the dimension of the model input,
                                      the order of input dimensions is consistent with the original model.
                                      For some models, the model structure can be further optimized,
                                      but the transformed model may lose the characteristics of dynamic shape.
                                      e.g. inTensor1:1,32,32,32;inTensor2:1,1,32,32,4.
        input_format (str, optional): Assign the input format of exported model. Only Valid for 4-dimensional input.
                                      NHWC | NCHW", "NHWC.
        input_data_type (str, optional): Data type of input tensors, default is same with the type defined in model.
                                         FLOAT | INT8 | UINT8 | DEFAULT.
        output_data_type (str, optional): Data type of output and output tensors,
                                          default is same with the type defined in model. FLOAT | INT8.
        export_mindir (str, optional): Whether to export MindIR pb.
        decrypt_key (str, optional): The key used to decrypt the file, expressed in hexadecimal characters.
                                     Only valid when fmkIn is 'MINDIR'.
        decrypt_mode (str, optional): Decryption method for the MindIR file. Only valid when dec_key is set.
                                      AES-GCM | AES-CBC, default: AES-GCM.
        enable_encryption (str, optional): Whether to export the encryption model.
        encrypt_key (str, optional): The key used to encrypt the file, expressed in hexadecimal characters.
                                     Only support AES-GCM and the key length is 16.
        pre_infer (str, optional): Whether to do pre-inference after convert.
        train_model (str, optional): whether the model is going to be trained on device. true | false. default: false.
        no_fusion(bool, optional): Avoid fusion optimization true|false, default: false.

    Examples:
        >>> import mindspore_lite as mslite
        >>> device_info = mslite.context.AscendDeviceInfo(input_format="NHWC")
    """

    def __init__(self, fmk_type, model_file, output_file, weight_file="", config_file=None, weight_fp16=None,
                 input_shape=None, input_format=None, input_data_type=None, output_data_type=None, export_mindir=None,
                 decrypt_key=None, decrypt_mode=None, enable_encryption=None, encrypt_key=None,
                 pre_infer=None, train_model=None, no_fusion=None):
        self._converter = _c_lite_wrapper.ConverterBind()

    def __str__(self):
        res = f"config_file: {self._context.get_config_file()}, " \
              f"weight_fp16: {self._context.get_weight_fp16()}, " \
              f"input_shape: {self._context.get_input_shape()}, " \
              f"input_format: {self._context.get_input_format()}, " \
              f"input_data_type: {self._context.get_input_data_type()}, " \
              f"output_data_type: {self._context.get_output_data_type()}, " \
              f"export_mindir: {self._context.get_export_mindir()}, " \
              f"decrypt_key: {self._context.get_decrypt_key()}, " \
              f"decrypt_mode: {self._context.get_decrypt_mode()}, " \
              f"enable_encryption: {self._context.get_enable_encryption()}, " \
              f"encrypt_key: {self._context.get_encrypt_key()}, " \
              f"pre_infer: {self._context.get_pre_infer()}, " \
              f"train_model: {self._context.get_train_model()}, " \
              f"no_fusion: {self._context.get_no_fusion()}."
        return res

    def converter(self):
        pass
