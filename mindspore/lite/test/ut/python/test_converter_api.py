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
Test lite converter python API.
Note:
    Please make sure "export MSLITE_ENABLE_CONVERTER=ON" before compiling and using the API.
"""

import mindspore_lite as mslite
import pytest


# ============================ Converter ============================
def test_converter_fmk_type_type_error():
    with pytest.raises(TypeError) as raise_info:
        converter = mslite.Converter(fmk_type="", model_file="test.tflite", output_file="test.tflite")
    assert "fmk_type must be FmkType" in str(raise_info.value)


def test_converter_model_file_type_error():
    with pytest.raises(TypeError) as raise_info:
        converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file=1, output_file="mobilenetv2.tflite")
    assert "model_file must be str" in str(raise_info.value)


def test_converter_model_file_not_exist_error():
    with pytest.raises(RuntimeError) as raise_info:
        converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="test.tflite",
                                     output_file="mobilenetv2.tflite")
    assert "model_file does not exist" in str(raise_info.value)


def test_converter_output_file_type_error():
    with pytest.raises(TypeError) as raise_info:
        converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite", output_file=1)
    assert "output_file must be str" in str(raise_info.value)


def test_converter_weight_file_type_error():
    with pytest.raises(TypeError) as raise_info:
        converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                     output_file="mobilenetv2.tflite", weight_file=1)
    assert "weight_file must be str" in str(raise_info.value)


def test_converter_weight_file_not_exist_error():
    with pytest.raises(RuntimeError) as raise_info:
        converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                     output_file="mobilenetv2.tflite", weight_file="test.caffemodel")
    assert "weight_file does not exist" in str(raise_info.value)


def test_converter_common_01():
    converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                 output_file="mobilenetv2.tflite", weight_file="")
    assert "config_file:" in str(converter)


def test_converter_config_file_type_error():
    with pytest.raises(TypeError) as raise_info:
        converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                     output_file="mobilenetv2.tflite", config_file=1)
    assert "config_file must be str" in str(raise_info.value)


def test_converter_config_file_not_exist_error():
    with pytest.raises(RuntimeError) as raise_info:
        converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                     output_file="mobilenetv2.tflite", config_file="mobilenetv2_full_quant.cfg")
    assert "config_file does not exist" in str(raise_info.value)


def test_converter_weight_fp16_type_error():
    with pytest.raises(TypeError) as raise_info:
        converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                     output_file="mobilenetv2.tflite", weight_fp16=1)
    assert "weight_fp16 must be bool" in str(raise_info.value)


def test_converter_common_weight_fp16_01():
    converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                 output_file="mobilenetv2.tflite", weight_fp16=True)
    assert "weight_fp16: True" in str(converter)


def test_converter_input_shape_type_error():
    with pytest.raises(TypeError) as raise_info:
        converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                     output_file="mobilenetv2.tflite", input_shape="{'input': [1, 112, 112, 3]}")
    assert "input_shape must be dict" in str(raise_info.value)


def test_converter_input_shape_key_type_error():
    with pytest.raises(TypeError) as raise_info:
        converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                     output_file="mobilenetv2.tflite", input_shape={1: '[1, 112, 112, 3]'})
    assert "input_shape key must be str" in str(raise_info.value)


def test_converter_input_shape_value_type_error():
    with pytest.raises(TypeError) as raise_info:
        converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                     output_file="mobilenetv2.tflite", input_shape={'input': '[1, 112, 112, 3]'})
    assert "input_shape value must be list" in str(raise_info.value)


def test_converter_input_shape_value_element_type_error():
    with pytest.raises(TypeError) as raise_info:
        converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                     output_file="mobilenetv2.tflite", input_shape={'input': [1, '112', 112, 3]})
    assert "input_shape value's element must be int" in str(raise_info.value)


def test_converter_input_shape_01():
    converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                 output_file="mobilenetv2.tflite", input_shape={})
    assert "input_shape: {}" in str(converter)


def test_converter_input_shape_02():
    converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                 output_file="mobilenetv2.tflite", input_shape={'input': []})
    assert "input_shape: {'input': []}" in str(converter)


def test_converter_input_shape_03():
    converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                 output_file="mobilenetv2.tflite",
                                 input_shape={'input1': [1, 2, 3, 4], 'input2': [4, 3, 2, 1]})
    assert "input_shape: {'input1': [1, 2, 3, 4], 'input2': [4, 3, 2, 1]}" in str(converter)


def test_converter_input_format_type_error():
    with pytest.raises(TypeError) as raise_info:
        converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                     output_file="mobilenetv2.tflite", input_format=1)
    assert "input_format must be Format" in str(raise_info.value)


def test_converter_input_format_01():
    converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                 output_file="mobilenetv2.tflite", input_format=mslite.Format.NCHW)
    assert "input_format: Format.NCHW" in str(converter)


def test_converter_input_data_type_type_error():
    with pytest.raises(TypeError) as raise_info:
        converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                     output_file="mobilenetv2.tflite", input_data_type=1)
    assert "input_data_type must be DataType" in str(raise_info.value)


def test_converter_input_data_type_01():
    converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                 output_file="mobilenetv2.tflite", input_data_type=mslite.DataType.FLOAT16)
    assert "input_data_type: DataType.FLOAT16" in str(converter)


def test_converter_output_data_type_type_error():
    with pytest.raises(TypeError) as raise_info:
        converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                     output_file="mobilenetv2.tflite", output_data_type=1)
    assert "output_data_type must be DataType" in str(raise_info.value)


def test_converter_output_data_type_01():
    converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                 output_file="mobilenetv2.tflite", output_data_type=mslite.DataType.FLOAT16)
    assert "output_data_type: DataType.FLOAT16" in str(converter)


def test_converter_save_type_type_error():
    with pytest.raises(TypeError) as raise_info:
        converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                     output_file="mobilenetv2.tflite", save_type=1)
    assert "save_type must be ModelType" in str(raise_info.value)


def test_converter_save_type_01():
    converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                 output_file="mobilenetv2.tflite", save_type=mslite.ModelType.MINDIR_LITE)
    assert "save_type: ModelType.MINDIR_LITE" in str(converter)


def test_converter_decrypt_key_type_error():
    with pytest.raises(TypeError) as raise_info:
        converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                     output_file="mobilenetv2.tflite", decrypt_key=1)
    assert "decrypt_key must be str" in str(raise_info.value)


def test_converter_decrypt_key_01():
    converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                 output_file="mobilenetv2.tflite", decrypt_key="111")
    assert "decrypt_key: 111" in str(converter)


def test_converter_decrypt_mode_type_error():
    with pytest.raises(TypeError) as raise_info:
        converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                     output_file="mobilenetv2.tflite", decrypt_mode=1)
    assert "decrypt_mode must be str" in str(raise_info.value)


def test_converter_decrypt_mode_01():
    converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                 output_file="mobilenetv2.tflite", decrypt_mode="AES-CBC")
    assert "decrypt_mode: AES-CBC" in str(converter)


def test_converter_enable_encryption_type_error():
    with pytest.raises(TypeError) as raise_info:
        converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                     output_file="mobilenetv2.tflite", enable_encryption="")
    assert "enable_encryption must be bool" in str(raise_info.value)


def test_converter_enable_encryption_01():
    converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                 output_file="mobilenetv2.tflite", enable_encryption=True)
    assert "enable_encryption: True" in str(converter)


def test_converter_encrypt_key_type_error():
    with pytest.raises(TypeError) as raise_info:
        converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                     output_file="mobilenetv2.tflite", encrypt_key=1)
    assert "encrypt_key must be str" in str(raise_info.value)


def test_converter_encrypt_key_01():
    converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                 output_file="mobilenetv2.tflite", encrypt_key="111")
    assert "encrypt_key: 111" in str(converter)


def test_converter_infer_type_error():
    with pytest.raises(TypeError) as raise_info:
        converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                     output_file="mobilenetv2.tflite", infer=1)
    assert "infer must be bool" in str(raise_info.value)


def test_converter_infer_01():
    converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                 output_file="mobilenetv2.tflite", infer=True)
    assert "infer: True" in str(converter)


def test_converter_train_model_type_error():
    with pytest.raises(TypeError) as raise_info:
        converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                     output_file="mobilenetv2.tflite", train_model=1)
    assert "train_model must be bool" in str(raise_info.value)


def test_converter_train_model_01():
    converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                 output_file="mobilenetv2.tflite", train_model=True)
    assert "train_model: True" in str(converter)


def test_converter_optimize_type_error():
    with pytest.raises(TypeError) as raise_info:
        converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                     output_file="mobilenetv2.tflite", optimize=1)
    assert "optimize must be str" in str(raise_info.value)


def test_converter_optimize_01():
    converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                 output_file="mobilenetv2.tflite", optimize="none")
    assert "optimize: True" in str(converter)


def test_converter_converter_01():
    converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                 output_file="mobilenetv2.tflite")
    converter.converter()
    assert "config_file:" in str(converter)


def test_converter_set_config_info_section_type_error():
    with pytest.raises(TypeError) as raise_info:
        converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                     output_file="mobilenetv2.tflite")
        section = 2
        config_info = {"device": "3"}
        converter.set_config_info(section, config_info)
    assert "section must be str" in str(raise_info.value)


def test_converter_set_config_info_config_info_type_error():
    with pytest.raises(TypeError) as raise_info:
        converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                     output_file="mobilenetv2.tflite")
        section = "acl_param"
        config_info = ["device_id", 3]
        converter.set_config_info(section, config_info)
    assert "config_info must be dict" in str(raise_info.value)


def test_converter_set_config_info_config_info_key_type_error():
    with pytest.raises(TypeError) as raise_info:
        converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                     output_file="mobilenetv2.tflite")
        section = "acl_param"
        config_info = {2: "3"}
        converter.set_config_info(section, config_info)
    assert "config_info key must be str" in str(raise_info.value)


def test_converter_set_config_info_config_info_value_type_error():
    with pytest.raises(TypeError) as raise_info:
        converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                     output_file="mobilenetv2.tflite")
        section = "acl_param"
        config_info = {"device_id": 3}
        converter.set_config_info(section, config_info)
    assert "config_info val must be str" in str(raise_info.value)


def test_converter_set_config_info_01():
    converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="mobilenetv2.tflite",
                                 output_file="mobilenetv2.tflite")
    section = "acl_param"
    config_info = {"device_id": "3"}
    converter.set_config_info(section, config_info)
    converter.get_config_info()
    assert "config_info: {'acl_param': {'device_id': '3'}" in str(converter)
