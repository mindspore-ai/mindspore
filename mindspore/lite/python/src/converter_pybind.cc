/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "include/converter.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"

namespace mindspore::lite {
namespace py = pybind11;

void ConverterPyBind(const py::module &m) {
  py::enum_<converter::FmkType>(m, "FmkType")
    .value("kFmkTypeTf", converter::FmkType::kFmkTypeTf)
    .value("kFmkTypeCaffe", converter::FmkType::kFmkTypeCaffe)
    .value("kFmkTypeOnnx", converter::FmkType::kFmkTypeOnnx)
    .value("kFmkTypeMs", converter::FmkType::kFmkTypeMs)
    .value("kFmkTypeTflite", converter::FmkType::kFmkTypeTflite)
    .value("kFmkTypePytorch", converter::FmkType::kFmkTypePytorch);

  py::class_<Converter, std::shared_ptr<Converter>>(m, "ConverterBind")
    .def(py::init<>())
    .def("set_config_file", py::overload_cast<const std::string &>(&Converter::SetConfigFile))
    .def("get_config_file", &Converter::GetConfigFile)
    .def("set_config_info",
         py::overload_cast<const std::string &, const std::map<std::string, std::string> &>(&Converter::SetConfigInfo))
    .def("get_config_info", &Converter::GetConfigInfo)
    .def("set_weight_fp16", &Converter::SetWeightFp16)
    .def("get_weight_fp16", &Converter::GetWeightFp16)
    .def("set_input_shape",
         py::overload_cast<const std::map<std::string, std::vector<int64_t>> &>(&Converter::SetInputShape))
    .def("get_input_shape", &Converter::GetInputShape)
    .def("set_input_format", &Converter::SetInputFormat)
    .def("get_input_format", &Converter::GetInputFormat)
    .def("set_input_data_type", &Converter::SetInputDataType)
    .def("get_input_data_type", &Converter::GetInputDataType)
    .def("set_output_data_type", &Converter::SetOutputDataType)
    .def("get_output_data_type", &Converter::GetOutputDataType)
    .def("set_save_type", &Converter::SetSaveType)
    .def("get_save_type", &Converter::GetSaveType)
    .def("set_decrypt_key", py::overload_cast<const std::string &>(&Converter::SetDecryptKey))
    .def("get_decrypt_key", &Converter::GetDecryptKey)
    .def("set_decrypt_mode", py::overload_cast<const std::string &>(&Converter::SetDecryptMode))
    .def("get_decrypt_mode", &Converter::GetDecryptMode)
    .def("set_enable_encryption", &Converter::SetEnableEncryption)
    .def("get_enable_encryption", &Converter::GetEnableEncryption)
    .def("set_encrypt_key", py::overload_cast<const std::string &>(&Converter::SetEncryptKey))
    .def("get_encrypt_key", &Converter::GetEncryptKey)
    .def("set_infer", &Converter::SetInfer)
    .def("get_infer", &Converter::GetInfer)
#if !defined(ENABLE_CLOUD_FUSION_INFERENCE) && !defined(ENABLE_CLOUD_INFERENCE)
    .def("set_train_model", &Converter::SetTrainModel)
    .def("get_train_model", &Converter::GetTrainModel)
#endif
    .def("set_no_fusion", &Converter::SetNoFusion)
    .def("get_no_fusion", &Converter::GetNoFusion)
    .def("set_device", py::overload_cast<const std::string &>(&Converter::SetDevice))
    .def("get_device", &Converter::GetDevice)
    .def("converter",
         py::overload_cast<converter::FmkType, const std::string &, const std::string &, const std::string &>(
           &Converter::Convert));
}
}  // namespace mindspore::lite
