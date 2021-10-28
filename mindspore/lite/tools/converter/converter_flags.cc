/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "tools/converter/converter_flags.h"
#include <climits>
#include <cstdlib>
#include <string>
#include <fstream>
#include <vector>
#include <memory>
#include <algorithm>
#include "ir/dtype/type_id.h"
#include "common/file_utils.h"
#include "tools/common/string_util.h"
#include "common/log_util.h"
#include "tools/converter/converter_context.h"
#include "tools/converter/config_parser/config_file_parser.h"
#include "tools/converter/config_parser/preprocess_parser.h"
#include "tools/converter/config_parser/quant_param_parser.h"

namespace mindspore {
namespace converter {
using mindspore::lite::RET_INPUT_PARAM_INVALID;
using mindspore::lite::RET_OK;
namespace {
constexpr size_t kPluginPathMaxNum = 10;
constexpr int kQuantBitNumInt16 = 16;
constexpr int kPathLengthUpperLimit = 1024;
constexpr int kMinShapeSizeInStr = 2;
}  // namespace
Flags::Flags() {
  AddFlag(&Flags::fmkIn, "fmk", "Input model framework type. TF | TFLITE | CAFFE | MINDIR | ONNX", "");
  AddFlag(&Flags::modelFile, "modelFile",
          "Input model file. TF: *.pb | TFLITE: *.tflite | CAFFE: *.prototxt | MINDIR: *.mindir | ONNX: *.onnx", "");
  AddFlag(&Flags::outputFile, "outputFile", "Output model file path. Will add .ms automatically", "");
  AddFlag(&Flags::weightFile, "weightFile", "Input model weight file. Needed when fmk is CAFFE. CAFFE: *.caffemodel",
          "");
  AddFlag(&Flags::inputDataTypeStr, "inputDataType",
          "Data type of input tensors, default is same with the type defined in model. FLOAT | INT8 | UINT8 | DEFAULT",
          "DEFAULT");
  AddFlag(&Flags::outputDataTypeStr, "outputDataType",
          "Data type of output and output tensors, default is same with the type defined in model. FLOAT | INT8 | "
          "UINT8 | DEFAULT",
          "DEFAULT");
  AddFlag(&Flags::configFile, "configFile",
          "Configuration for post-training, offline split op to parallel,"
          "disable op fusion ability and set plugin so path",
          "");
  AddFlag(&Flags::saveFP16Str, "fp16",
          "Serialize const tensor in Float16 data type, only effective for const tensor in Float32 data type. on | off",
          "off");
  AddFlag(&Flags::trainModelIn, "trainModel",
          "whether the model is going to be trained on device. "
          "true | false",
          "false");
  AddFlag(&Flags::dec_key, "decryptKey",
          "The key used to decrypt the file, expressed in hexadecimal characters. Only valid when fmkIn is 'MINDIR'",
          "");
  AddFlag(&Flags::dec_mode, "decryptMode",
          "Decryption method for the MindIR file. Only valid when dec_key is set."
          "AES-GCM | AES-CBC",
          "AES-GCM");
  AddFlag(&Flags::inTensorShape, "inputShape",
          "Set the dimension of the model input, the order of input dimensions is consistent with the original model. "
          "For some models, the model structure can be further optimized, but the transformed model may lose the "
          "characteristics of dynamic shape. "
          "e.g. \"inTensor1:1,32,32,32;inTensor2:1,1,32,32,4\"",
          "");
  AddFlag(&Flags::graphInputFormatStr, "inputDataFormat",
          "Assign the input format of exported model. Only Valid for 4-dimensional input. NHWC | NCHW", "NHWC");
}

int Flags::InitInputOutputDataType() {
  if (this->inputDataTypeStr == "FLOAT") {
    this->inputDataType = TypeId::kNumberTypeFloat32;
  } else if (this->inputDataTypeStr == "INT8") {
    this->inputDataType = TypeId::kNumberTypeInt8;
  } else if (this->inputDataTypeStr == "UINT8") {
    this->inputDataType = TypeId::kNumberTypeUInt8;
  } else if (this->inputDataTypeStr == "DEFAULT") {
    this->inputDataType = TypeId::kTypeUnknown;
  } else {
    std::cerr << "INPUT INVALID: inputDataType is invalid: %s, supported inputDataType: FLOAT | INT8 | UINT8 | DEFAULT",
      this->inputDataTypeStr.c_str();
    return RET_INPUT_PARAM_INVALID;
  }

  if (this->outputDataTypeStr == "FLOAT") {
    this->outputDataType = TypeId::kNumberTypeFloat32;
  } else if (this->outputDataTypeStr == "INT8") {
    this->outputDataType = TypeId::kNumberTypeInt8;
  } else if (this->outputDataTypeStr == "UINT8") {
    this->outputDataType = TypeId::kNumberTypeUInt8;
  } else if (this->outputDataTypeStr == "DEFAULT") {
    this->outputDataType = TypeId::kTypeUnknown;
  } else {
    std::cerr
      << "INPUT INVALID: outputDataType is invalid: %s, supported outputDataType: FLOAT | INT8 | UINT8 | DEFAULT",
      this->outputDataTypeStr.c_str();
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

int Flags::InitFmk() {
  if (this->fmkIn == "CAFFE") {
    this->fmk = kFmkTypeCaffe;
  } else if (this->fmkIn == "MINDIR") {
    this->fmk = kFmkTypeMs;
  } else if (this->fmkIn == "TFLITE") {
    this->fmk = kFmkTypeTflite;
  } else if (this->fmkIn == "ONNX") {
    this->fmk = kFmkTypeOnnx;
  } else if (this->fmkIn == "TF") {
    this->fmk = kFmkTypeTf;
  } else {
    std::cerr << "INPUT ILLEGAL: fmk must be TF|TFLITE|CAFFE|MINDIR|ONNX" << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }

  if (this->fmk != kFmkTypeCaffe && !weightFile.empty()) {
    std::cerr << "INPUT ILLEGAL: weightFile is not a valid flag" << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

int Flags::InitTrainModel() {
  if (this->trainModelIn == "true") {
    this->trainModel = true;
  } else if (this->trainModelIn == "false") {
    this->trainModel = false;
  } else {
    std::cerr << "INPUT ILLEGAL: trainModel must be true|false " << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }

  if (this->trainModel) {
    if (this->fmk != kFmkTypeMs) {
      std::cerr << "INPUT ILLEGAL: train model converter supporting only MINDIR format" << std::endl;
      return RET_INPUT_PARAM_INVALID;
    }
    if ((this->inputDataType != TypeId::kNumberTypeFloat32) && (this->inputDataType != TypeId::kTypeUnknown)) {
      std::cerr << "INPUT ILLEGAL: train model converter supporting only FP32 input tensors" << std::endl;
      return RET_INPUT_PARAM_INVALID;
    }
    if ((this->outputDataType != TypeId::kNumberTypeFloat32) && (this->outputDataType != TypeId::kTypeUnknown)) {
      std::cerr << "INPUT ILLEGAL: train model converter supporting only FP32 output tensors" << std::endl;
      return RET_INPUT_PARAM_INVALID;
    }
  }
  return RET_OK;
}

int Flags::InitInTensorShape() {
  if (this->inTensorShape.empty()) {
    return RET_OK;
  }
  std::string content = this->inTensorShape;
  std::vector<int64_t> shape;
  auto shape_strs = lite::StrSplit(content, std::string(";"));
  for (const auto &shape_str : shape_strs) {
    if (shape_str.empty()) {
      continue;
    }
    shape.clear();
    auto string_split = lite::StrSplit(shape_str, std::string(":"));
    CHECK_LESS_RETURN(string_split.size(), kMinShapeSizeInStr);
    auto name = string_split[0];
    if (name.empty()) {
      MS_LOG(ERROR) << "input tensor name is empty";
    }
    auto dim_strs = string_split[1];
    if (dim_strs.empty()) {
      MS_LOG(ERROR) << "input tensor dim string is empty";
    }
    auto dims = lite::StrSplit(dim_strs, std::string(","));
    if (dims.empty()) {
      MS_LOG(ERROR) << "input tensor dim is empty";
    }
    for (const auto &dim : dims) {
      auto dim_value = -1;
      try {
        dim_value = std::stoi(dim);
      } catch (const std::exception &e) {
        MS_LOG(ERROR) << "Get dim failed: " << e.what();
        return lite::RET_ERROR;
      }
      if (dim_value < 0) {
        MS_LOG(ERROR) << "Unsupported dim < 0.";
        return lite::RET_ERROR;
      } else {
        shape.push_back(dim_value);
      }
    }
    lite::ConverterContext::GetInstance()->UpdateGraphInputTensorShape(name, shape);
  }
  return RET_OK;
}

int Flags::InitGraphInputFormat() {
  if (this->graphInputFormatStr == "NHWC") {
    graphInputFormat = mindspore::NHWC;
  } else if (this->graphInputFormatStr == "NCHW") {
    graphInputFormat = mindspore::NCHW;
  } else if (!this->graphInputFormatStr.empty()) {
    MS_LOG(ERROR) << "graph input format is invalid.";
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

int Flags::InitExtendedIntegrationInfo(const lite::ConfigFileParser &config_file_parser) {
  auto extended_info = config_file_parser.GetRegistryInfoString();
  if (!extended_info.plugin_path.empty()) {
    const char *delimiter = ";";
    auto relative_path = lite::SplitStringToVector(extended_info.plugin_path, *delimiter);
    if (relative_path.size() > kPluginPathMaxNum) {
      MS_LOG(ERROR) << "extended plugin library's num is too big, which shouldn't be larger than 10.";
      return RET_INPUT_PARAM_INVALID;
    }
    for (size_t i = 0; i < relative_path.size(); i++) {
      this->pluginsPath.push_back(lite::RealPath(relative_path[i].c_str()));
    }
  }

  if (!extended_info.disable_fusion.empty()) {
    if (extended_info.disable_fusion == "on") {
      this->disableFusion = true;
    } else if (extended_info.disable_fusion == "off") {
      this->disableFusion = false;
    } else {
      std::cerr << "CONFIG SETTING ILLEGAL: disable_fusion should be on/off" << std::endl;
      return RET_INPUT_PARAM_INVALID;
    }
  }
  return RET_OK;
}

int Flags::InitConfigFile() {
  lite::ConfigFileParser config_file_parser;
  auto ret = config_file_parser.ParseConfigFile(this->configFile);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Parse config file failed.";
    return ret;
  }
  lite::PreprocessParser preprocess_parser;
  ret = preprocess_parser.ParsePreprocess(config_file_parser.GetDataPreProcessString(), &this->dataPreProcessParam);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Parse preprocess failed.";
    return ret;
  }
  lite::QuantParamParser quant_param_parser;
  ret = quant_param_parser.ParseCommonQuant(config_file_parser.GetCommonQuantString(), &this->commonQuantParam);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Parse common quant param failed.";
    return ret;
  }
  ret = quant_param_parser.ParseFullQuant(config_file_parser.GetFullQuantString(), &this->fullQuantParam);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Parse full quant param failed.";
    return ret;
  }
  ret = quant_param_parser.ParseMixedBitWeightQuant(config_file_parser.GetMixedBitWeightQuantString(),
                                                    &this->mixedBitWeightQuantParam);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Parse mixed bit weight quant param failed.";
    return ret;
  }
  ret = InitExtendedIntegrationInfo(config_file_parser);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Parse extended integration info failed.";
    return ret;
  }
  (void)CheckOfflineParallelConfig(this->configFile, &parallel_split_config_);
  return RET_OK;
}

int Flags::Init(int argc, const char **argv) {
  int ret;
  if (argc == 1) {
    std::cout << this->Usage() << std::endl;
    return lite::RET_SUCCESS_EXIT;
  }
  lite::Option<std::string> err = this->ParseFlags(argc, argv);

  if (err.IsSome()) {
    std::cerr << err.Get() << std::endl;
    std::cerr << this->Usage() << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }

  if (this->help) {
    std::cout << this->Usage() << std::endl;
    return lite::RET_SUCCESS_EXIT;
  }
  if (this->modelFile.empty()) {
    std::cerr << "INPUT MISSING: model file path is necessary" << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }
  if (this->outputFile.empty()) {
    std::cerr << "INPUT MISSING: output file path is necessary" << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }

#ifdef _WIN32
  replace(this->outputFile.begin(), this->outputFile.end(), '/', '\\');
#endif

  if (this->outputFile.rfind('/') == this->outputFile.length() - 1 ||
      this->outputFile.rfind('\\') == this->outputFile.length() - 1) {
    std::cerr << "INPUT ILLEGAL: outputFile must be a valid file path" << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }

  if (this->fmkIn.empty()) {
    std::cerr << "INPUT MISSING: fmk is necessary" << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }

  if (!this->configFile.empty()) {
    ret = InitConfigFile();
    if (ret != RET_OK) {
      std::cerr << "Init config file failed." << std::endl;
      return RET_INPUT_PARAM_INVALID;
    }
  }

  if (saveFP16Str == "on") {
    saveFP16 = true;
  } else if (saveFP16Str == "off") {
    saveFP16 = false;
  } else {
    std::cerr << "Init save_fp16 failed." << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }

  ret = InitInputOutputDataType();
  if (ret != RET_OK) {
    std::cerr << "Init input output datatype failed." << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }

  ret = InitFmk();
  if (ret != RET_OK) {
    std::cerr << "Init fmk failed." << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }

  ret = InitTrainModel();
  if (ret != RET_OK) {
    std::cerr << "Init train model failed." << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }

  ret = InitInTensorShape();
  if (ret != RET_OK) {
    std::cerr << "Init input tensor shape failed." << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }

  ret = InitGraphInputFormat();
  if (ret != RET_OK) {
    std::cerr << "Init graph input format failed." << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

bool CheckOfflineParallelConfig(const std::string &file, ParallelSplitConfig *parallel_split_config) {
  // device: [device0 device1] ---> {cpu, gpu}
  // computeRate: [x: y] x >=0 && y >=0 && x/y < 10
  MS_ASSERT(parallel_split_config != nullptr);
  std::vector<std::string> config_devices = {"cpu", "gpu", "npu"};
  auto compute_rate_result = GetStrFromConfigFile(file, kComputeRate);
  if (compute_rate_result.empty()) {
    return false;
  }
  std::string device0_result = GetStrFromConfigFile(file, kSplitDevice0);
  if (device0_result.empty()) {
    return false;
  }
  std::string device1_result = GetStrFromConfigFile(file, kSplitDevice1);
  if (device1_result.empty()) {
    return false;
  }
  bool device0_flag = false;
  bool device1_flag = false;
  for (const auto &device : config_devices) {
    if (device == device0_result) {
      device0_flag = true;
    }
    if (device == device1_result) {
      device1_flag = true;
    }
  }
  if (!device0_flag || !device1_flag) {
    return false;
  }
  const char *delimiter = ";";
  std::vector<std::string> device_rates = lite::SplitStringToVector(compute_rate_result, *delimiter);
  const char *colon = ":";
  for (const auto &device : device_rates) {
    std::vector<std::string> rate = lite::SplitStringToVector(device, *colon);
    int64_t compute_rate = 0;
    try {
      compute_rate = std::stoi(rate.back());
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "Get compute rate failed: " << e.what();
      return false;
    }
    parallel_split_config->parallel_compute_rates_.push_back(compute_rate);
  }
  if (parallel_split_config->parallel_compute_rates_.size() != 2) {
    return false;
  }
  int64_t bigger_rate = INT32_MIN;
  int64_t smaller_rate = INT32_MAX;
  for (const auto &rate : parallel_split_config->parallel_compute_rates_) {
    if (rate <= 0 || rate > INT32_MAX) {
      return false;
    }
    bigger_rate = std::max(rate, bigger_rate);
    smaller_rate = std::min(rate, smaller_rate);
  }
  parallel_split_config->parallel_devices_.push_back(device0_result);
  parallel_split_config->parallel_devices_.push_back(device1_result);
  // parall_split_type will extend by other user's attr
  parallel_split_config->parallel_split_type_ = SplitByUserRatio;
  // unsuitable rate
  return bigger_rate / smaller_rate <= kMaxSplitRatio;
}

std::string GetStrFromConfigFile(const std::string &file, const std::string &target_key) {
  std::string res;
  if (file.empty()) {
    MS_LOG(ERROR) << "file is nullptr";
    return res;
  }
  auto resolved_path = std::make_unique<char[]>(PATH_MAX);
  if (resolved_path == nullptr) {
    MS_LOG(ERROR) << "new resolved_path failed";
    return "";
  }

#ifdef _WIN32
  char *real_path = _fullpath(resolved_path.get(), file.c_str(), kPathLengthUpperLimit);
#else
  char *real_path = realpath(file.c_str(), resolved_path.get());
#endif
  if (real_path == nullptr || strlen(real_path) == 0) {
    MS_LOG(ERROR) << "file path is not valid : " << file;
    return "";
  }
  std::ifstream ifs(resolved_path.get());
  if (!ifs.good()) {
    MS_LOG(ERROR) << "file: " << real_path << " is not exist";
    return res;
  }
  if (!ifs.is_open()) {
    MS_LOG(ERROR) << "file: " << real_path << "open failed";
    return res;
  }
  std::string line;
  while (std::getline(ifs, line)) {
    lite::Trim(&line);
    if (line.empty() || line.at(0) == '#' || line.at(0) == '[') {
      continue;
    }
    auto index = line.find('=');
    if (index == std::string::npos) {
      MS_LOG(ERROR) << "the config file is invalid, can not find '=', please check";
      return "";
    }
    auto key = line.substr(0, index);
    auto value = line.substr(index + 1);
    lite::Trim(&key);
    lite::Trim(&value);
    if (key == target_key) {
      return value;
    }
  }
  return res;
}
}  // namespace converter
}  // namespace mindspore
