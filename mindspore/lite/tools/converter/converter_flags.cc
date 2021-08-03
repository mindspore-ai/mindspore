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
#include <algorithm>
#include <fstream>
#include <vector>
#include <memory>
#include "ir/dtype/type_id.h"
#include "common/file_utils.h"
#include "tools/converter/converter_context.h"

namespace mindspore {
namespace lite {
namespace converter {
namespace {
constexpr int kBase = 10;
constexpr int kQuantBitNumInt16 = 16;
constexpr int kPathLengthUpperLimit = 1024;
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
  AddFlag(&Flags::quantTypeStr, "quantType", "Quantization Type. PostTraining | WeightQuant", "");
  AddFlag(&Flags::bitNumIn, "bitNum", "Weight quantization bitNum", "8");
  AddFlag(&Flags::quantWeightSizeStr, "quantWeightSize", "Weight quantization size threshold", "0");
  AddFlag(&Flags::quantWeightChannelStr, "quantWeightChannel", "Channel threshold for weight quantization", "16");
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
          "Set shape for model inputs for further optimizing model. Output model will lose dynamic shape feature of "
          "set this option. Model input shapes is same with origin model by default."
          "e.g. inTensor1:1,32,32,32;inTensor2:1,1,32,32,4",
          "");
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
    this->fmk = FmkType_CAFFE;
  } else if (this->fmkIn == "MINDIR") {
    this->fmk = FmkType_MS;
  } else if (this->fmkIn == "TFLITE") {
    this->fmk = FmkType_TFLITE;
  } else if (this->fmkIn == "ONNX") {
    this->fmk = FmkType_ONNX;
  } else if (this->fmkIn == "TF") {
    this->fmk = FmkType_TF;
  } else {
    std::cerr << "INPUT ILLEGAL: fmk must be TF|TFLITE|CAFFE|MINDIR|ONNX" << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }

  if (this->fmk != FmkType_CAFFE && !weightFile.empty()) {
    std::cerr << "INPUT ILLEGAL: weightFile is not a valid flag" << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

bool Flags::IsValidNum(const std::string &str, int *num) {
  char *ptr = nullptr;
  *num = strtol(str.c_str(), &ptr, kBase);
  return ptr == (str.c_str() + str.size());
}

int Flags::QuantParamInputCheck() {
  if (!Flags::IsValidNum(this->quantWeightChannelStr, &this->quantWeightChannel)) {
    std::cerr << "quantWeightChannel should be a valid number." << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }
  if (this->quantWeightChannel < 0) {
    std::cerr << "quantWeightChannel should be greater than or equal to zero." << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }
  if (!Flags::IsValidNum(this->quantWeightSizeStr, &this->quantWeightSize)) {
    std::cerr << "quantWeightSize should be a valid number." << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }
  if (this->quantWeightSize < 0) {
    std::cerr << "quantWeightSize should be greater than or equal to zero." << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }
  if (!Flags::IsValidNum(this->bitNumIn, &this->bitNum)) {
    std::cerr << "bitNum should be a valid number." << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }
  if (this->bitNum <= 0 || this->bitNum > kQuantBitNumInt16) {
    std::cerr << "bitNum should be greater than zero and lesser than 16 currently." << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

int Flags::InitQuantParam() {
  if (this->quantTypeStr == "WeightQuant") {
    this->quantType = QuantType_WeightQuant;
  } else if (this->quantTypeStr == "PostTraining") {
    this->quantType = QuantType_PostTraining;
  } else if (this->quantTypeStr.empty()) {
    this->quantType = QuantType_QUANT_NONE;
  } else {
    std::cerr << "INPUT ILLEGAL: quantType must be WeightQuant|PostTraining" << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }

  auto ret = QuantParamInputCheck();
  return ret;
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
    if (this->fmk != FmkType_MS) {
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
  std::string content = this->inTensorShape;
  std::vector<int64_t> shape;
  auto shape_strs = StrSplit(content, std::string(";"));
  for (const auto &shape_str : shape_strs) {
    shape.clear();
    auto string_split = StrSplit(shape_str, std::string(":"));
    auto name = string_split[0];
    if (name.empty()) {
      MS_LOG(ERROR) << "input tensor name is empty";
    }
    auto dim_strs = string_split[1];
    if (dim_strs.empty()) {
      MS_LOG(ERROR) << "input tensor dim string is empty";
    }
    auto dims = StrSplit(dim_strs, std::string(","));
    if (dims.empty()) {
      MS_LOG(ERROR) << "input tensor dim is empty";
    }
    for (const auto &dim : dims) {
      if (std::stoi(dim) < 0) {
        MS_LOG(ERROR) << "Unsupported dim < 0.";
        return RET_ERROR;
      } else {
        shape.push_back(std::stoi(dim));
      }
    }
    ConverterContext::GetInstance()->UpdateGraphInputTensorShape(name, shape);
  }
  return RET_OK;
}

int Flags::InitConfigFile() {
  auto plugins_path_str = GetStrFromConfigFile(this->configFile, "plugin_path");
  if (!plugins_path_str.empty()) {
    const char *delimiter = ";";
    auto relative_path = SplitStringToVector(plugins_path_str, *delimiter);
    for (size_t i = 0; i < relative_path.size(); i++) {
      this->pluginsPath.push_back(RealPath(relative_path[i].c_str()));
    }
  }

  auto disable_fusion_flag = GetStrFromConfigFile(this->configFile, "disable_fusion");
  if (!disable_fusion_flag.empty()) {
    if (disable_fusion_flag == "on") {
      this->disableFusion = true;
    } else if (disable_fusion_flag == "off") {
      this->disableFusion = false;
    } else {
      std::cerr << "CONFIG SETTING ILLEGAL: disable_fusion should be on/off" << std::endl;
      return RET_INPUT_PARAM_INVALID;
    }
  }

  (void)CheckOfflineParallelConfig(this->configFile, &parallel_split_config_);
  return RET_OK;
}

int Flags::Init(int argc, const char **argv) {
  int ret;
  if (argc == 1) {
    std::cout << this->Usage() << std::endl;
    return RET_SUCCESS_EXIT;
  }
  Option<std::string> err = this->ParseFlags(argc, argv);

  if (err.IsSome()) {
    std::cerr << err.Get() << std::endl;
    std::cerr << this->Usage() << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }

  if (this->help) {
    std::cout << this->Usage() << std::endl;
    return RET_SUCCESS_EXIT;
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

  ret = InitQuantParam();
  if (ret != RET_OK) {
    std::cerr << "Init quant param failed." << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }

  ret = InitTrainModel();
  if (ret != RET_OK) {
    std::cerr << "Init train model failed." << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }

  if (!this->inTensorShape.empty()) {
    ret = InitInTensorShape();
    if (ret != RET_OK) {
      std::cerr << "Init input tensor shape failed." << std::endl;
      return RET_INPUT_PARAM_INVALID;
    }
  }
  return RET_OK;
}

bool CheckOfflineParallelConfig(const std::string &file, ParallelSplitConfig *parallel_split_config) {
  // device: [device0 device1] ---> {cpu, gpu}
  // computeRate: [x: y] x >=0 && y >=0 && x/y < 10
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
  std::vector<std::string> device_rates = SplitStringToVector(compute_rate_result, *delimiter);
  const char *colon = ":";
  for (const auto &device : device_rates) {
    std::vector<std::string> rate = SplitStringToVector(device, *colon);
    parallel_split_config->parallel_compute_rates_.push_back(std::stoi(rate.back()));
  }
  if (parallel_split_config->parallel_compute_rates_.size() != 2) {
    return false;
  }
  int64_t bigger_rate = INT32_MIN;
  int64_t smaller_rate = INT32_MAX;
  for (const auto &rate : parallel_split_config->parallel_compute_rates_) {
    bigger_rate = std::max(rate, bigger_rate);
    smaller_rate = std::min(rate, smaller_rate);
    if (rate <= 0 || rate > INT32_MAX) {
      return false;
    }
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
    if (line.empty()) {
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

std::vector<std::string> SplitStringToVector(const std::string &raw_str, const char &delimiter) {
  if (raw_str.empty()) {
    MS_LOG(ERROR) << "input string is empty.";
    return {};
  }
  std::vector<std::string> res;
  std::string::size_type last_pos = 0;
  auto cur_pos = raw_str.find(delimiter);
  while (cur_pos != std::string::npos) {
    res.push_back(raw_str.substr(last_pos, cur_pos - last_pos));
    cur_pos++;
    last_pos = cur_pos;
    cur_pos = raw_str.find(delimiter, cur_pos);
  }
  if (last_pos < raw_str.size()) {
    res.push_back(raw_str.substr(last_pos, raw_str.size() - last_pos + 1));
  }
  return res;
}
}  // namespace converter
}  // namespace lite
}  // namespace mindspore
