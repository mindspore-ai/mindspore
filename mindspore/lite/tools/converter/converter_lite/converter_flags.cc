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
#include "tools/converter/converter_lite/converter_flags.h"
#include <climits>
#include <cstdlib>
#include <string>
#include <fstream>
#include <vector>
#include <memory>
#include <algorithm>

#include "tools/common/string_util.h"

namespace mindspore::converter {
using mindspore::lite::RET_INPUT_PARAM_INVALID;
using mindspore::lite::RET_OK;

Flags::Flags() {
  AddFlag(&Flags::fmkIn, "fmk", "Input model framework type. TF | TFLITE | CAFFE | MINDIR | ONNX | PYTORCH", "");
  AddFlag(&Flags::modelFile, "modelFile",
          "Input model file. TF: *.pb | TFLITE: *.tflite | CAFFE: *.prototxt | MINDIR: *.mindir | ONNX: *.onnx", "");
  AddFlag(&Flags::outputFile, "outputFile", "Output model file path.", "");
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
          "Assign the input format of exported model. Only Valid for 4-dimensional input. NHWC | NCHW", "");
#ifdef ENABLE_OPENSSL
  AddFlag(&Flags::encryptionStr, "encryption",
          "Whether to export the encryption model."
          "true | false",
          "true");
  AddFlag(&Flags::encKeyStr, "encryptKey",
          "The key used to encrypt the file, expressed in hexadecimal characters. Only support AES-GCM and the key "
          "length is 16.",
          "");
#endif
  AddFlag(&Flags::inferStr, "infer",
          "Whether to do pre-inference after convert. "
          "true | false",
          "false");
  AddFlag(&Flags::exportMindIR, "exportMindIR", "MINDIR  | MINDIR_LITE", "MINDIR_LITE");
  AddFlag(&Flags::noFusionStr, "NoFusion",
          "Avoid fusion optimization true|false. NoFusion is true when saveType is MINDIR.", "");
  AddFlag(&Flags::device, "device",
          "Set the target device, support Ascend, Ascend310 and Ascend310P will be deprecated.", "");
  AddFlag(&Flags::saveTypeStr, "saveType", "The type of saved model. MINDIR | MINDIR_LITE", "MINDIR_LITE");
  AddFlag(&Flags::optimizeStr, "optimize", "The type of optimization. none | general | ascend_oriented", "general");
  AddFlag(&Flags::optimizeTransformerStr, "optimizeTransformer", "Enable Fast-Transformer fusion true|false", "false");
}

int Flags::InitInputOutputDataType() {
  // value check not here, it is in converter c++ API's CheckValueParam method.
  std::map<std::string, DataType> StrToEnumDataTypeMap = {{"FLOAT", DataType::kNumberTypeFloat32},
                                                          {"INT8", DataType::kNumberTypeInt8},
                                                          {"UINT8", DataType::kNumberTypeUInt8},
                                                          {"DEFAULT", DataType::kTypeUnknown}};
  if (StrToEnumDataTypeMap.find(this->inputDataTypeStr) != StrToEnumDataTypeMap.end()) {
    this->inputDataType = StrToEnumDataTypeMap.at(this->inputDataTypeStr);
  } else {
    std::cerr
      << "INPUT INVALID: inputDataType is invalid: %s, supported inputDataType: FLOAT | INT8 | UINT8 | DEFAULT, got: "
      << this->inputDataTypeStr << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }

  if (StrToEnumDataTypeMap.find(this->outputDataTypeStr) != StrToEnumDataTypeMap.end()) {
    this->outputDataType = StrToEnumDataTypeMap.at(this->outputDataTypeStr);
  } else {
    std::cerr
      << "INPUT INVALID: outputDataType is invalid: %s, supported outputDataType: FLOAT | INT8 | UINT8 | DEFAULT, got: "
      << this->outputDataTypeStr << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }

  return RET_OK;
}

int Flags::InitFmk() {
  // value check not here, it is in converter c++ API's CheckValueParam method.
  std::map<std::string, FmkType> StrToEnumFmkTypeMap = {{"CAFFE", kFmkTypeCaffe},   {"MINDIR", kFmkTypeMs},
                                                        {"TFLITE", kFmkTypeTflite}, {"ONNX", kFmkTypeOnnx},
                                                        {"TF", kFmkTypeTf},         {"PYTORCH", kFmkTypePytorch}};
  if (StrToEnumFmkTypeMap.find(this->fmkIn) != StrToEnumFmkTypeMap.end()) {
    this->fmk = StrToEnumFmkTypeMap.at(this->fmkIn);
  } else {
    std::cerr << "INPUT ILLEGAL: fmk must be TF|TFLITE|CAFFE|MINDIR|ONNX" << std::endl;
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

  return RET_OK;
}

int Flags::InitInTensorShape() const {
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
    constexpr int kMinShapeSizeInStr = 2;
    if (string_split.size() < kMinShapeSizeInStr) {
      MS_LOG(ERROR) << "shape size must not be less than " << kMinShapeSizeInStr;
      return lite::RET_INPUT_PARAM_INVALID;
    }
    auto name = string_split[0];
    for (size_t i = 1; i < string_split.size() - 1; ++i) {
      name += ":" + string_split[i];
    }
    if (name.empty()) {
      MS_LOG(ERROR) << "input tensor name is empty";
      return lite::RET_INPUT_PARAM_INVALID;
    }
    auto dim_strs = string_split[string_split.size() - 1];
    if (dim_strs.empty()) {
      MS_LOG(ERROR) << "input tensor dim string is empty";
      return lite::RET_INPUT_PARAM_INVALID;
    }
    auto dims = lite::StrSplit(dim_strs, std::string(","));
    if (dims.empty()) {
      MS_LOG(ERROR) << "input tensor dim is empty";
      return lite::RET_INPUT_PARAM_INVALID;
    }
    for (const auto &dim : dims) {
      int64_t dim_value;
      try {
        dim_value = std::stoi(dim);
      } catch (const std::exception &e) {
        MS_LOG(ERROR) << "Get dim failed: " << e.what();
        return lite::RET_INPUT_PARAM_INVALID;
      }
      shape.push_back(dim_value);
    }
    graph_input_shape_map[name] = shape;
  }
  return RET_OK;
}

int Flags::InitGraphInputFormat() {
  // value check not here, it is in converter c++ API's CheckValueParam method.
  std::map<std::string, Format> StrToEnumFormatMap = {{"NHWC", NHWC}, {"NCHW", NCHW}};
  if (StrToEnumFormatMap.find(this->graphInputFormatStr) != StrToEnumFormatMap.end()) {
    graphInputFormat = StrToEnumFormatMap.at(this->graphInputFormatStr);
  } else if (!this->graphInputFormatStr.empty()) {
    MS_LOG(ERROR) << "graph input format is invalid.";
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

int Flags::InitSaveFP16() {
  if (saveFP16Str == "on") {
    saveFP16 = true;
  } else if (saveFP16Str == "off") {
    saveFP16 = false;
  } else {
    std::cerr << "Init save_fp16 failed." << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

int Flags::InitPreInference() {
  if (this->inferStr == "true") {
    this->infer = true;
  } else if (this->inferStr == "false") {
    this->infer = false;
  } else {
    std::cerr << "INPUT ILLEGAL: infer must be true|false " << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

int Flags::InitNoFusion() {
  if (this->noFusionStr == "true") {
    this->disableFusion = true;
  } else if (this->noFusionStr == "false") {
    this->disableFusion = false;
  } else if (!this->noFusionStr.empty()) {
    std::cerr << "INPUT ILLEGAL: NoFusion must be true|false " << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

int Flags::InitOptimize() {
  // For compatibility of interface, the check will be removed when nofusion is deleted
  if (!this->noFusionStr.empty() || !this->device.empty()) {
    return RET_OK;
  }
  if (this->optimizeStr == "none") {
    this->disableFusion = true;
  } else if (this->optimizeStr == "general") {
    this->disableFusion = false;
  } else if (this->optimizeStr == "ascend_oriented") {
    this->disableFusion = false;
    this->device = "Ascend";
  } else if (!this->optimizeStr.empty()) {
    std::cerr << "INPUT ILLEGAL: optimize must be none|general|ascend_oriented " << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

int Flags::InitExportMindIR() {
  // value check not here, it is in converter c++ API's CheckValueParam method.
  std::map<std::string, ModelType> StrToEnumModelTypeMap = {{"MINDIR", kMindIR}, {"MINDIR_LITE", kMindIR_Lite}};
  if (StrToEnumModelTypeMap.find(this->exportMindIR) != StrToEnumModelTypeMap.end()) {
    this->export_mindir = StrToEnumModelTypeMap.at(this->exportMindIR);
  } else {
    std::cerr << "INPUT ILLEGAL: exportMindIR must be MINDIR|MINDIR_LITE " << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }

  if ((this->exportMindIR == "MINDIR") && (this->optimizeTransformer == false)) {
    this->disableFusion = true;
  }
  return RET_OK;
}

int Flags::InitEncrypt() {
  if (this->encryptionStr == "true") {
    this->encryption = true;
  } else if (this->encryptionStr == "false") {
    this->encryption = false;
  } else {
    std::cerr << "INPUT ILLEGAL: encryption must be true|false " << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

int Flags::InitSaveType() {
  // For compatibility of interface, the check will be removed when exportMindIR is deleted
  if (this->exportMindIR == "MINDIR") {
    return RET_OK;
  }
  std::map<std::string, ModelType> StrToEnumModelTypeMap = {{"MINDIR", kMindIR}, {"MINDIR_LITE", kMindIR_Lite}};
  if (StrToEnumModelTypeMap.find(this->saveTypeStr) != StrToEnumModelTypeMap.end()) {
    this->export_mindir = StrToEnumModelTypeMap.at(this->saveTypeStr);
  } else {
    std::cerr << "INPUT ILLEGAL: saveType must be MINDIR|MINDIR_LITE " << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }

  return RET_OK;
}

int Flags::InitOptimizeTransformer() {
  if (this->optimizeTransformerStr == "true") {
    this->optimizeTransformer = true;
  } else if (this->optimizeTransformerStr == "false") {
    this->optimizeTransformer = false;
  } else {
    std::cerr << "INPUT ILLEGAL:  optimizeTransformer must be true|false " << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

int Flags::PreInit(int argc, const char **argv) {
  if (argc == 1) {
    std::cout << this->Usage() << std::endl;
    return lite::RET_OK;
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

  if (this->fmkIn.empty()) {
    std::cerr << "INPUT MISSING: fmk is necessary" << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }

  return RET_OK;
}

int Flags::Init(int argc, const char **argv) {
  auto ret = PreInit(argc, argv);
  if (ret != RET_OK) {
    return ret;
  }
  ret = InitSaveFP16();
  if (ret != RET_OK) {
    std::cerr << "Init save fp16 failed." << std::endl;
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

  ret = InitEncrypt();
  if (ret != RET_OK) {
    std::cerr << "Init encrypt failed." << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }

  ret = InitOptimizeTransformer();
  if (ret != RET_OK) {
    std::cerr << "Init optimize transformers failed." << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }

  ret = InitPreInference();
  if (ret != RET_OK) {
    std::cerr << "Init pre inference failed." << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }

  ret = InitExportMindIR();
  if (ret != RET_OK) {
    std::cerr << "Init export mindir failed." << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }

  ret = InitSaveType();
  if (ret != RET_OK) {
    std::cerr << "Init save type failed." << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }

  ret = InitNoFusion();
  if (ret != RET_OK) {
    std::cerr << "Init no fusion failed." << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }

  ret = InitOptimize();
  if (ret != RET_OK) {
    std::cerr << "Init optimize failed" << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }

  return RET_OK;
}
}  // namespace mindspore::converter
