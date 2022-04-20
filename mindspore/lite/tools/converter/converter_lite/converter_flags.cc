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
  AddFlag(&Flags::exportMindIR, "exportMindIR",
          "Whether to export MindIR pb. "
          "true | false",
          "false");
  AddFlag(&Flags::noFusionStr, "NoFusion", "Avoid fusion optimization true|false", "false");
}

int Flags::InitInputOutputDataType() {
  if (this->inputDataTypeStr == "FLOAT") {
    this->inputDataType = DataType::kNumberTypeFloat32;
  } else if (this->inputDataTypeStr == "INT8") {
    this->inputDataType = DataType::kNumberTypeInt8;
  } else if (this->inputDataTypeStr == "UINT8") {
    this->inputDataType = DataType::kNumberTypeUInt8;
  } else if (this->inputDataTypeStr == "DEFAULT") {
    this->inputDataType = DataType::kTypeUnknown;
  } else {
    std::cerr
      << "INPUT INVALID: inputDataType is invalid: %s, supported inputDataType: FLOAT | INT8 | UINT8 | DEFAULT, got: "
      << this->inputDataTypeStr << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }

  if (this->outputDataTypeStr == "FLOAT") {
    this->outputDataType = DataType::kNumberTypeFloat32;
  } else if (this->outputDataTypeStr == "INT8") {
    this->outputDataType = DataType::kNumberTypeInt8;
  } else if (this->outputDataTypeStr == "UINT8") {
    this->outputDataType = DataType::kNumberTypeUInt8;
  } else if (this->outputDataTypeStr == "DEFAULT") {
    this->outputDataType = DataType::kTypeUnknown;
  } else {
    std::cerr
      << "INPUT INVALID: outputDataType is invalid: %s, supported outputDataType: FLOAT | INT8 | UINT8 | DEFAULT, got: "
      << this->outputDataTypeStr << std::endl;
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
  } else if (this->fmkIn == "PYTORCH") {
    this->fmk = kFmkTypePytorch;
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
    if ((this->inputDataType != DataType::kNumberTypeFloat32) && (this->inputDataType != DataType::kTypeUnknown)) {
      std::cerr << "INPUT ILLEGAL: train model converter supporting only FP32 input tensors" << std::endl;
      return RET_INPUT_PARAM_INVALID;
    }
    if ((this->outputDataType != DataType::kNumberTypeFloat32) && (this->outputDataType != DataType::kTypeUnknown)) {
      std::cerr << "INPUT ILLEGAL: train model converter supporting only FP32 output tensors" << std::endl;
      return RET_INPUT_PARAM_INVALID;
    }
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
      return mindspore::lite::RET_ERROR;
    }
    auto name = string_split[0];
    for (size_t i = 1; i < string_split.size() - 1; ++i) {
      name += ":" + string_split[i];
    }
    if (name.empty()) {
      MS_LOG(ERROR) << "input tensor name is empty";
      return lite::RET_ERROR;
    }
    auto dim_strs = string_split[string_split.size() - 1];
    if (dim_strs.empty()) {
      MS_LOG(ERROR) << "input tensor dim string is empty";
      return lite::RET_ERROR;
    }
    auto dims = lite::StrSplit(dim_strs, std::string(","));
    if (dims.empty()) {
      MS_LOG(ERROR) << "input tensor dim is empty";
      return lite::RET_ERROR;
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
    graph_input_shape_map[name] = shape;
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
  } else {
    std::cerr << "INPUT ILLEGAL: NoFusion must be true|false " << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

int Flags::InitExportMindIR() {
  if (this->exportMindIR == "true") {
    this->export_mindir = true;
  } else if (this->exportMindIR == "false") {
    this->export_mindir = false;
  } else {
    std::cerr << "INPUT ILLEGAL: exportMindIR must be true|false " << std::endl;
    return RET_INPUT_PARAM_INVALID;
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
  if (this->encryption) {
    if (encKeyStr.empty()) {
      MS_LOG(ERROR) << "If you don't need to use model encryption, please set --encryption=false.";
      return RET_INPUT_PARAM_INVALID;
    }
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
    return lite::RET_OK;
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
  ret = InitPreInference();
  if (ret != RET_OK) {
    std::cerr << "Init pre inference failed." << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }
  ret = InitNoFusion();
  if (ret != RET_OK) {
    std::cerr << "Init no fusion failed." << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }

  ret = InitExportMindIR();
  if (ret != RET_OK) {
    std::cerr << "Init export mindir failed." << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}
}  // namespace mindspore::converter
