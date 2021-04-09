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
#include <regex>
#include <string>
#include <algorithm>
#include "ir/dtype/type_id.h"

namespace mindspore {
namespace lite {
namespace converter {
Flags::Flags() {
  AddFlag(&Flags::fmkIn, "fmk", "Input model framework type. TF | TFLITE | CAFFE | MINDIR | ONNX", "");
  AddFlag(&Flags::modelFile, "modelFile",
          "Input model file. TF: *.pb | TFLITE: *.tflite | CAFFE: *.prototxt | MINDIR: *.mindir | ONNX: *.onnx", "");
  AddFlag(&Flags::outputFile, "outputFile", "Output model file path. Will add .ms automatically", "");
  AddFlag(&Flags::weightFile, "weightFile", "Input model weight file. Needed when fmk is CAFFE. CAFFE: *.caffemodel",
          "");
  AddFlag(&Flags::inputDataTypeIn, "inputDataType",
          "Data type of input tensors, default is same with the type defined in model. FLOAT | INT8 | UINT8 | DEFAULT",
          "DEFAULT");
  AddFlag(&Flags::outputDataTypeIn, "outputDataType",
          "Data type of output and output tensors, default is same with the type defined in model. FLOAT | INT8 | "
          "UINT8 | DEFAULT",
          "DEFAULT");
  AddFlag(&Flags::quantTypeIn, "quantType", "Quantization Type. PostTraining | WeightQuant", "");
  AddFlag(&Flags::bitNumIn, "bitNum", "Weight quantization bitNum", "8");
  AddFlag(&Flags::quantWeightSizeIn, "quantWeightSize", "Weight quantization size threshold", "0");
  AddFlag(&Flags::quantWeightChannelIn, "quantWeightChannel", "Channel threshold for weight quantization", "16");
  AddFlag(&Flags::configFile, "configFile", "Configuration for post-training.", "");
  AddFlag(&Flags::trainModelIn, "trainModel",
          "whether the model is going to be trained on device. "
          "true | false",
          "false");
}

int Flags::InitInputOutputDataType() {
  if (this->inputDataTypeIn == "FLOAT") {
    this->inputDataType = TypeId::kNumberTypeFloat32;
  } else if (this->inputDataTypeIn == "INT8") {
    this->inputDataType = TypeId::kNumberTypeInt8;
  } else if (this->inputDataTypeIn == "UINT8") {
    this->inputDataType = TypeId::kNumberTypeUInt8;
  } else if (this->inputDataTypeIn == "DEFAULT") {
    this->inputDataType = TypeId::kTypeUnknown;
  } else {
    std::cerr << "INPUT INVALID: inputDataType is invalid: %s, supported inputDataType: FLOAT | INT8 | UINT8 | DEFAULT",
      this->inputDataTypeIn.c_str();
    return RET_INPUT_PARAM_INVALID;
  }

  if (this->outputDataTypeIn == "FLOAT") {
    this->outputDataType = TypeId::kNumberTypeFloat32;
  } else if (this->outputDataTypeIn == "INT8") {
    this->outputDataType = TypeId::kNumberTypeInt8;
  } else if (this->outputDataTypeIn == "UINT8") {
    this->outputDataType = TypeId::kNumberTypeUInt8;
  } else if (this->outputDataTypeIn == "DEFAULT") {
    this->outputDataType = TypeId::kTypeUnknown;
  } else {
    std::cerr
      << "INPUT INVALID: outputDataType is invalid: %s, supported outputDataType: FLOAT | INT8 | UINT8 | DEFAULT",
      this->outputDataTypeIn.c_str();
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
    std::cerr << "INPUT ILLEGAL: fmk must be TF|TFLITE|CAFFE|MINDIR|ONNX";
    return RET_INPUT_PARAM_INVALID;
  }

  if (this->fmk != FmkType_CAFFE && !weightFile.empty()) {
    std::cerr << "INPUT ILLEGAL: weightFile is not a valid flag";
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

bool Flags::IsValidNum(const std::string &str, int *num) {
  char *ptr = nullptr;
  *num = strtol(str.c_str(), &ptr, 10);
  return ptr == (str.c_str() + str.size());
}

int Flags::QuantParamInputCheck() {
  if (!Flags::IsValidNum(this->quantWeightChannelIn, &this->quantWeightChannel)) {
    std::cerr << "quantWeightChannel should be a valid number.";
    return RET_INPUT_PARAM_INVALID;
  }
  if (this->quantWeightChannel < 0) {
    std::cerr << "quantWeightChannel should be greater than or equal to zero.";
    return RET_INPUT_PARAM_INVALID;
  }
  if (!Flags::IsValidNum(this->quantWeightSizeIn, &this->quantWeightSize)) {
    std::cerr << "quantWeightSize should be a valid number.";
    return RET_INPUT_PARAM_INVALID;
  }
  if (this->quantWeightSize < 0) {
    std::cerr << "quantWeightSize should be greater than or equal to zero.";
    return RET_INPUT_PARAM_INVALID;
  }
  if (!Flags::IsValidNum(this->bitNumIn, &this->bitNum)) {
    std::cerr << "bitNum should be a valid number.";
    return RET_INPUT_PARAM_INVALID;
  }
  if (this->bitNum <= 0 || this->bitNum > 16) {
    std::cerr << "bitNum should be greater than zero and lesser than 16 currently.";
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

int Flags::InitQuantParam() {
  if (this->quantTypeIn == "WeightQuant") {
    this->quantType = QuantType_WeightQuant;
  } else if (this->quantTypeIn == "PostTraining") {
    this->quantType = QuantType_PostTraining;
  } else if (this->quantTypeIn.empty()) {
    this->quantType = QuantType_QUANT_NONE;
  } else {
    std::cerr << "INPUT ILLEGAL: quantType must be WeightQuant|PostTraining";
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
    std::cerr << "INPUT ILLEGAL: trainModel must be true|false ";
    return RET_INPUT_PARAM_INVALID;
  }

  if (this->trainModel) {
    if (this->fmk != FmkType_MS) {
      std::cerr << "INPUT ILLEGAL: train model converter supporting only MINDIR format";
      return RET_INPUT_PARAM_INVALID;
    }
    if ((this->inputDataType != TypeId::kNumberTypeFloat32) && (this->inputDataType != TypeId::kTypeUnknown)) {
      std::cerr << "INPUT ILLEGAL: train model converter supporting only FP32 input tensors";
      return RET_INPUT_PARAM_INVALID;
    }
    if ((this->outputDataType != TypeId::kNumberTypeFloat32) && (this->outputDataType != TypeId::kTypeUnknown)) {
      std::cerr << "INPUT ILLEGAL: train model converter supporting only FP32 output tensors";
      return RET_INPUT_PARAM_INVALID;
    }
    if (this->quantType != QuantType_QUANT_NONE) {
      std::cerr << "INPUT ILLEGAL: train model converter is not supporting quantization";
      return RET_INPUT_PARAM_INVALID;
    }
  }
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
    std::cerr << err.Get();
    std::cerr << this->Usage() << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }

  if (this->help) {
    std::cout << this->Usage() << std::endl;
    return RET_SUCCESS_EXIT;
  }
  if (this->modelFile.empty()) {
    std::cerr << "INPUT MISSING: model file path is necessary";
    return RET_INPUT_PARAM_INVALID;
  }
  if (this->outputFile.empty()) {
    std::cerr << "INPUT MISSING: output file path is necessary";
    return RET_INPUT_PARAM_INVALID;
  }

#ifdef _WIN32
  replace(this->outputFile.begin(), this->outputFile.end(), '/', '\\');
#endif

  if (this->outputFile.rfind('/') == this->outputFile.length() - 1 ||
      this->outputFile.rfind('\\') == this->outputFile.length() - 1) {
    std::cerr << "INPUT ILLEGAL: outputFile must be a valid file path";
    return RET_INPUT_PARAM_INVALID;
  }

  if (this->fmkIn.empty()) {
    std::cerr << "INPUT MISSING: fmk is necessary";
    return RET_INPUT_PARAM_INVALID;
  }

  ret = InitInputOutputDataType();
  if (ret != RET_OK) {
    std::cerr << "Init input output datatype failed.";
    return RET_INPUT_PARAM_INVALID;
  }

  ret = InitFmk();
  if (ret != RET_OK) {
    std::cerr << "Init fmk failed.";
    return RET_INPUT_PARAM_INVALID;
  }

  ret = InitQuantParam();
  if (ret != RET_OK) {
    std::cerr << "Init quant param failed.";
    return RET_INPUT_PARAM_INVALID;
  }

  ret = InitTrainModel();
  if (ret != RET_OK) {
    std::cerr << "Init train model failed.";
    return RET_INPUT_PARAM_INVALID;
  }

  return RET_OK;
}
}  // namespace converter
}  // namespace lite
}  // namespace mindspore
