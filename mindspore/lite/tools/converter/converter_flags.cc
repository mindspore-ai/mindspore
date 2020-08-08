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

#include <regex>
#include <string>
#include "tools/converter/converter_flags.h"


namespace mindspore {
namespace lite {
namespace converter {
Flags::Flags() {
  AddFlag(&Flags::fmkIn, "fmk", "Input model framework type. TF | CAFFE | ONNX | MS | TFLITE", "");
  AddFlag(&Flags::modelFile, "modelFile",
          "Input model file path. TF: *.pb | CAFFE: *.prototxt | ONNX: *.onnx | MS: *.ms", "");
  AddFlag(&Flags::outputFile, "outputFile", "Output model file path. Will add .ms automatically", "");
  AddFlag(&Flags::weightFile, "weightFile",
          "Input model weight file path. Needed when fmk is CAFFE. CAFFE: *.caffemodel", "");
  AddFlag(&Flags::inferenceType, "inferenceType",
          "Real data type saved in output file, reserved param, NOT used for now. FLOAT | FP16 | UINT8", "FLOAT");
  AddFlag(&Flags::quantTypeIn, "quantType", "Quantization Type. AwareTrainning | WeightQuant | PostTraining", "");
  AddFlag(&Flags::inputInferenceTypeIn, "inputInferenceType", "Input inference data type. FLOAT | UINT8", "FLOAT");
  AddFlag(&Flags::stdDev, "stdDev", "Standard deviation value for aware-quantization", "128");
  AddFlag(&Flags::mean, "mean", "Mean value for aware-quantization", "127");
  AddFlag(&Flags::quantSize, "quantSize", "Weight quantization size threshold", "0");
  AddFlag(&Flags::configFile, "config_file", "Configuration for post-training.", "");
  AddFlag(&Flags::formatTrans, "formatTrans", "whether transform format. true | false", "true");
}

int Flags::Init(int argc, const char **argv) {
  Option<std::string> err = this->ParseFlags(argc, argv);

  if (err.IsSome()) {
    MS_LOG(ERROR) << err.Get();
    std::cerr << this->Usage() << std::endl;
    return 1;
  }

  if (this->help) {
    std::cerr << this->Usage() << std::endl;
    return 0;
  }
  if (this->modelFile.empty()) {
    MS_LOG(ERROR) << "INPUT MISSING: model file path is necessary";
    return 1;
  }
  if (this->outputFile.empty()) {
    MS_LOG(ERROR) << "INPUT MISSING: output file path is necessary";
    return 1;
  }

  if (this->outputFile.rfind('/') == this->outputFile.length() - 1) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: outputFile must be a valid file path";
    return 1;
  }

  if (this->fmkIn.empty()) {
    MS_LOG(ERROR) << "INPUT MISSING: fmk is necessary";
    return 1;
  }
  if (this->inputInferenceTypeIn == "FLOAT") {
    this->inputInferenceType = 0;
  } else if (this->inputInferenceTypeIn == "UINT8") {
    this->inputInferenceType = 1;
  } else {
    MS_LOG(ERROR) << "INPUT INVALID: inputInferenceType is invalid: %s", this->inputInferenceTypeIn.c_str();
    return 1;
  }
  if (this->fmkIn == "TF") {
    this->fmk = FmkType_TF;
  } else if (this->fmkIn == "CAFFE") {
    this->fmk = FmkType_CAFFE;
  } else if (this->fmkIn == "ONNX") {
    this->fmk = FmkType_ONNX;
  } else if (this->fmkIn == "MS") {
    this->fmk = FmkType_MS;
  } else if (this->fmkIn == "TFLITE") {
    this->fmk = FmkType_TFLITE;
  } else {
    MS_LOG(ERROR) << "INPUT ILLEGAL: fmk must be TF|CAFFE|ONNX|MS";
    return 1;
  }

  if (this->fmk != FmkType_CAFFE && !weightFile.empty()) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: weightFile is not a valid flag";
    return 1;
  }
  if (this->quantTypeIn == "AwareTrainning") {
    this->quantType = QuantType_AwareTrainning;
  } else if (this->quantTypeIn == "WeightQuant") {
    this->quantType = QuantType_WeightQuant;
  } else if (this->quantTypeIn == "PostTraining") {
    this->quantType = QuantType_PostTraining;
  } else if (this->quantTypeIn.empty()) {
    this->quantType = QuantType_QUANT_NONE;
  } else {
    MS_LOG(ERROR) << "INPUT ILLEGAL: quantType must be AwareTrainning|WeightQuant|PostTraining";
    return 1;
  }

  //  auto status = ValidateAwareQuantizerCLI();
  //  if (status != RET_OK) {
  //    MS_PRINT_ERROR("Parse aware quantization command line failed: %d", status);
  //    return status;
  //  }
  //  status = ValidateWeighQuantCLI();
  //  if (status != RET_OK) {
  //    MS_PRINT_ERROR("ValidateWeighQuantCLI failed: %d", status);
  //    return status;
  //  }
  return 0;
}

// bool Flags::ValidateString(const string pattern, const string input) {
//  std::regex repPattern(pattern, std::regex_constants::extended);
//  std::match_results<string::const_iterator> regResult;
//  return regex_match(input, regResult, repPattern);
//}

// int Flags::ValidateAwareQuantizerCLI() {
//  // check input inference type
//  if (this->inputInferenceType == DataType_DT_FLOAT) {
//    if (this->mean.empty()) {
//      MS_PRINT_ERROR("mean value shound not be null!")
//      return RET_PARAM_INVALID;
//    }
//    if (this->stdDev.empty()) {
//      MS_PRINT_ERROR("standard deviation value shound not be null!")
//      return RET_PARAM_INVALID;
//    }
//    const std::string pattern = "^[+-]?([0-9]*\.?[0-9]+|[0-9]+\.?[0-9]*)([eE][+-]?[0-9]+)?$";
//    if (!ValidateString(pattern, this->mean)) {
//      MS_PRINT_ERROR("invalid input mean values: %s", this->mean.c_str());
//      return RET_PARAM_INVALID;
//    }
//    if (!ValidateString(pattern, this->stdDev)) {
//      MS_PRINT_ERROR("invalid input standard deviation value: %s", this->stdDev.c_str());
//      return RET_PARAM_INVALID;
//    }
//  } else {
//    if (!this->mean.empty()) {
//      MS_PRINT_INFO("useless mean value: %s", this->mean.c_str());
//    }
//    if (!this->stdDev.empty()) {
//      MS_PRINT_INFO("useless stdDev value: %s", this->stdDev.c_str());
//    }
//  }
//  return RET_OK;
//}

// int Flags::ValidateWeighQuantCLI() {
//  if (!this->quantSize.empty()) {
//    if (!ValidateString("^[0-9]*$", this->quantSize)) {
//      MS_PRINT_ERROR("invalid input quantSize: %s, only support positive integer type!", this->quantSize.c_str());
//      return RET_PARAM_INVALID;
//    }
//  }
//  return RET_OK;
//}
}  // namespace converter
}  // namespace lite
}  // namespace mindspore

