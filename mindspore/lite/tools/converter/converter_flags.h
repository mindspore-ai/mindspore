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

#ifndef CONVERTER_FLAGS_H
#define CONVERTER_FLAGS_H

#include <string>
#include "tools/common/flag_parser.h"
#include "ir/dtype/type_id.h"
#include "schema/inner/model_generated.h"

namespace mindspore {
namespace lite {
using mindspore::schema::QuantType;
using mindspore::schema::QuantType_PostTraining;
using mindspore::schema::QuantType_QUANT_NONE;
using mindspore::schema::QuantType_AwareTraining;
using mindspore::schema::QuantType_WeightQuant;
using mindspore::schema::QuantType_PostTraining;
using mindspore::schema::QuantType_PostTraining;
namespace converter {
enum FmkType {
  FmkType_TF = 0,
  FmkType_CAFFE = 1,
  FmkType_ONNX = 2,
  FmkType_MS = 3,
  FmkType_TFLITE = 4
};

class Flags : public virtual mindspore::lite::FlagParser {
 public:
  Flags();

  ~Flags() override = default;

  int Init(int argc, const char **argv);

 private:
  bool ValidateString(std::string pattern, std::string input);

 public:
  std::string modelFile;
  std::string outputFile;
  std::string fmkIn;
  FmkType fmk;
  std::string weightFile;
  std::string inputArrays;
  std::string outputArrays;
  std::string inputShapes;
  // used for quantization
  std::string quantTypeIn;
  QuantType quantType;
  std::string inferenceType;
  // used for parse aware trainning
  std::string inputInferenceTypeIn;
  //  mindspore::predict::DataType inputInferenceType = DataType_DT_FLOAT;
  TypeId inputInferenceType = TypeId::kNumberTypeFloat;
  std::string stdDev;
  std::string mean;
  // used for post-trainning-weight
  std::string quantSize;
  std::string bitNum;
  std::string configFile;
  bool formatTrans = true;
  std::string convWeightQuantChannelThreshold;
};
}  // namespace converter
}  // namespace lite
}  // namespace mindspore

#endif

