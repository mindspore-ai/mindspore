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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_CONVERTER_FLAGS_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_CONVERTER_FLAGS_H

#include <string>
#include <vector>
#include "include/registry/framework.h"
#include "tools/common/flag_parser.h"
#include "ir/dtype/type_id.h"
#include "schema/inner/model_generated.h"

namespace mindspore {
namespace lite {
using mindspore::schema::QuantType;
using mindspore::schema::QuantType_AwareTraining;
using mindspore::schema::QuantType_PostTraining;
using mindspore::schema::QuantType_QUANT_NONE;
using mindspore::schema::QuantType_WeightQuant;
namespace converter {
enum ParallelSplitType { SplitNo = 0, SplitByUserRatio = 1, SplitByUserAttr = 2 };
constexpr auto kMaxSplitRatio = 10;
constexpr auto kComputeRate = "computeRate";
constexpr auto kSplitDevice0 = "device0";
constexpr auto kSplitDevice1 = "device1";
struct ParallelSplitConfig {
  ParallelSplitType parallel_split_type_ = SplitNo;
  std::vector<int64_t> parallel_compute_rates_;
  std::vector<std::string> parallel_devices_;
};

class Flags : public virtual mindspore::lite::FlagParser {
 public:
  Flags();

  ~Flags() override = default;

  int InitInputOutputDataType();

  int InitFmk();

  bool IsValidNum(const std::string &str, int *num);

  int QuantParamInputCheck();

  int InitQuantParam();

  int InitTrainModel();

  int InitConfigFile();

  int InitInTensorShape();

  int Init(int argc, const char **argv);

 public:
  std::string modelFile;
  std::string outputFile;
  std::string fmkIn;
  FmkType fmk;
  std::string weightFile;
  TypeId inputDataType;
  TypeId outputDataType;
  std::string saveFP16Str = "off";
  bool saveFP16 = false;
  // used for quantization
  std::string quantTypeStr;
  schema::QuantType quantType;
  std::string inputDataTypeStr;
  std::string outputDataTypeStr;
  // used for post-trainning-weight
  std::string quantWeightSizeStr;
  int quantWeightSize;
  std::string bitNumIn;
  int bitNum;
  ParallelSplitConfig parallel_split_config_{};
  std::string configFile;
  std::string quantWeightChannelStr;
  int quantWeightChannel;
  std::string trainModelIn;
  bool trainModel = false;
  std::vector<std::string> pluginsPath;
  bool disableFusion = false;
  std::string inTensorShape;
  std::string dec_key = "";
  std::string dec_mode = "AES-GCM";
};

bool CheckOfflineParallelConfig(const std::string &file, ParallelSplitConfig *parallel_split_config);

std::string GetStrFromConfigFile(const std::string &file, const std::string &target_key);

std::vector<std::string> SplitStringToVector(const std::string &raw_str, const char &delimiter);
}  // namespace converter
}  // namespace lite
}  // namespace mindspore

#endif
