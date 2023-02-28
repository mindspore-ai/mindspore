/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_CONVERTER_LITE_CONVERTER_FLAGS_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_CONVERTER_LITE_CONVERTER_FLAGS_H_

#include <string>
#include <vector>
#include <map>
#include "include/api/format.h"
#include "include/api/data_type.h"
#include "include/registry/converter_context.h"
#include "tools/common/flag_parser.h"

namespace mindspore {
namespace converter {
class Flags : public virtual mindspore::lite::FlagParser {
 public:
  Flags();
  ~Flags() = default;
  int InitInputOutputDataType();
  int InitFmk();
  int InitTrainModel();
  int InitInTensorShape() const;
  int InitGraphInputFormat();
  int InitEncrypt();
  int InitPreInference();
  int InitSaveFP16();
  int InitNoFusion();
  int InitOptimize();
  int InitSaveType();
  int InitOptimizeTransformer();

  int Init(int argc, const char **argv);
  int PreInit(int argc, const char **argv);

  std::string fmkIn;
  FmkType fmk;
  std::string modelFile;
  std::string outputFile;
  std::string weightFile;
  std::string saveFP16Str = "off";
  bool saveFP16 = false;
  std::string noFusionStr = "false";
  bool disableFusion = false;
  std::string inputDataTypeStr;
  DataType inputDataType;
  std::string outputDataTypeStr;
  DataType outputDataType;
  std::string configFile;
  std::string trainModelIn;
  bool trainModel = false;
  std::string inTensorShape;
  mutable std::map<std::string, std::vector<int64_t>> graph_input_shape_map;
  std::string dec_key = "";
  std::string dec_mode = "AES-GCM";
  std::string graphInputFormatStr;
  mindspore::Format graphInputFormat = mindspore::DEFAULT_FORMAT;
  std::string encKeyStr;
  std::string encMode = "AES-GCM";
  std::string inferStr;
  bool infer = false;
  std::string saveTypeStr;
#if defined(ENABLE_CLOUD_FUSION_INFERENCE) || defined(ENABLE_CLOUD_INFERENCE)
  ModelType save_type = kMindIR;
#else
  ModelType save_type = kMindIR_Lite;
#endif
  std::string optimizeStr;
#ifdef ENABLE_OPENSSL
  std::string encryptionStr = "true";
  bool encryption = true;
#else
  std::string encryptionStr = "false";
  bool encryption = false;
#endif
  std::string device;
  std::string optimizeTransformerStr;
  bool optimizeTransformer = false;
};
}  // namespace converter
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_CONVERTER_LITE_CONVERTER_FLAGS_H_
