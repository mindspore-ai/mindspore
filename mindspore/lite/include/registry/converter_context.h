/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_INCLUDE_REGISTRY_CONVERTER_CONTEXT_H_
#define MINDSPORE_LITE_INCLUDE_REGISTRY_CONVERTER_CONTEXT_H_

#include <map>
#include <string>
#include <vector>
#include "include/api/types.h"
#include "include/api/dual_abi_helper.h"

namespace mindspore {
namespace converter {
constexpr auto KConverterParam = "converter_parameters";
constexpr auto KCommonQuantParam = "common_quant_param";
constexpr auto KFullQuantParam = "full_quant_param";
constexpr auto KDataPreProcess = "data_preprocess_param";
constexpr auto KMixBitWeightQuantParam = "mixed_bit_weight_quant_param";

/// \brief FmkType defined frameworks which converter tool supports.
enum MS_API FmkType : int {
  kFmkTypeTf = 0,
  kFmkTypeCaffe = 1,
  kFmkTypeOnnx = 2,
  kFmkTypeMs = 3,
  kFmkTypeTflite = 4,
  kFmkTypePytorch = 5,
};

/// \brief ConverterParameters defined read-only converter parameters used by users in ModelParser.
struct MS_API ConverterParameters {
  FmkType fmk;
  ModelType save_type = kMindIR_Lite;
  std::string model_file;
  std::string weight_file;
  std::map<std::string, std::string> attrs;
};

/// \brief ConverterContext defined is to set the basic information of the exported model.
class MS_API ConverterContext {
 public:
  /// \brief Constructor.
  ConverterContext() = default;

  /// \brief Destructor.
  ~ConverterContext() = default;

  /// \brief Static method to set exported model's output name as needed by users.
  ///
  /// \param[in] output_names Define model's output name, the order of which is consistent with the original model.
  static void SetGraphOutputTensorNames(const std::vector<std::string> &output_names) {
    SetGraphOutputTensorNames(VectorStringToChar(output_names));
  }

  /// \brief Static method to obtain the outputs' name.
  ///
  /// \return the outputs' name.
  static std::vector<std::string> GetGraphOutputTensorNames() {
    return VectorCharToString(GetGraphOutputTensorNamesInChar());
  }

  /// \brief Static method to get configure information which is used only by external extension.
  ///
  /// \param[in] section Define config section name.
  ///
  /// \return config key-value map.
  static std::map<std::string, std::string> GetConfigInfo(const std::string &section) {
    return MapVectorCharToString(GetConfigInfo(StringToChar(section)));
  }

 private:
  static void SetGraphOutputTensorNames(const std::vector<std::vector<char>> &&output_names);
  static std::vector<std::vector<char>> GetGraphOutputTensorNamesInChar();
  static std::map<std::vector<char>, std::vector<char>> GetConfigInfo(const std::vector<char> &&section);
};
}  // namespace converter
}  // namespace mindspore

#endif  // MINDSPORE_LITE_INCLUDE_REGISTRY_CONVERTER_CONTEXT_H_
