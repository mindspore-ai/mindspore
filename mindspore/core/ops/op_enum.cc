/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "ops/op_enum.h"

#include <algorithm>
#include <utility>

#include "mindapi/base/types.h"
#include "utils/check_convert_utils.h"
#include "mindapi/base/format.h"

namespace mindspore {
namespace ops {

namespace {
using StrToEnumMap = std::unordered_map<std::string, int64_t>;

class RegStringToEnumHelper {
 public:
  template <typename T>
  std::string AddValues(T &&string_to_enum) {
    for (const auto &kv : string_to_enum) {
      if (string_to_enum_.find(kv.first) != string_to_enum_.end()) {
        MS_LOG_EXCEPTION << kv.first << " has been registered!";
      }
    }
    string_to_enum_.merge(std::move(string_to_enum));
    return "";
  }

  const StrToEnumMap &GetValues() { return string_to_enum_; }

 private:
  StrToEnumMap string_to_enum_;
};
RegStringToEnumHelper reg_string_to_enum_helper;

#define REG_STRING_TO_ENUM(enum_type, ...) \
  const auto op_enum_##enum_type = reg_string_to_enum_helper.AddValues(__VA_ARGS__);

// Convert to uppercase uniformly
inline std::string StrToUpper(const std::string &str) {
  auto res = str;
  for (auto &c : res) {
    c = std::toupper(c);
  }
  return res;
}

// Format
inline std::unordered_map<std::string, int64_t> GetStringToFormatMap() {
  const auto &names = GetFormatNames();
  std::unordered_map<std::string, int64_t> map{{"DEFAULT_FORMAT", static_cast<int64_t>(Format::DEFAULT_FORMAT)}};
  for (size_t i = 0; i < names.size(); ++i) {
    map[StrToUpper(names[i])] = static_cast<int64_t>(i);
  }
  return map;
}
REG_STRING_TO_ENUM(format, GetStringToFormatMap())

// PadMode
StrToEnumMap StrToPadModeMap = {{"PAD", PadMode::PAD}, {"SAME", PadMode::SAME}, {"VALID", PadMode::VALID}};
REG_STRING_TO_ENUM(pad_mode, StrToPadModeMap)

// Reduction
StrToEnumMap StrToReductionMap = {
  {"SUM", Reduction::REDUCTION_SUM}, {"MEAN", Reduction::MEAN}, {"NONE", Reduction::NONE}};
REG_STRING_TO_ENUM(reduction, StrToReductionMap)

// Activation
StrToEnumMap StrToActivationMap = {{"NO_ACTIVATION", ActivationType::NO_ACTIVATION},
                                   {"RELU", ActivationType::RELU},
                                   {"SIGMOID", ActivationType::SIGMOID},
                                   {"RELU6", ActivationType::RELU6},
                                   {"ELU", ActivationType::ELU},
                                   {"LEAKY_RELU", ActivationType::LEAKY_RELU},
                                   {"ABS", ActivationType::ABS},
                                   {"RELU1", ActivationType::RELU1},
                                   {"SOFTSIGN", ActivationType::SOFTSIGN},
                                   {"SOFTPLUS", ActivationType::SOFTPLUS},
                                   {"TANH", ActivationType::TANH},
                                   {"SELU", ActivationType::SELU},
                                   {"HSWISH", ActivationType::HSWISH},
                                   {"HSIGMOID", ActivationType::HSIGMOID},
                                   {"THRESHOLDRELU", ActivationType::THRESHOLDRELU},
                                   {"LINEAR", ActivationType::LINEAR},
                                   {"HARD_TANH", ActivationType::HARD_TANH},
                                   {"SIGN", ActivationType::SIGN},
                                   {"SWISH", ActivationType::SWISH},
                                   {"GELU", ActivationType::GELU},
                                   {"GLU", ActivationType::GLU},
                                   {"UNKNOWN", ActivationType::UNKNOWN}};
REG_STRING_TO_ENUM(activation, StrToActivationMap)

// GateOrder
REG_STRING_TO_ENUM(gate_order, StrToEnumMap{{"RZH", GateOrderMode::RZH}, {"ZRH", GateOrderMode::ZRH}})

// CoordinateTransformationMode
StrToEnumMap StrToCoordinateTransformationModeMap = {{"ASYMMETRIC", CoordinateTransformMode::ASYMMETRIC},
                                                     {"ALIGN_CORNERS", CoordinateTransformMode::ALIGN_CORNERS},
                                                     {"HALF_PIXEL", CoordinateTransformMode::HALF_PIXEL},
                                                     {"CROP_AND_RESIZE", CoordinateTransformMode::CROP_AND_RESIZE}};
REG_STRING_TO_ENUM(coordinate_transformation_mode, StrToCoordinateTransformationModeMap)

// PaddingMode
StrToEnumMap StrToPaddingModeMap = {{"CONSTANT", PaddingMode::CONSTANT},
                                    {"REFLECT", PaddingMode::REFLECT},
                                    {"SYMMETRIC", PaddingMode::SYMMETRIC},
                                    {"MODE_RESERVED", PaddingMode::MODE_RESERVED}};
REG_STRING_TO_ENUM(padding_mode, StrToPaddingModeMap)

// Direction
REG_STRING_TO_ENUM(direction, StrToEnumMap{{"UNIDIRECTIONAL", Direction::UNIDIRECTIONAL}})

// CellType
REG_STRING_TO_ENUM(cell_type, StrToEnumMap{{"LSTM", CellType::LSTM}})

// Group
REG_STRING_TO_ENUM(group, StrToEnumMap{{"SYNC_BN_GROUP0", Group::SYNC_BN_GROUP0}})

// InterpolationMode
REG_STRING_TO_ENUM(interpolation_mode,
                   StrToEnumMap{{"BILINEAR", InterpolationMode::BILINEAR}, {"NEAREST", InterpolationMode::NEAREST}})

// NormMode
StrToEnumMap StrToNormModeMap = {
  {"BACKWARD", NormMode::BACKWARD}, {"FORWARD", NormMode::FORWARD}, {"ORTHO", NormMode::ORTHO}};
REG_STRING_TO_ENUM(norm_mode, StrToNormModeMap)

// GridSamplerPaddingMode
StrToEnumMap StrToGridSamplerPaddingMode = {{"ZEROS", GridSamplerPaddingMode::ZEROS},
                                            {"BORDER", GridSamplerPaddingMode::BORDER},
                                            {"REFLECTION", GridSamplerPaddingMode::REFLECTION}};
REG_STRING_TO_ENUM(grid_sampler_padding_mode, StrToGridSamplerPaddingMode)

// KVCacheAlignMode
REG_STRING_TO_ENUM(k_v_cache_align_mode,
                   StrToEnumMap{{"LEFT", KVCacheAlignMode::LEFT}, {"RIGHT", KVCacheAlignMode::RIGHT}})

}  // namespace

int64_t StringToEnumImpl(const std::string &enum_string) {
  const auto &string_to_enum_map = reg_string_to_enum_helper.GetValues();
  const auto enum_val_iter = string_to_enum_map.find(StrToUpper(enum_string));
  if (enum_val_iter == string_to_enum_map.end()) {
    MS_LOG_EXCEPTION << "Can not find '" << enum_string << "', please add it";
  }
  return enum_val_iter->second;
}
}  // namespace ops
}  // namespace mindspore
