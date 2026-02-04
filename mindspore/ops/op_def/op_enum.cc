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
#include "include/op_enum.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <utility>

#include "mindapi/base/types.h"
#include "utils/check_convert_utils.h"
#include "mindapi/base/format.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {

namespace {
using StrToEnumMap = std::unordered_map<std::string, int64_t>;

class RegStringToEnumHelper {
 public:
  template <typename T>
  std::string AddValues(T &&string_to_enum, const std::string &key = "") {
    auto &string_to_enum_target = string_to_enum_memory_[key];
    for (const auto &kv : string_to_enum) {
      if (string_to_enum_target.find(kv.first) != string_to_enum_target.end()) {
        MS_LOG(EXCEPTION) << kv.first << " has been registered!";
      }
    }
    string_to_enum_target.merge(std::move(string_to_enum));
    return "";
  }

  const StrToEnumMap &GetValues(const std::string &key = "") {
    auto it = string_to_enum_memory_.find(key);
    if (it != string_to_enum_memory_.end()) {
      return it->second;
    }
    return string_to_enum_memory_[""];
  }

 private:
  std::unordered_map<std::string, StrToEnumMap> string_to_enum_memory_;
};
RegStringToEnumHelper reg_string_to_enum_helper;

#define REG_STRING_TO_ENUM_COMMON(enum_type, ...) \
  const auto op_enum_##enum_type = reg_string_to_enum_helper.AddValues(__VA_ARGS__);

#define REG_STRING_TO_ENUM_SPECIAL(enum_type, ...) \
  const auto op_enum_##enum_type = reg_string_to_enum_helper.AddValues(__VA_ARGS__, #enum_type);

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
REG_STRING_TO_ENUM_COMMON(format, GetStringToFormatMap())

// RoundingMode
StrToEnumMap StrToRoundingModeMap = {{"ROUND", RoundingMode::ROUND},
                                     {"TRUNC", RoundingMode::TRUNC},
                                     {"FLOOR", RoundingMode::FLOOR},
                                     {"CEIL", RoundingMode::CEIL}};
REG_STRING_TO_ENUM_COMMON(rounding_mode, StrToRoundingModeMap)

// PadMode
StrToEnumMap StrToPadModeMap = {
  {"PAD", PadMode::PAD}, {"SAME", PadMode::SAME}, {"VALID", PadMode::VALID}, {"FULL", PadMode::FULL}};
REG_STRING_TO_ENUM_COMMON(pad_mode, StrToPadModeMap)

// Reduction
StrToEnumMap StrToReductionMap = {{"SUM", Reduction::REDUCTION_SUM},
                                  {"MEAN", Reduction::MEAN},
                                  {"NONE", Reduction::NONE},
                                  {"UPDATE", Reduction::UPDATE}};
REG_STRING_TO_ENUM_COMMON(reduction, StrToReductionMap)

// Reduce
StrToEnumMap StrToReduceMap = {{"NONE", Reduce::REDUCE_NONE},
                               {"ADD", Reduce::ADD},
                               {"MULTIPLY", Reduce::MULTIPLY},
                               {"UPDATE", Reduce::REDUCE_UPDATE}};
REG_STRING_TO_ENUM_SPECIAL(reduce, StrToReduceMap)

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
                                   {"UNKNOWN", ActivationType::UNKNOWN},
                                   {"FASTGELU", ActivationType::FASTGELU},
                                   {"SILU", ActivationType::SILU},
                                   {"GEGLU", ActivationType::GEGLU},
                                   {"SWIGLU", ActivationType::SWIGLU},
                                   {"REGLU", ActivationType::REGLU}};
REG_STRING_TO_ENUM_COMMON(activation, StrToActivationMap)

// GateOrder
REG_STRING_TO_ENUM_COMMON(gate_order, StrToEnumMap{{"RZH", GateOrderMode::RZH}, {"ZRH", GateOrderMode::ZRH}})

// CoordinateTransformationMode
StrToEnumMap StrToCoordinateTransformationModeMap = {{"ASYMMETRIC", CoordinateTransformMode::ASYMMETRIC},
                                                     {"ALIGN_CORNERS", CoordinateTransformMode::ALIGN_CORNERS},
                                                     {"HALF_PIXEL", CoordinateTransformMode::HALF_PIXEL},
                                                     {"CROP_AND_RESIZE", CoordinateTransformMode::CROP_AND_RESIZE}};
REG_STRING_TO_ENUM_COMMON(coordinate_transformation_mode, StrToCoordinateTransformationModeMap)

// PaddingMode
StrToEnumMap StrToPaddingModeMap = {{"CONSTANT", PaddingMode::CONSTANT},
                                    {"REFLECT", PaddingMode::REFLECT},
                                    {"SYMMETRIC", PaddingMode::SYMMETRIC},
                                    {"MODE_RESERVED", PaddingMode::MODE_RESERVED}};
REG_STRING_TO_ENUM_COMMON(padding_mode, StrToPaddingModeMap)

// Direction
REG_STRING_TO_ENUM_COMMON(direction, StrToEnumMap{{"UNIDIRECTIONAL", Direction::UNIDIRECTIONAL}})

// CellType
REG_STRING_TO_ENUM_COMMON(cell_type, StrToEnumMap{{"LSTM", CellType::CELL_TYPE_LSTM}})

// Group
REG_STRING_TO_ENUM_COMMON(group, StrToEnumMap{{"SYNC_BN_GROUP0", Group::SYNC_BN_GROUP0}})

// InterpolationMode
REG_STRING_TO_ENUM_COMMON(interpolation_mode, StrToEnumMap{{"BILINEAR", InterpolationMode::BILINEAR},
                                                           {"NEAREST", InterpolationMode::NEAREST},
                                                           {"BICUBIC2D", InterpolationMode::BICUBIC}})

// indexing
REG_STRING_TO_ENUM_COMMON(indexing, StrToEnumMap{{"IJ", Indexing::IJ}, {"XY", Indexing::XY}})

// NormMode
StrToEnumMap StrToNormModeMap = {
  {"BACKWARD", NormMode::BACKWARD}, {"FORWARD", NormMode::FORWARD}, {"ORTHO", NormMode::ORTHO}};
REG_STRING_TO_ENUM_COMMON(norm_mode, StrToNormModeMap)

// GridSamplerPaddingMode
StrToEnumMap StrToGridSamplerPaddingMode = {{"ZEROS", GridSamplerPaddingMode::ZEROS},
                                            {"BORDER", GridSamplerPaddingMode::BORDER},
                                            {"REFLECTION", GridSamplerPaddingMode::REFLECTION}};
REG_STRING_TO_ENUM_COMMON(grid_sampler_padding_mode, StrToGridSamplerPaddingMode)

// KVCacheAlignMode
REG_STRING_TO_ENUM_COMMON(k_v_cache_align_mode,
                          StrToEnumMap{{"LEFT", KVCacheAlignMode::LEFT}, {"RIGHT", KVCacheAlignMode::RIGHT}})

// FAS input_layout
REG_STRING_TO_ENUM_COMMON(fas_input_layout_mode, StrToEnumMap{{"BSH", FASInputLayoutMode::BSH},
                                                              {"BNSD", FASInputLayoutMode::BNSD},
                                                              {"SBH", FASInputLayoutMode::SBH},
                                                              {"BSND", FASInputLayoutMode::BSND},
                                                              {"TND", FASInputLayoutMode::TND},
                                                              {"TH", FASInputLayoutMode::TH},
                                                              {"NSD", FASInputLayoutMode::NSD},
                                                              {"SH", FASInputLayoutMode::SH},
                                                              {"BNSD_BSND", FASInputLayoutMode::BNSD_BSND}})

// DriverName
REG_STRING_TO_ENUM_COMMON(driver_name, StrToEnumMap{{"GELS", DriverName::GELS},
                                                    {"GELSY", DriverName::GELSY},
                                                    {"GELSD", DriverName::GELSD},
                                                    {"GELSS", DriverName::GELSS}})

// PagedAttentionKVCacheQuantMode
REG_STRING_TO_ENUM_COMMON(kv_cache_quant_mode, StrToEnumMap{{"DEFAULT", PagedAttentionKVCacheQuantMode::DEFAULT},
                                                            {"PERTOKEN", PagedAttentionKVCacheQuantMode::PERTOKEN}})

// PagedAttentionMaskMode
REG_STRING_TO_ENUM_COMMON(mask_mode, StrToEnumMap{{"MASK_DEFAULT", PagedAttentionMaskMode::MASK_DEFAULT},
                                                  {"TRAPEZOIDAL", PagedAttentionMaskMode::TRAPEZOIDAL},
                                                  {"MASK_NONE", MLAMode::MASK_NONE},
                                                  {"MASK_NORM", MLAMode::MASK_NORM},
                                                  {"MASK_ALIBI", MLAMode::MASK_ALIBI},
                                                  {"MASK_SPEC", MLAMode::MASK_SPEC},
                                                  {"MASK_FREE", MLAMode::MASK_FREE}})
// CompareOp
REG_STRING_TO_ENUM_COMMON(compare_op,
                          StrToEnumMap{
                            {"EQ", CompareOp::EQUAL}, {"GT", CompareOp::GREATER_THAN}, {"LT", CompareOp::LESS_THAN}})

// SignalOp
REG_STRING_TO_ENUM_COMMON(signal_op, StrToEnumMap{{"SET", SignalOpMode::SET}, {"ADD", SignalOpMode::ADD}})

// ErrorMode
REG_STRING_TO_ENUM_SPECIAL(error_mode, StrToEnumMap{{"CYCLE", ErrorMode::CYCLE}, {"SPECIFIC", ErrorMode::SPECIFIC}});

// FlipMode
REG_STRING_TO_ENUM_SPECIAL(flip_mode, StrToEnumMap{{"BITFLIP", FlipMode::BITFLIP},
                                                   {"BITFLIP_DESIGNED", FlipMode::BITFLIP_DESIGNED},
                                                   {"MULTIPLY", FlipMode::MULTIPLY},
                                                   {"MULTIPLY_MAX", FlipMode::MULTIPLY_MAX}});

// LinalgQrMode
REG_STRING_TO_ENUM_SPECIAL(mode, StrToEnumMap{{"REDUCED", LinalgQrMode::REDUCED},
                                              {"COMPLETE", LinalgQrMode::COMPLETE},
                                              {"R", LinalgQrMode::R}});

// DequantSwigluQuant quant_mode
REG_STRING_TO_ENUM_SPECIAL(quant_mode, StrToEnumMap{{"STATIC", QuantMode::STATIC}, {"DYNAMIC", QuantMode::DYNAMIC}});

// FlipMode
REG_STRING_TO_ENUM_SPECIAL(approximate, StrToEnumMap{{"NONE", Approximate::NONE}, {"TANH", Approximate::TANH}});

// Device
StrToEnumMap StrToDeviceMap = {{"Ascend", Device::DEVICE_ASCEND},
                               {"npu", Device::DEVICE_NPU_LOWER},
                               {"CPU", Device::DEVICE_CPU},
                               {"cpu", Device::DEVICE_CPU_LOWER}};
REG_STRING_TO_ENUM_SPECIAL(device, StrToDeviceMap);
}  // namespace

int64_t StringToEnumImpl(const std::string &op_name, const std::string &arg_name, const std::string &enum_string) {
  const auto &string_to_enum_map = reg_string_to_enum_helper.GetValues(arg_name);
  const auto enum_val_iter = string_to_enum_map.find(arg_name == "device" ? enum_string : StrToUpper(enum_string));
  if (enum_val_iter == string_to_enum_map.end()) {
    MS_EXCEPTION(ValueError) << "Failed to convert the value \"" << enum_string << "\" of input '" << arg_name
                             << "' of '" << op_name << "' to enum.";
  }
  return enum_val_iter->second;
}
}  // namespace ops
}  // namespace mindspore
