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

#include "tools/converter/config_parser/quant_param_parser.h"
#include <vector>
#include "src/common/log_adapter.h"
#include "mindspore/lite/tools/common/string_util.h"
#include "include/errorcode.h"

namespace mindspore {
namespace lite {
namespace {
constexpr int kQuantBitNumInt16 = 16;
constexpr int kQuantBitNumInt8 = 8;
constexpr int kMinSize = 0;
constexpr int kMaxSize = 65535;
}  // namespace
int QuantParamParser::ParseFilter(const CommonQuantString &common_quant_string, quant::CommonQuantParam *common_quant) {
  MS_ASSERT(common_quant != nullptr);
  if (!common_quant_string.min_quant_weight_size.empty()) {
    if (!ConvertIntNum(common_quant_string.min_quant_weight_size, &common_quant->min_quant_weight_size)) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: min_quant_weight_size should be a valid number.";
      return RET_INPUT_PARAM_INVALID;
    }
    if (common_quant->min_quant_weight_size < kMinSize || common_quant->min_quant_weight_size > kMaxSize) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: min_quant_weight_size should in [0,65535]." << std::endl;
      return RET_INPUT_PARAM_INVALID;
    }
  }
  if (!common_quant_string.min_quant_weight_channel.empty()) {
    if (!ConvertIntNum(common_quant_string.min_quant_weight_channel, &common_quant->min_quant_weight_channel)) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: min_quant_weight_channel should be a valid number.";
      return RET_INPUT_PARAM_INVALID;
    }

    if (common_quant->min_quant_weight_channel < kMinSize || common_quant->min_quant_weight_channel > kMaxSize) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: min_quant_weight_channel should be in the range [0,65535]." << std::endl;
      return RET_INPUT_PARAM_INVALID;
    }
  }
  if (!common_quant_string.skip_quant_node.empty()) {
    std::vector<std::string> nodes = SplitStringToVector(common_quant_string.skip_quant_node, ',');
    for (const auto &node : nodes) {
      common_quant->skip_quant_node.insert(node);
    }
  }
  return RET_OK;
}

int QuantParamParser::ParseBitNum(const CommonQuantString &common_quant_string, quant::CommonQuantParam *common_quant) {
  if (!common_quant_string.bit_num.empty() && !ConvertIntNum(common_quant_string.bit_num, &common_quant->bit_num)) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: bit_num should be a valid number.";
    return RET_INPUT_PARAM_INVALID;
  }
  if (common_quant->quant_type == quant::QUANT_WEIGHT) {
    if (common_quant->bit_num < 0 || common_quant->bit_num > kQuantBitNumInt16) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: bit_num should be [0,16].";
      return RET_INPUT_PARAM_INVALID;
    }
  } else if (common_quant->quant_type == quant::QUANT_ALL) {
    if (common_quant->bit_num != kQuantBitNumInt8) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: bit_num should be 8.";
      return RET_INPUT_PARAM_INVALID;
    }
  } else if (common_quant->quant_type == quant::QUANT_DYNAMIC) {
    if (common_quant->bit_num != kQuantBitNumInt8) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: bit_num should be 8.";
      return RET_INPUT_PARAM_INVALID;
    }
  }
  return RET_OK;
}

int QuantParamParser::ParseEnableEncode(const CommonQuantString &common_quant_string,
                                        quant::CommonQuantParam *common_quant) {
  if (!common_quant_string.enable_encode.empty() &&
      !ConvertBool(common_quant_string.enable_encode, &common_quant->enable_encode)) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: enable_encode should be true or false.";
    return RET_INPUT_PARAM_INVALID;
  }
  if (common_quant->quant_type == quant::QUANT_WEIGHT &&
      (common_quant->bit_num != kQuantBitNumInt8 && common_quant->bit_num != kQuantBitNumInt16)) {
    if (!common_quant->enable_encode) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: enable_encode should be true when parameter bit_num belongs to [0,7] or [9,15].";
      return RET_INPUT_PARAM_INVALID;
    }
  }
  return RET_OK;
}

int QuantParamParser::ParseCommonQuant(const CommonQuantString &common_quant_string,
                                       quant::CommonQuantParam *common_quant) {
  MS_ASSERT(common_quant != nullptr);
  if (!common_quant_string.quant_type.empty()) {
    auto ret = ParseQuantType(common_quant_string.quant_type, &common_quant->quant_type);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Parse quant_type failed.";
      return ret;
    }
  }

  auto ret = ParseBitNum(common_quant_string, common_quant);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Parse bit num failed.";
    return ret;
  }

  ret = ParseEnableEncode(common_quant_string, common_quant);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Parse enable encode failed.";
    return ret;
  }

  ret = ParseFilter(common_quant_string, common_quant);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Parse filter failed.";
    return ret;
  }

  common_quant->debug_info_save_path = common_quant_string.debug_info_save_path;
  if (!common_quant->debug_info_save_path.empty()) {
    common_quant->is_debug = true;
  }
  return RET_OK;
}

int QuantParamParser::ParseMixedBitWeightQuant(const MixedBitWeightQuantString &mixed_bit_weight_quant_string,
                                               quant::MixedBitWeightQuantParam *mixed_bit_weight_quant) {
  if (!mixed_bit_weight_quant_string.init_scale.empty() &&
      !ConvertDoubleNum(mixed_bit_weight_quant_string.init_scale, &mixed_bit_weight_quant->init_scale)) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: init_scale should be a valid number.";
    return RET_INPUT_PARAM_INVALID;
  }
  if (mixed_bit_weight_quant->init_scale <= 0 || mixed_bit_weight_quant->init_scale >= 1) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: init_scale should at (0,1)";
    return RET_INPUT_PARAM_INVALID;
  }
  if (!mixed_bit_weight_quant_string.use_cv_data.empty() &&
      !ConvertBool(mixed_bit_weight_quant_string.use_cv_data, &mixed_bit_weight_quant->use_cv_data)) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: use_cv_data should be true or false.";
    return RET_INPUT_PARAM_INVALID;
  }
  if (!mixed_bit_weight_quant_string.max_iterations.empty()) {
    if (!ConvertIntNum(mixed_bit_weight_quant_string.max_iterations, &mixed_bit_weight_quant->max_iterations)) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: max_iterations should be a valid number.";
      return RET_INPUT_PARAM_INVALID;
    }
    if (mixed_bit_weight_quant->max_iterations < quant::kMinIterations ||
        mixed_bit_weight_quant->max_iterations > kMaxSize) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: max_iterations should be in the range [40,65535]." << std::endl;
      return RET_INPUT_PARAM_INVALID;
    }
  }

  if (!mixed_bit_weight_quant_string.auto_tune.empty() &&
      !ConvertBool(mixed_bit_weight_quant_string.auto_tune, &mixed_bit_weight_quant->auto_tune)) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: auto_tune should be true or false.";
    return RET_INPUT_PARAM_INVALID;
  }

  // this is required only for model larger than 2G
  mixed_bit_weight_quant->workspace = mixed_bit_weight_quant_string.workspace;
  return RET_OK;
}

int QuantParamParser::ParseFullQuant(const FullQuantString &full_quant_string, quant::FullQuantParam *full_quant) {
  if (!full_quant_string.activation_quant_method.empty() &&
      ParseActivationQuantizedMethod(full_quant_string.activation_quant_method, &full_quant->activation_quant_method) !=
        RET_OK) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: Parse activation_quant_method failed.";
    return RET_INPUT_PARAM_INVALID;
  }
  if (!full_quant_string.bias_correction.empty() &&
      !ConvertBool(full_quant_string.bias_correction, &full_quant->bias_correction)) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: bias_correction should be true or false.";
    return RET_INPUT_PARAM_INVALID;
  }
  if (!full_quant_string.target_device.empty()) {
    auto ret = ParseTargetDevice(full_quant_string.target_device, &full_quant->target_device);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Parse device failed.";
      return ret;
    }
  }
  if (!full_quant_string.per_channel.empty() && !ConvertBool(full_quant_string.per_channel, &full_quant->per_channel)) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: per_channel should be true or false.";
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

int QuantParamParser::ParseQuantType(const std::string &quant_type_str, quant::QuantType *quant_type) {
  if (quant_type_str == "WEIGHT_QUANT") {
    (*quant_type) = quant::QUANT_WEIGHT;
    return RET_OK;
  } else if (quant_type_str == "FULL_QUANT") {
    (*quant_type) = quant::QUANT_ALL;
    return RET_OK;
  } else if (quant_type_str == "DYNAMIC_QUANT") {
    (*quant_type) = quant::QUANT_DYNAMIC;
    return RET_OK;
  } else if (quant_type_str.empty()) {
    (*quant_type) = quant::QUANT_NONE;
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "INPUT ILLEGAL: quant_type must be WEIGHT_QUANT|FULL_QUANT|DYNAMIC_QUANT.";
    return RET_INPUT_PARAM_INVALID;
  }
}

int QuantParamParser::ParseTargetDevice(const std::string &target_device_str, quant::TargetDevice *target_device) {
  if (target_device_str == "KIRIN") {
    (*target_device) = quant::KIRIN;
    return RET_OK;
  } else if (target_device_str == "NVGPU") {
    (*target_device) = quant::NVGPU;
    return RET_OK;
  } else if (target_device_str == "DSP") {
    (*target_device) = quant::DSP;
    return RET_OK;
  } else if (target_device_str == "ASCEND") {
    (*target_device) = quant::ASCEND;
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "INPUT ILLEGAL: target_device must be KIRIN|NVGPU|DSP|ASCEND.";
    return RET_INPUT_PARAM_INVALID;
  }
}

int QuantParamParser::ParseActivationQuantizedMethod(const std::string &activation_quant_method_str,
                                                     quant::ActivationQuantizedMethod *activation_quant_method) {
  if (activation_quant_method_str == "MAX_MIN") {
    (*activation_quant_method) = quant::MAX_MIN;
    return RET_OK;
  } else if (activation_quant_method_str == "KL") {
    (*activation_quant_method) = quant::KL;
    return RET_OK;
  } else if (activation_quant_method_str == "REMOVAL_OUTLIER") {
    (*activation_quant_method) = quant::REMOVAL_OUTLIER;
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "INPUT ILLEGAL: activation_quant_method must be MAX_MIN|KL|REMOVAL_OUTLIER.";
    return RET_INPUT_PARAM_INVALID;
  }
}

int QuantParamParser::ParseWeightQuant(const WeightQuantString &weight_quant_string,
                                       quant::WeightQuantParam *weight_quant) {
  if (!weight_quant_string.dequant_strategy.empty()) {
    if (weight_quant_string.dequant_strategy == "ON_THE_FLY") {
      weight_quant->dequant_strategy = quant::ON_THE_FLY;
    } else {
      MS_LOG(ERROR) << "INPUT ILLEGAL: dequant_strategy must be ON_THE_FLY.";
      return RET_INPUT_PARAM_INVALID;
    }
  }
  if (!weight_quant_string.update_mindir.empty() &&
      !ConvertBool(weight_quant_string.update_mindir, &weight_quant->update_mindir)) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: update_mindir should be true or false.";
    return RET_INPUT_PARAM_INVALID;
  }
  if (!weight_quant_string.max_segments.empty() &&
      !ConvertIntNum(weight_quant_string.max_segments, &weight_quant->max_segments)) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: decode_threads should be a number.";
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

int QuantParamParser::ParseExportPrecisionMode(const std::string &precision_modeL_str,
                                               quant::PrecisionMode *precision_mode) {
  if (precision_modeL_str == "QUANT") {
    (*precision_mode) = quant::QUANT;
    return RET_OK;
  } else if (precision_modeL_str == "FLOAT32") {
    (*precision_mode) = quant::FLOAT32;
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "INPUT ILLEGAL: export_precision_mode must be QUANT or FLOAT32.";
    return RET_INPUT_PARAM_INVALID;
  }
}

int QuantParamParser::ParseTransformQuant(const TransformQuantString &transform_quant_string,
                                          quant::TransformQuantParam *transform_quant) {
  if (!transform_quant_string.export_precision_mode.empty()) {
    auto ret = ParseExportPrecisionMode(transform_quant_string.export_precision_mode, &transform_quant->precision_mode);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Parse precision mode failed.";
      return ret;
    }
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
