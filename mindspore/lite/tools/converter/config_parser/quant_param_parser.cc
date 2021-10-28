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
int QuantParamParser::ParseCommonQuant(const CommonQuantString &common_quant_string,
                                       quant::CommonQuantParam *common_quant) {
  if (!common_quant_string.quant_type.empty()) {
    auto ret = ParseQuantType(common_quant_string.quant_type, &common_quant->quant_type);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Parse quant_type failed.";
      return ret;
    }
  }

  if (!common_quant_string.bit_num.empty() && !ConvertIntNum(common_quant_string.bit_num, &common_quant->bit_num)) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: bit_num should be a valid number.";
    return RET_INPUT_PARAM_INVALID;
  }
  if (common_quant->quant_type == schema::QuantType_QUANT_WEIGHT) {
    if (common_quant->bit_num < 0 || common_quant->bit_num > kQuantBitNumInt16) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: bit_num should be [0,16].";
      return RET_INPUT_PARAM_INVALID;
    }
  } else if (common_quant->quant_type == schema::QuantType_QUANT_ALL) {
    if (common_quant->bit_num <= 0 || common_quant->bit_num > kQuantBitNumInt8) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: bit_num should be [1,8].";
      return RET_INPUT_PARAM_INVALID;
    }
  }
  if (!common_quant_string.min_quant_weight_size.empty() &&
      !ConvertIntNum(common_quant_string.min_quant_weight_size, &common_quant->min_quant_weight_size)) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: min_quant_weight_size should be a valid number.";
    return RET_INPUT_PARAM_INVALID;
  }
  if (!common_quant_string.min_quant_weight_channel.empty() &&
      !ConvertIntNum(common_quant_string.min_quant_weight_channel, &common_quant->min_quant_weight_channel)) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: min_quant_weight_channel should be a valid number.";
    return RET_INPUT_PARAM_INVALID;
  }
  if (common_quant->min_quant_weight_size < kMinSize || common_quant->min_quant_weight_size > kMaxSize) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: min_quant_weight_size should in [0,65535]." << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }

  if (common_quant->min_quant_weight_channel < kMinSize || common_quant->min_quant_weight_channel > kMaxSize) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: min_quant_weight_channel should in [0,65535]." << std::endl;
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

int QuantParamParser::ParseMixedBitWeightQuant(const MixedBitWeightQuantString &mixed_bit_weight_quant_string,
                                               quant::MixedBitWeightQuantParam *mixed_bit_weight_quant) {
  if (mixed_bit_weight_quant_string.init_scale.empty()) {
    return RET_OK;
  }
  if (!ConvertDoubleNum(mixed_bit_weight_quant_string.init_scale, &mixed_bit_weight_quant->init_scale)) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: init_scale should be a valid number.";
    return RET_INPUT_PARAM_INVALID;
  }
  if (mixed_bit_weight_quant->init_scale <= 0 || mixed_bit_weight_quant->init_scale >= 1) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: init_scale should at (0,1)";
    return RET_INPUT_PARAM_INVALID;
  }
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
  return RET_OK;
}

int QuantParamParser::ParseQuantType(const std::string &quant_type_str, schema::QuantType *quant_type) {
  if (quant_type_str == "WEIGHT_QUANT") {
    (*quant_type) = schema::QuantType_QUANT_WEIGHT;
    return RET_OK;
  } else if (quant_type_str == "FULL_QUANT") {
    (*quant_type) = schema::QuantType_QUANT_ALL;
    return RET_OK;
  } else if (quant_type_str.empty()) {
    (*quant_type) = schema::QuantType_QUANT_NONE;
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "INPUT ILLEGAL: quant_type must be WEIGHT_QUANT|FULL_QUANT.";
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
}  // namespace lite
}  // namespace mindspore
