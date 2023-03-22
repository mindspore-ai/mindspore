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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_CONFIG_PARSER_QUANT_PARAM_PARSER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_CONFIG_PARSER_QUANT_PARAM_PARSER_H_
#include <string>
#include "tools/converter/config_parser/config_file_parser.h"
#include "tools/converter/quantizer/quant_params.h"
namespace mindspore {
namespace lite {
class QuantParamParser {
 public:
  static int ParseCommonQuant(const CommonQuantString &common_quant_string, quant::CommonQuantParam *common_quant);
  static int ParseMixedBitWeightQuant(const MixedBitWeightQuantString &mixed_bit_weight_quant_string,
                                      quant::MixedBitWeightQuantParam *mixed_bit_weight_quant);
  static int ParseFullQuant(const FullQuantString &full_quant_string, quant::FullQuantParam *full_quant);
  static int ParseWeightQuant(const WeightQuantString &weight_quant_string, quant::WeightQuantParam *weight_quant);
  static int ParseTransformQuant(const TransformQuantString &transform_quant_string,
                                 quant::TransformQuantParam *transform_quant);

 private:
  static int ParseQuantType(const std::string &quant_type_str, quant::QuantType *quant_type);
  static int ParseTargetDevice(const std::string &target_device_str, quant::TargetDevice *target_device);

  static int ParseActivationQuantizedMethod(const std::string &activation_quant_method_str,
                                            quant::ActivationQuantizedMethod *activation_quant_method);
  static int ParseFilter(const CommonQuantString &common_quant_string, quant::CommonQuantParam *common_quant);
  static int ParseBitNum(const CommonQuantString &common_quant_string, quant::CommonQuantParam *common_quant);
  static int ParseEnableEncode(const CommonQuantString &common_quant_string, quant::CommonQuantParam *common_quant);
  static int ParseExportPrecisionMode(const std::string &precision_modeL_str, quant::PrecisionMode *precision_mode);
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_CONFIG_PARSER_QUANT_PARAM_PARSER_H_
