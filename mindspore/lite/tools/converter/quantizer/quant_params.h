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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_PARAM_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_PARAM_H
#include <map>
#include <string>
#include <vector>
#include "schema/inner/model_generated.h"
namespace mindspore::lite::quant {
enum ActivationQuantizedMethod {
  MAX_MIN = 0,
  KL = 1,
  REMOVAL_OUTLIER = 2,
};

struct CommonQuantParam {
  schema::QuantType quant_type = schema::QuantType_QUANT_NONE;
  int bit_num = 8;
  int min_quant_weight_size = 0;
  int min_quant_weight_channel = 16;
};

struct MixedBitWeightQuantParam {
  double init_scale = 0.02;
};

struct FullQuantParam {
  ActivationQuantizedMethod activation_quant_method = MAX_MIN;
  bool bias_correction = true;
  int thread_num = 1;
};
}  // namespace mindspore::lite::quant

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_PARAM_H
