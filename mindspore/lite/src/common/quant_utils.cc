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

#include "schema/inner/model_generated.h"
#include "src/common/quant_utils.h"
#include "src/lite_kernel.h"

namespace mindspore {
namespace lite {
STATUS GetMaxMinPerChannel(int channels, int one_filter_size, int i, int elem_count, const float *raw_datas,
                           bool channel_at_first, float *desired_max, float *desired_min) {
  float min = FLT_MAX;
  float max = -FLT_MAX;
  // find min and max
  for (int j = 0; j < one_filter_size; j++) {
    auto index = j + i * one_filter_size;
    if (!channel_at_first) {
      index = j * channels + i;
    }
    if (index >= elem_count) {
      MS_LOG(ERROR) << "over flow!";
      return RET_ERROR;
    }
    min = std::min(min, raw_datas[index]);
    max = std::max(max, raw_datas[index]);
  }
  *desired_max = max;
  *desired_min = min;
  return RET_OK;
}

STATUS CalQuantizationParams(schema::QuantParamT *quant_param, double real_min, double real_max, bool narrow_range,
                             int quant_max, int quant_min, int num_bits) {
  MS_ASSERT(quant_param != nullptr);
  if (real_min > 0.0f) {
    MS_LOG(DEBUG) << "min " << real_min << " is bigger then 0, set to 0, this may course low precision";
    real_min = 0.0f;
  }
  if (real_max < 0.0f) {
    MS_LOG(DEBUG) << "real_max " << real_max << " is smaller than 0, set to 0, this may course low precision";
    real_max = 0.0f;
  }
  if (real_min > real_max) {
    MS_LOG(ERROR) << "cal error while min" << real_min << ">" << real_max;
    return RET_PARAM_INVALID;
  }
  if (real_min == real_max) {
    if (real_min != 0.0f) {
      MS_LOG(ERROR) << "min and max should both be zero if they are equal to each other";
      return RET_ERROR;
    }
    MS_LOG(WARNING) << "The maximum and minimum values are equal to 0.";
    quant_param->inited = true;
    quant_param->min = real_min;
    quant_param->max = real_max;
    quant_param->scale = 1;
    quant_param->zeroPoint = 0;
    quant_param->narrowRange = narrow_range;
    quant_param->numBits = num_bits;
    return RET_OK;
  }

  auto quantMinFloat = static_cast<double>(quant_min);
  auto quantMaxFloat = static_cast<double>(quant_max);
  if (fabs(quantMaxFloat - quantMinFloat) <= 0.0f) {
    MS_LOG(ERROR) << "divisor cannot be 0";
    return RET_ERROR;
  }
  double scale = (real_max - real_min) / (quantMaxFloat - quantMinFloat);
  if (fabs(scale) <= 0.0f) {
    MS_LOG(ERROR) << "divisor 'scale' cannot be 0";
    return RET_ERROR;
  }
  const double zeroPointFromMin = quantMinFloat - real_min / scale;
  int zeroPoint = static_cast<int32_t>(std::round(zeroPointFromMin));

  // The zero point should always be in the range of quantized value,
  // [qmin, qmax].
  MS_ASSERT(zeroPoint >= quant_min);
  MS_ASSERT(zeroPoint <= quant_max);
  quant_param->inited = true;
  quant_param->min = real_min;
  quant_param->max = real_max;
  quant_param->scale = scale;
  quant_param->zeroPoint = zeroPoint;
  quant_param->narrowRange = narrow_range;
  quant_param->numBits = num_bits;

  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
