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

#ifndef MINDSPORE_LITE_SRC_COMMON_QUANT_UTILS_H_
#define MINDSPORE_LITE_SRC_COMMON_QUANT_UTILS_H_

#include <float.h>
#include <cmath>
#include <climits>
#include <limits>
#include <algorithm>
#include <vector>
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "ir/dtype/type_id.h"

namespace mindspore {

namespace schema {
struct QuantParamT;
}

namespace lite {
const int RET_QUANT_CONTINUE = 2;
static constexpr double SCALE_THREASHOLD = 1e-38;

static constexpr int kPerTensor = 1;

inline int QuantMax(int bits, TypeId type) {
  if (type == kNumberTypeInt8) {
    return (1 << (bits - 1)) - 1;
  } else if (type == kNumberTypeUInt8) {
    return (1 << bits) - 1;
  }
  return 0;
}

inline int QuantMin(int bits, TypeId type) {
  if (type == kNumberTypeInt8) {
    return -(1 << (bits - 1));
  }
  return 0;
}

STATUS GetMaxMinPerChannel(int channels, int one_filter_size, int i, int elem_count, const float *raw_datas,
                           bool channel_at_first, float *desired_max, float *desired_min);

STATUS CalQuantizationParams(schema::QuantParamT *quant_param, double real_min, double real_max, bool narrow_range,
                             int quant_max, int quant_min, int num_bits);

template <typename T>
T QuantizeData(const float originData, const schema::QuantParamT *quantParam) {
  MS_ASSERT(quantParam != nullptr);
  MS_ASSERT(quantParam->inited);
  const auto scale = quantParam->scale;
  const auto zeroPoint = quantParam->zeroPoint;
  const auto numBit = quantParam->numBits;
  const auto narrowRange = quantParam->narrowRange;
  const int32_t quantMax = (1 << (unsigned int)(numBit - 1)) - 1;
  const int32_t quantMin = -1 * (1 << (unsigned int)(numBit - 1)) + (narrowRange ? 1 : 0);
  const double maxLimit = static_cast<float>(quantMax - zeroPoint) * scale;
  const double minLimit = static_cast<float>(quantMin - zeroPoint) * scale;

  return [maxLimit, minLimit, zeroPoint, scale, narrowRange, originData] {
    double tmp;
    if (originData > maxLimit) {
      tmp = maxLimit;
    } else if (originData < minLimit) {
      tmp = minLimit;
    } else {
      tmp = originData;
    }
    auto quantData = static_cast<T>(std::round(zeroPoint + tmp / scale));
    return quantData;
  }();
}

template <typename T>
T QuantizeData(float originData, const schema::QuantParamT *quantParam, int quant_max, int quant_min) {
  MS_ASSERT(quantParam != nullptr);
  MS_ASSERT(quantParam->inited);
  const auto scale = quantParam->scale;
  const int zeroPoint = quantParam->zeroPoint;
  const int maxLimit = quant_max;
  const int minLimit = quant_min;

  if (scale <= SCALE_THREASHOLD) {
    return 0;
  }

  return [maxLimit, minLimit, zeroPoint, scale, originData] {
    auto quant_data = std::round(originData / scale + zeroPoint);
    if (quant_data > maxLimit) {
      quant_data = maxLimit;
    } else if (quant_data < minLimit) {
      quant_data = minLimit;
    }
    return static_cast<T>(quant_data);
  }();
}

template <typename T>
STATUS DoPerLayerQuant(const float *raw_datas, size_t elem_count, std::vector<schema::QuantParamT> *quant_params,
                       const int &quant_max, const int &quant_min, const size_t &bit_num, const bool &k_means,
                       std::vector<T> *quant_datas) {
  float min = FLT_MAX;
  float max = -FLT_MIN;
  for (uint32_t i = 0; i < elem_count; i++) {
    min = std::min(min, raw_datas[i]);
    max = std::max(max, raw_datas[i]);
  }

  schema::QuantParamT quant_param;
  if (!k_means) {
    STATUS status = CalQuantizationParams(&quant_param, min, max, false, quant_max, quant_min, bit_num);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "CalQuantizationParams failed" << status;
      return status;
    }
  }
  quant_params->emplace_back(quant_param);
  // update data and datatype
  for (uint32_t i = 0; i < elem_count; i++) {
    float raw_data = raw_datas[i];
    if (!k_means) {
      auto quant_data = QuantizeData<T>(raw_data, &quant_param, quant_max, quant_min);
      (*quant_datas)[i] = quant_data;
    }
  }
  return RET_OK;
}

template <typename T>
STATUS DoPerChannelQuant(const float *raw_datas, size_t elem_count, const schema::QuantType &quant_type,
                         std::vector<schema::QuantParamT> *quant_params, const int &quant_max, const int &quant_min,
                         const size_t &bit_num, const bool &k_means, std::vector<T> *quant_datas, int channels,
                         bool channel_at_first = true) {
  static const int quant_param_size = 32 * 8;
  std::vector<float> dequant_datas(quant_datas->size());
  if (channels <= 0) {
    MS_LOG(ERROR) << "channels must be greater than 0";
    return RET_ERROR;
  }
  size_t one_filter_size = elem_count / channels;
  bool do_quant = quant_param_size / (sizeof(float) * 8 - bit_num) < one_filter_size;
  if (!do_quant && quant_type == schema::QuantType_QUANT_WEIGHT) {
    MS_LOG(INFO) << "too few elements in a filter, no need to quantize. " << one_filter_size;
    return RET_QUANT_CONTINUE;
  }
  for (int i = 0; i < channels; i++) {
    float min = FLT_MAX;
    float max = -FLT_MAX;
    STATUS status =
      GetMaxMinPerChannel(channels, one_filter_size, i, elem_count, raw_datas, channel_at_first, &max, &min);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "GetMaxMinPerChannel failed" << status;
      return status;
    }
    schema::QuantParamT quant_param;
    status = CalQuantizationParams(&quant_param, min, max, false, quant_max, quant_min, bit_num);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "CalQuantizationParams failed" << status;
      return status;
    }
    // do quantization
    double average_dequant = 0;
    double average_raw = 0;
    for (uint32_t j = 0; j < one_filter_size; j++) {
      auto index = j + i * one_filter_size;
      if (!channel_at_first) {
        index = j * channels + i;
      }
      MS_ASSERT(index < elem_count);
      float raw_data = raw_datas[index];
      auto quant_data = QuantizeData<T>(raw_data, &quant_param, quant_max, quant_min);
      (*quant_datas)[index] = quant_data;

      if (quant_type == schema::QuantType_QUANT_WEIGHT) {
        float dequant_data = quant_param.scale * (quant_data - quant_param.zeroPoint);
        dequant_datas[index] = dequant_data;
        average_dequant += dequant_data;
        average_raw += raw_data;
      }
    }
    if (quant_type == schema::QuantType_QUANT_WEIGHT && !k_means) {
      // mean
      average_dequant = average_dequant / one_filter_size;
      average_raw = average_raw / one_filter_size;
      // std
      double variance_dequant = 0;
      double variance_raw = 0;
      for (uint32_t j = 0; j < one_filter_size; j++) {
        auto index = j + i * one_filter_size;
        if (!channel_at_first) {
          index = j * channels + i;
        }
        MS_ASSERT(index < elem_count);
        variance_dequant += std::pow(dequant_datas[index] - average_dequant, 2);
        variance_raw += std::pow(raw_datas[index] - average_raw, 2);
      }
      variance_dequant = std::sqrt(variance_dequant / one_filter_size);
      variance_raw = std::sqrt(variance_raw / one_filter_size);
      quant_param.varCorr = 1;
      if (variance_raw != 0 && variance_dequant != 0) {
        auto temp_var_corr = variance_raw / variance_dequant;
        if (temp_var_corr > 0 && temp_var_corr < 10) {
          quant_param.varCorr = temp_var_corr;
        } else {
          MS_LOG(WARNING) << "unexpected var_corr: " << temp_var_corr;
        }
      }
      quant_param.meanCorr = average_raw - average_dequant * quant_param.varCorr;
    }
    quant_params->emplace_back(quant_param);
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_COMMON_QUANT_UTILS_H_
