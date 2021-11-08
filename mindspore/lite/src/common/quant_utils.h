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
#include <numeric>
#include <functional>
#include <map>
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/common/log_util.h"
#include "ir/dtype/type_id.h"
#include "schema/inner/model_generated.h"

namespace mindspore {

namespace schema {
struct QuantParamT;
}

namespace lite {

typedef struct {
  float min;
  float max;
} MinMax;

static constexpr double SCALE_THREASHOLD = 1e-38;

static constexpr int kPerTensor = 1;

inline int QuantMax(int bits, TypeId type) {
  if (type == kNumberTypeInt8) {
    return (1 << static_cast<unsigned int>(bits - 1)) - 1;
  } else if (type == kNumberTypeUInt8) {
    return (1 << static_cast<unsigned int>(bits)) - 1;
  }
  return 0;
}

inline int QuantMin(int bits, TypeId type) {
  if (type == kNumberTypeInt8) {
    return -(1 << static_cast<unsigned int>(bits - 1));
  }
  return 0;
}

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

// Get the index of the bucket to which the current data belongs.
int GetBucketIndex(const std::vector<int> &dims, int preferred_dim, int data_index);

// Calculate the Compression effect of per-channel
STATUS CalPerChannelGain(size_t bit_num, const std::vector<int> &dims, int preferred_dim);

// Get the min max of each channel
STATUS GetAllChannelMinMmax(const float *raw_datas, int elem_count, const std::vector<int> &dims, int preferred_dim,
                            std::map<int, MinMax> *per_channel_min_max);

// Calculate the distribution difference between quant and origin
STATUS CalWeightQuantBias(const float *raw_datas, size_t elem_count, const std::vector<float> &dequant_datas,
                          std::vector<schema::QuantParamT> *quant_params, const std::vector<int> &dims,
                          int preferred_dim);

template <typename T>
STATUS DoPerChannelQuant(const float *raw_datas, size_t elem_count, const schema::QuantType &quant_type,
                         std::vector<schema::QuantParamT> *quant_params, const int &quant_max, const int &quant_min,
                         const size_t &bit_num, const bool &k_means, std::vector<T> *quant_datas,
                         const std::vector<int> &dims, int preferred_dim) {
  STATUS ret;
  auto count = std::accumulate(std::begin(dims), std::end(dims), 1, std::multiplies<>());
  if (static_cast<size_t>(count) != elem_count) {
    MS_LOG(ERROR) << " element != count";
    return RET_ERROR;
  }

  CHECK_LESS_RETURN(dims.size(), static_cast<size_t>(preferred_dim + 1));
  if (quant_type == schema::QuantType_QUANT_WEIGHT) {
    ret = CalPerChannelGain(bit_num, dims, preferred_dim);
    if (ret == RET_NO_CHANGE) {
      return RET_NO_CHANGE;
    }
  }

  std::vector<float> dequant_datas(quant_datas->size());
  // the key is bucket_index
  std::map<int, MinMax> per_channel_min_max;
  ret = GetAllChannelMinMmax(raw_datas, elem_count, dims, preferred_dim, &per_channel_min_max);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get all channel min max failed.";
    return ret;
  }

  // Cal Quant param
  for (auto min_max_map : per_channel_min_max) {
    float min = min_max_map.second.min;
    float max = min_max_map.second.max;
    schema::QuantParamT quant_param;
    ret = CalQuantizationParams(&quant_param, min, max, false, quant_max, quant_min, bit_num);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Cal quantization params failed.";
      return ret;
    }
    quant_params->emplace_back(quant_param);
  }

  // Do quant
  for (size_t i = 0; i < elem_count; i++) {
    float raw_data = raw_datas[i];
    auto bucket_index = GetBucketIndex(dims, preferred_dim, i);
    auto quant_param = quant_params->at(bucket_index);
    auto quant_data = QuantizeData<T>(raw_data, &quant_param, quant_max, quant_min);
    (*quant_datas)[i] = quant_data;
    // cal dequant(use for cal weight bias)
    dequant_datas.at(i) = quant_param.scale * (quant_data - quant_param.zeroPoint);
  }

  if (quant_type == schema::QuantType_QUANT_WEIGHT) {
    ret = CalWeightQuantBias(raw_datas, elem_count, dequant_datas, quant_params, dims, preferred_dim);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Cal weight quant bias failed.";
      return ret;
    }
  }
  return RET_OK;
}

}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_COMMON_QUANT_UTILS_H_
