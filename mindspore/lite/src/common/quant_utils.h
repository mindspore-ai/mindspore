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

#include <cfloat>
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
#include "tools/common/statistic_utils.h"

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

inline int QuantMax(int bits, bool is_unsigned = false) {
  if (!is_unsigned) {
    return (1 << static_cast<unsigned int>(bits - 1)) - 1;
  } else {
    return (1 << static_cast<unsigned int>(bits)) - 1;
  }
}

inline int QuantMin(int bits, bool is_unsigned = false, bool is_narrow = true) {
  if (!is_unsigned) {
    return -(1 << static_cast<unsigned int>(bits - 1)) + (is_narrow ? 1 : 0);
  } else {
    return 0;
  }
}

int CalQuantizationParams(schema::QuantParamT *quant_param, double real_min, double real_max, int num_bits,
                          int quant_min, int quant_max, bool symmetric, bool narrow_range = false);

int CalQuantizationParams(schema::QuantParamT *quant_param, double real_min, double real_max, int num_bits,
                          bool symmetric, bool narrow_range = false);

void EncodeMinMax(float min_value, float max_value, int quant_min, int quant_max, bool symmetric, float *encode_min,
                  float *encode_max);
template <typename T>
T QuantizeData(float origin_data, const schema::QuantParamT *quant_param, int quant_max, int quant_min) {
  MS_ASSERT(quant_param != nullptr);
  MS_ASSERT(quant_param->inited);
  const auto scale = quant_param->scale;
  const int zero_point = quant_param->zeroPoint;
  if (scale <= SCALE_THREASHOLD) {
    return 0;
  }
  return [quant_max, quant_min, zero_point, scale, origin_data] {
    auto quant_data = std::round(origin_data / scale + zero_point);
    if (quant_data > quant_max) {
      quant_data = quant_max;
    } else if (quant_data < quant_min) {
      quant_data = quant_min;
    }
    return static_cast<T>(quant_data);
  }();
}

template <typename T>
T QuantizeData(const float origin_data, const schema::QuantParamT *quant_param) {
  MS_ASSERT(quant_param != nullptr);
  MS_ASSERT(quant_param->inited);
  const auto num_bit = quant_param->numBits;
  const auto narrow_range = quant_param->narrowRange;
  const int32_t quant_max = QuantMax(num_bit, false);
  const int32_t quant_min = QuantMin(num_bit, false, narrow_range);
  return QuantizeData<T>(origin_data, quant_param, quant_max, quant_min);
}

template <typename T>
int DoPerLayerQuant(const float *raw_datas, size_t elem_count, std::vector<schema::QuantParamT> *quant_params,
                    const int &quant_max, const int &quant_min, const size_t &bit_num, std::vector<T> *quant_datas,
                    bool symmetric = false, bool narrow_range = false) {
  auto min_max = GetFloatMinMaxValue(raw_datas, elem_count);
  schema::QuantParamT quant_param;
  int status = CalQuantizationParams(&quant_param, min_max.first, min_max.second, bit_num, quant_min, quant_max,
                                     symmetric, narrow_range);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "CalQuantizationParams failed" << status;
    return status;
  }
  quant_params->emplace_back(quant_param);
  // update data and datatype
  for (size_t i = 0; i < elem_count; i++) {
    float raw_data = raw_datas[i];
    auto quant_data = QuantizeData<T>(raw_data, &quant_param, quant_max, quant_min);
    (*quant_datas)[i] = quant_data;
  }
  return RET_OK;
}

// Get the index of the bucket to which the current data belongs.
int GetBucketIndex(const std::vector<int> &dims, int preferred_dim, int data_index);

// Calculate the Compression effect of per-channel
int CalPerChannelGain(size_t bit_num, const std::vector<int> &dims, int preferred_dim);

// Get the min max of each channel
void GetAllChannelMinMax(const float *raw_datas, size_t elem_count, const std::vector<int> &dims, int preferred_dim,
                         std::map<int, MinMax> *per_channel_min_max);

// Calculate the distribution difference between quant and origin
int CalWeightQuantBias(const float *raw_datas, size_t elem_count, const std::vector<float> &dequant_datas,
                       std::vector<schema::QuantParamT> *quant_params, const std::vector<int> &dims, int preferred_dim);

template <typename T>
int DoPerChannelQuant(const float *raw_datas, size_t elem_count, const schema::QuantType &quant_type,
                      std::vector<schema::QuantParamT> *quant_params, const int &quant_max, const int &quant_min,
                      const size_t &bit_num, std::vector<T> *quant_datas, const std::vector<int> &dims,
                      int preferred_dim, bool symmetric = false, bool narrow_range = false) {
  if (raw_datas == nullptr || quant_params == nullptr || quant_datas == nullptr) {
    MS_LOG(ERROR) << "raw_data, quant_params or quant_data is nullptr.";
    return RET_ERROR;
  }
  int ret;
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
  GetAllChannelMinMax(raw_datas, elem_count, dims, preferred_dim, &per_channel_min_max);

  // Cal Quant param
  for (auto min_max_map : per_channel_min_max) {
    float min = min_max_map.second.min;
    float max = min_max_map.second.max;
    schema::QuantParamT quant_param;
    ret = CalQuantizationParams(&quant_param, min, max, bit_num, quant_min, quant_max, symmetric, narrow_range);
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
