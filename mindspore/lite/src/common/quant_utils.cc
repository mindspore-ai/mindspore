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

#include "src/common/quant_utils.h"
#include <functional>
#include <map>
#include <cmath>
#include <cfloat>

namespace mindspore {
namespace lite {
// `symmetric` == true -> q range is [-127 , 127];
//  abs_max = max(abs(r_min),abs(r_max)); r_min = -abs_max and r_max = abs_max.
//  `symmetric` == false q range is [-128 , 127]. r_min or r_max keep the original value.
// `narrow_range` is used to adjust q_min, and symmetric is always true.
int CalQuantizationParams(schema::QuantParamT *quant_param, double real_min, double real_max, int num_bits,
                          bool symmetric, bool narrow_range) {
  CHECK_NULL_RETURN(quant_param);
  int quant_max = QuantMax(num_bits);
  int quant_min = QuantMin(num_bits, false, narrow_range);
  return CalQuantizationParams(quant_param, real_min, real_max, num_bits, quant_min, quant_max, symmetric,
                               narrow_range);
}

void EncodeMinMax(float min_value, float max_value, int quant_min, int quant_max, bool symmetric, float *encode_min,
                  float *encode_max) {
  // handle case where encode_min_ == encode_max_
  float epsilon = 1e-10;
  if (max_value - min_value < epsilon) {
    MS_LOG(INFO) << min_value << " - " << max_value;
  }
  max_value = std::max(max_value, min_value + epsilon);
  if (symmetric) {
    auto abs_max = std::max(std::fabs(min_value), std::fabs(max_value));
    *encode_min = -abs_max;
    *encode_max = abs_max;
  } else {
    *encode_min = min_value;
    *encode_max = max_value;
  }
  // Handling 0
  // Inputs are strictly positive, set the real min to 0. e.g. input range = [1.0, 5.0] -> [0.0, 5.0]
  if (*encode_min > 0.0f) {
    MS_LOG(DEBUG) << "min " << *encode_min << " is bigger then 0, set to 0, this may course low precision";
    *encode_min = 0.0f;
  }
  // Inputs are strictly negative, set the real max to 0. e.g. input range = [-5.0, -1.0] -> [-5.0, 0.0]
  if (*encode_max < 0.0f) {
    MS_LOG(DEBUG) << "real_max " << *encode_max << " is smaller than 0, set to 0, this may course low precision";
    *encode_max = 0.0f;
  }
  auto q_range = quant_max - quant_min;
  MS_ASSERT(quant_max - quant_min > 0);
  // Inputs are both negative and positive, real_min and real_max are slightly shifted to make the floating point zero
  // exactly representable. e.g. input range = [-5.1, 5.1] -> [-5.12, 5.08]
  double step_size = static_cast<double>(*encode_max - *encode_min) / q_range;
  auto close_0 = std::round(-(*encode_min) / step_size);
  *encode_min = (0 - close_0) * step_size;
  *encode_max = (q_range - close_0) * step_size;
}

int CalQuantizationParams(schema::QuantParamT *quant_param, double real_min, double real_max, int num_bits,
                          int quant_min, int quant_max, bool symmetric, bool narrow_range) {
  CHECK_NULL_RETURN(quant_param);
  float encode_min = real_min;
  float encode_max = real_max;
  EncodeMinMax(real_min, real_max, quant_min, quant_max, symmetric, &encode_min, &encode_max);
  auto q_range = quant_max - quant_min;
  double scale = (encode_max - encode_min) / q_range;
  if (fabs(scale) <= 0.0f) {
    MS_LOG(ERROR) << "divisor 'scale' cannot be 0";
    return RET_ERROR;
  }
  int zero_point = static_cast<int32_t>(std::round(quant_min - encode_min / scale));

  // The zero point should always be in the range of quantized value,
  // [qmin, qmax].
  MS_ASSERT(zero_point >= quant_min);
  MS_ASSERT(zero_point <= quant_max);
  quant_param->inited = true;
  quant_param->min = encode_min;
  quant_param->max = encode_max;
  quant_param->scale = scale;
  quant_param->zeroPoint = zero_point;
  quant_param->narrowRange = narrow_range;
  quant_param->numBits = num_bits;

  return RET_OK;
}

// Get the index of the bucket to which the current data belongs.
int GetBucketIndex(const std::vector<int> &dims, int preferred_dim, int data_index) {
  int stride = 1;
  int bucket_count = dims[preferred_dim];
  for (size_t i = static_cast<size_t>(preferred_dim + 1); i < dims.size(); i++) {
    stride *= dims[i];
  }
  if (stride == 0 || bucket_count == 0) {
    MS_LOG(ERROR) << "stride or bucket_count is 0.";
    return 0;
  }
  return (data_index / stride) % bucket_count;
}

void GetAllChannelMinMax(const float *raw_datas, size_t elem_count, const std::vector<int> &dims, int preferred_dim,
                         std::map<int, MinMax> *per_channel_min_max) {
  MS_ASSERT(raw_datas != nullptr);
  MS_ASSERT(per_channel_min_max != nullptr);
  // the key is bucket_index
  for (int i = 0; i < dims[preferred_dim]; ++i) {
    per_channel_min_max->insert({i, {FLT_MAX, -FLT_MAX}});
  }
  // the first dim.
  if (preferred_dim == 0) {
    auto bucket_size = elem_count / dims[preferred_dim];
    for (int i = 0; i < dims[preferred_dim]; ++i) {
      auto mim_max = GetFloatMinMaxValue(raw_datas + i * bucket_size, bucket_size);
      auto iter = per_channel_min_max->find(i);
      MS_ASSERT(iter != per_channel_min_max->end());
      iter->second.min = mim_max.first;
      iter->second.max = mim_max.second;
    }
  } else {
    for (size_t i = 0; i < elem_count; ++i) {
      auto bucket_index = GetBucketIndex(dims, preferred_dim, i);
      auto iter = per_channel_min_max->find(bucket_index);
      MS_ASSERT(iter != per_channel_min_max->end());
      iter->second.min = std::min(iter->second.min, raw_datas[i]);
      iter->second.max = std::max(iter->second.max, raw_datas[i]);
    }
  }
}

int CalPerChannelGain(size_t bit_num, const std::vector<int> &dims, int preferred_dim) {
  auto elem_count = std::accumulate(std::begin(dims), std::end(dims), 1, std::multiplies<>());
  const int bits_per_byte = 8;
  const int quant_param_size = 32;
  int channels = dims.at(preferred_dim);
  if (channels < 1) {
    MS_LOG(ERROR) << "channels must not less 1";
    return RET_ERROR;
  }
  size_t bucket_size = static_cast<size_t>(elem_count / channels);
  bool do_quant = (quant_param_size * bits_per_byte) / (sizeof(float) * bits_per_byte - bit_num) < bucket_size;
  if (do_quant) {
    return RET_OK;
  } else {
    MS_LOG(INFO) << "too few elements in a filter, no need to quantize. " << bucket_size;
    return RET_NO_CHANGE;
  }
}

int CalWeightQuantBias(const float *raw_datas, size_t elem_count, const std::vector<float> &dequant_datas,
                       std::vector<schema::QuantParamT> *quant_params, const std::vector<int> &dims,
                       int preferred_dim) {
  CHECK_NULL_RETURN(raw_datas);
  CHECK_NULL_RETURN(quant_params);
  std::map<int, double> total_raws;
  std::map<int, double> total_dequants;
  std::map<int, double> average_raws;
  std::map<int, double> average_dequants;
  std::map<int, double> var_raws;
  std::map<int, double> var_dequants;
  size_t bucket_size = quant_params->size();
  int bucket_volume = static_cast<size_t>(elem_count / dims[preferred_dim]);
  // Init Map
  for (size_t i = 0; i < bucket_size; i++) {
    total_raws[i] = 0;
    total_dequants[i] = 0;
    average_raws[i] = 0;
    average_dequants[i] = 0;
    var_raws[i] = 0;
    var_dequants[i] = 0;
  }
  for (size_t data_index = 0; data_index < elem_count; data_index++) {
    auto data = raw_datas[data_index];
    auto dequant_data = dequant_datas[data_index];
    auto bucket_index = GetBucketIndex(dims, preferred_dim, data_index);
    total_raws[bucket_index] += data;
    total_dequants[bucket_index] += dequant_data;
  }
  for (size_t bucket_index = 0; bucket_index < bucket_size; bucket_index++) {
    average_raws[bucket_index] = total_raws[bucket_index] / bucket_volume;
    average_dequants[bucket_index] = total_dequants[bucket_index] / bucket_volume;
  }

  constexpr int pow_exponent = 2;
  for (size_t data_index = 0; data_index < elem_count; data_index++) {
    auto bucket_index = GetBucketIndex(dims, preferred_dim, data_index);
    var_raws[bucket_index] += std::pow(raw_datas[data_index] - average_raws[bucket_index], pow_exponent);
    var_dequants[bucket_index] += std::pow(dequant_datas[data_index] - average_dequants[bucket_index], pow_exponent);
  }
  for (size_t bucket_index = 0; bucket_index < bucket_size; bucket_index++) {
    var_raws[bucket_index] = std::sqrt(var_raws[bucket_index] / bucket_volume);
    var_dequants[bucket_index] = std::sqrt(var_dequants[bucket_index] / bucket_volume);
  }
  for (size_t bucket_index = 0; bucket_index < bucket_size; bucket_index++) {
    quant_params->at(bucket_index).varCorr = 1;
    if (fabs(var_raws[bucket_index]) > DBL_EPSILON && fabs(var_dequants[bucket_index]) > DBL_EPSILON) {
      auto temp_var_corr = var_raws[bucket_index] / var_dequants[bucket_index];
      const int min_var_corr = 0;
      const int max_var_corr = 10;
      if (temp_var_corr > min_var_corr && temp_var_corr < max_var_corr) {
        quant_params->at(bucket_index).varCorr = temp_var_corr;
      } else {
        MS_LOG(WARNING) << "unexpected var_corr: " << temp_var_corr;
      }
    }
    quant_params->at(bucket_index).meanCorr =
      average_raws[bucket_index] - average_dequants[bucket_index] * quant_params->at(bucket_index).varCorr;
    MS_LOG(INFO) << "dims:" << dims << " bucket_index:" << bucket_index
                 << " average_raws[bucket_index]:" << average_raws[bucket_index]
                 << " average_dequants[bucket_index]:" << average_dequants[bucket_index]
                 << " var_raws[bucket_index]:" << var_dequants[bucket_index]
                 << " var_dequants[bucket_index]:" << var_dequants[bucket_index]
                 << " varCorr:" << quant_params->at(bucket_index).varCorr
                 << " meanCorr:" << quant_params->at(bucket_index).meanCorr;
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
