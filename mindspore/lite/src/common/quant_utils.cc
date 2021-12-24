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

int CalQuantizationParams(schema::QuantParamT *quant_param, double real_min, double real_max, int num_bits,
                          int quant_min, int quant_max, bool symmetric, bool narrow_range) {
  CHECK_NULL_RETURN(quant_param);
  if (symmetric) {
    auto abs_max = std::max(std::abs(real_min), std::abs(real_max));
    real_min = -abs_max;
    real_max = abs_max;
    narrow_range = true;
  }
  // Handling 0
  // Inputs are strictly positive, set the real min to 0. e.g. input range = [1.0, 5.0] -> [0.0, 5.0]
  if (real_min > 0.0f) {
    MS_LOG(DEBUG) << "min " << real_min << " is bigger then 0, set to 0, this may course low precision";
    real_min = 0.0f;
  }
  // Inputs are strictly negative, set the real max to 0. e.g. input range = [-5.0, -1.0] -> [-5.0, 0.0]
  if (real_max < 0.0f) {
    MS_LOG(DEBUG) << "real_max " << real_max << " is smaller than 0, set to 0, this may course low precision";
    real_max = 0.0f;
  }
  // Inputs are both negative and positive, real_min and real_max are slightly shifted to make the floating point zero
  // exactly representable. e.g. input range = [-5.1, 5.1] -> [-5.12, 5.08]

  if (real_min > real_max) {
    MS_LOG(ERROR) << "cal error while min" << real_min << ">" << real_max;
    return RET_PARAM_INVALID;
  }
  if (real_max - real_min <= 0.0f) {
    if (real_min != 0.0f) {
      MS_LOG(ERROR) << "min and max should both be zero if they are equal to each other";
      return RET_ERROR;
    }
    MS_LOG(INFO) << "The maximum and minimum values are equal to 0.";
    quant_param->inited = true;
    quant_param->min = real_min;
    quant_param->max = real_max;
    quant_param->scale = 1;
    quant_param->zeroPoint = 0;
    quant_param->narrowRange = narrow_range;
    quant_param->numBits = num_bits;
    return RET_OK;
  }

  if (quant_max - quant_min == 0) {
    MS_LOG(ERROR) << "divisor cannot be 0";
    return RET_ERROR;
  }
  double scale = (real_max - real_min) / (quant_max - quant_min);
  if (fabs(scale) <= 0.0f) {
    MS_LOG(ERROR) << "divisor 'scale' cannot be 0";
    return RET_ERROR;
  }
  int zero_point = static_cast<int32_t>(std::round(quant_min - real_min / scale));

  // The zero point should always be in the range of quantized value,
  // [qmin, qmax].
  MS_ASSERT(zero_point >= quant_min);
  MS_ASSERT(zero_point <= quant_max);
  quant_param->inited = true;
  quant_param->min = real_min;
  quant_param->max = real_max;
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
  for (size_t i = preferred_dim + 1; i < dims.size(); i++) {
    stride *= dims[i];
  }
  return (data_index / stride) % bucket_count;
}

int GetAllChannelMinMax(const float *raw_datas, int elem_count, const std::vector<int> &dims, int preferred_dim,
                        std::map<int, MinMax> *per_channel_min_max) {
  // the key is bucket_index
  std::map<int, std::vector<float>> sorted_data;
  for (int i = 0; i < elem_count; ++i) {
    auto bucket_index = GetBucketIndex(dims, preferred_dim, i);
    auto iter = sorted_data.find(bucket_index);
    if (iter == sorted_data.end()) {
      sorted_data.insert({bucket_index, {raw_datas[i]}});
    } else {
      iter->second.push_back(raw_datas[i]);
    }
  }
  for (size_t i = 0; i < sorted_data.size(); ++i) {
    auto data = sorted_data.at(i);
    MinMax min_max;
    min_max.max = *max_element(data.begin(), data.end());
    min_max.min = *min_element(data.begin(), data.end());
    per_channel_min_max->insert({i, min_max});
  }
  return RET_OK;
}

int CalPerChannelGain(size_t bit_num, const std::vector<int> &dims, int preferred_dim) {
  auto elem_count = std::accumulate(std::begin(dims), std::end(dims), 1, std::multiplies<>());
  static const int quant_param_size = 32 * 8;
  int channels = dims.at(preferred_dim);
  CHECK_LESS_RETURN(channels, 1);
  size_t bucket_size = elem_count / channels;
  bool do_quant = quant_param_size / (sizeof(float) * 8 - bit_num) < bucket_size;
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
  int bucket_volume = elem_count / dims[preferred_dim];
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
    if (var_raws[bucket_index] != 0 && var_dequants[bucket_index] != 0) {
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
