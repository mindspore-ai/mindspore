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

#include "tools/converter/quantizer/fix_bit_weight_quantizer.h"
#include <cmath>

namespace mindspore::lite::quant {
// the error is currently measured per channel.
// it could be measured per layer but it would be less good.
// the `preferred` dim should point to the output channels dimension.
float FixBitWeightQuantizer::MeasureQuantizationError(float *weights, const int *shape, int dims, int preferred_dim,
                                                      float scale) {
  int numel = 1;
  for (int i = 0; i < dims; i++) {
    numel *= shape[i];
  }
  int bucket_count = shape[preferred_dim];
  std::vector<float> norms2(bucket_count);
  std::vector<float> dnorms2(bucket_count);
  for (int i = 0; i < bucket_count; i++) {
    norms2[i] = 0.0;
    dnorms2[i] = 0.0;
  }
  double average_dequant = 0;
  double average_raw = 0;
  std::vector<float> dequant_datas(numel);
  int bucket_volume = 1;
  for (int i = preferred_dim; i < dims; i++) {
    bucket_volume *= shape[i];
  }
  for (int i = 0; i < numel; i++) {
    float dequant = scale * (floorf(weights[i] / scale + 0.5));
    dequant_datas[i] = dequant;
    average_raw += weights[i];
    average_dequant += dequant;
  }
  // mean
  average_dequant = average_dequant / numel;
  average_raw = average_raw / numel;
  // std
  double variance_dequant = 0;
  double variance_raw = 0;
  for (int i = 0; i < numel; i++) {
    variance_dequant += std::pow(dequant_datas[i] - average_dequant, 2);
    variance_raw += std::pow(weights[i] - average_raw, 2);
  }
  variance_dequant = std::sqrt(variance_dequant / numel);
  variance_raw = std::sqrt(variance_raw / numel);
  var_corr = variance_raw / variance_dequant;
  mean_corr = average_raw - average_dequant * var_corr;

  for (int i = 0; i < numel; i++) {
    int bucket = (i / bucket_volume) % bucket_count;
    norms2[bucket] += weights[i] * weights[i];
    float dequant = var_corr * (scale * (floorf(weights[i] / scale + 0.5))) + mean_corr;
    float d = weights[i] - dequant;
    dnorms2[bucket] += d * d;
  }

  int c = 0;
  float t = 1e-10;
  for (int i = 0; i < bucket_count; i++) {
    if (norms2[i] < 1.0e-10) continue;
    c += 1;
    t += sqrtf(dnorms2[i] / norms2[i]);
  }
  return t / (c + 1e-7);
}

MinMax FixBitWeightQuantizer::GetMinMax(const float *arr, int arrc) {
  MinMax min_max = {INFINITY, -INFINITY};
  for (int i = 0; i < arrc; i++)
    if (arr[i] > min_max.max)
      min_max.max = arr[i];
    else if (arr[i] < min_max.min)
      min_max.min = arr[i];
  return min_max;
}

BinarySearchResult FixBitWeightQuantizer::BinarySearchForQuantizationScale(float *weights, int *shape, int dims,
                                                                           int preferred_dim, int max_iters,
                                                                           float target_err, float rel_tol) {
  int element_num = 1;
  for (int i = 0; i < dims; i++) {
    element_num *= shape[i];
  }
  MinMax mm = GetMinMax(weights, element_num);
  if (mm.max < mm.min + 1.0e-5) {
    return {0, static_cast<float>(std::fabs(mm.max) + 1.0e-5)};
  }
  // start a binary search
  float curr_scale = (mm.max - mm.min) * target_err;
  float right_hs_dx = curr_scale * 2.0;
  while (MeasureQuantizationError(weights, shape, dims, preferred_dim, right_hs_dx) < target_err) {
    right_hs_dx *= 2.0;
  }
  float left_hs_dx = curr_scale / 2.0;
  while (MeasureQuantizationError(weights, shape, dims, preferred_dim, left_hs_dx) > target_err) {
    left_hs_dx /= 2.0;
  }
  int iter_count = 0;
  BinarySearchResult res = {0, curr_scale};
  while (true) {
    float curr_err = MeasureQuantizationError(weights, shape, dims, preferred_dim, res.scale);
    if (std::fabs(curr_err - target_err) / target_err < rel_tol) {
      return res;
    }
    if (iter_count > max_iters) {
      res.status = 1;
      return res;
    }
    if (curr_err > target_err)
      right_hs_dx = res.scale;
    else
      left_hs_dx = res.scale;
    res.scale = (left_hs_dx + right_hs_dx) / 2.0;
    iter_count += 1;
  }
}

int FixBitWeightQuantizer::DoQuantization(float *weights, std::vector<int64_t> shape, int preferred_dim,
                                          std::vector<schema::QuantParamT> *quant_params,
                                          std::vector<int16_t> *quant_datas) {
  int weight_count = 1;
  int dims = shape.size();
  int input_shape[4] = {0, 0, 0, 0};
  for (int i = 0; i < dims; i++) {
    weight_count *= shape[i];
    input_shape[i] = shape[i];
  }

  BinarySearchResult br = BinarySearchForQuantizationScale(weights, input_shape, dims, preferred_dim, max_search_iters,
                                                           target_relative_err, target_search_tolerance);
  if (br.status != 0) {
    MS_LOG(ERROR) << "reached_max_iters";
    return RET_ERROR;
  }
  schema::QuantParamT quant_param;
  int qr = QuantizeByScale(weights, weight_count, br.scale, &quant_param, quant_datas);
  if (qr != 0) {
    MS_LOG(ERROR) << "quant failed.";
    return RET_ERROR;
  }

  // It is used to calculate the Shannon entropy.
  quant_params->push_back(quant_param);
  return RET_OK;
}

int FixBitWeightQuantizer::QuantizeByScale(const float *weights, int weightsc, float scale,
                                           schema::QuantParamT *quant_params, std::vector<int16_t> *quant_datas) {
  for (int i = 0; i < weightsc; i++) {
    auto q = static_cast<int>(floorf(weights[i] / scale + 0.5));
    quant_datas->at(i) = q;
  }
  quant_params->meanCorr = mean_corr;
  quant_params->varCorr = var_corr;
  quant_params->scale = scale;
  quant_params->zeroPoint = 0;
  quant_params->numBits = 0;
  quant_params->inited = true;
  return RET_OK;
}
}  // namespace mindspore::lite::quant
