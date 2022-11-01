/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "tools/converter/quantizer/mixed_bit_weight_quantization.h"
#include <cmath>
#include <cfloat>
#include <map>
#include "tools/common/statistic_utils.h"
#include "tools/converter/quantizer/quantize_util.h"

namespace mindspore::lite::quant {
constexpr float kTwentyFour = 24.0f;

void MixedBitWeightQuantization::CalculateBiasCorrection(float *weights, int element_num, float scale,
                                                         float *origin_dequant_datas) {
  MS_ASSERT(weights != nullptr);
  MS_ASSERT(origin_dequant_datas != nullptr);
  MS_ASSERT(element_num > 0);
  double average_dequant = 0;
  double average_raw = 0;
  const float upround_offset = 0.5;
  for (int i = 0; i < element_num; i++) {
    float dequant = scale * (floorf(weights[i] / scale + upround_offset));
    origin_dequant_datas[i] = dequant;
    average_raw += weights[i];
    average_dequant += dequant;
  }

  // mean
  average_dequant = average_dequant / element_num;
  average_raw = average_raw / element_num;
  // std
  double variance_dequant = 0;
  double variance_raw = 0;
  const int exponent = 2;
  for (int i = 0; i < element_num; i++) {
    variance_dequant += std::pow(origin_dequant_datas[i] - average_dequant, exponent);
    variance_raw += std::pow(weights[i] - average_raw, exponent);
  }
  MS_ASSERT(variance_dequant >= 0);
  MS_ASSERT(variance_raw >= 0);
  variance_dequant = std::sqrt(variance_dequant / element_num);
  variance_raw = std::sqrt(variance_raw / element_num);
  if (fabs(variance_dequant) < DBL_EPSILON) {
    var_corr_ = 1;
  } else {
    var_corr_ = variance_raw / variance_dequant;
  }
  mean_corr_ = average_raw - average_dequant * var_corr_;
}

// the error is currently measured per channel.
float MixedBitWeightQuantization::CalculateMeanError(std::vector<float> norms2, std::vector<float> dnorms2) {
  int error_count = 0;
  float mse_error = 1e-10f;
  const float soft = 1e-7f;
  const float tolerance_error = 1.0e-10f;
  for (size_t i = 0; i < norms2.size(); i++) {
    if (norms2[i] < tolerance_error) {
      continue;
    }
    error_count += 1;
    mse_error += sqrtf(dnorms2[i] / norms2[i]);
  }
  auto mean_error = mse_error / (error_count + soft);
  return mean_error;
}

// the `preferred` dim should point to the output channels dimension.
float MixedBitWeightQuantization::MeasureQuantizationError(float *weights, const int *shape, int dims,
                                                           int preferred_dim, float scale) {
  MS_ASSERT(weights != nullptr);
  MS_ASSERT(shape != nullptr);
  // Init
  int element_num = 1;
  for (int i = 0; i < dims; i++) {
    element_num *= shape[i];
  }
  if (element_num <= 0) {
    MS_LOG(ERROR) << "Element is less than or equal to 0.";
    return FLT_MAX;
  }
  int bucket_count = shape[preferred_dim];
  std::vector<float> norms2(bucket_count);
  std::vector<float> dnorms2(bucket_count);
  const float init_number = 0.0;
  for (int i = 0; i < bucket_count; i++) {
    norms2[i] = init_number;
    dnorms2[i] = init_number;
  }

  // Bucketing
  std::vector<float> origin_dequant_datas(element_num);
  std::vector<float> corr_dequant_datas(element_num);
  int bucket_volume = 1;
  for (int i = preferred_dim; i < dims; i++) {
    bucket_volume *= shape[i];
  }
  MS_ASSERT(bucket_volume != 0);
  const float upround_offset = 0.5;
  // Bias Correction
  CalculateBiasCorrection(weights, element_num, scale, origin_dequant_datas.data());
  for (int i = 0; i < element_num; i++) {
    int bucket = (i / bucket_volume) % bucket_count;
    norms2[bucket] += weights[i] * weights[i];
    float dequant = var_corr_ * (scale * (floorf(weights[i] / scale + upround_offset))) + mean_corr_;
    corr_dequant_datas[i] = dequant;
    float d = weights[i] - dequant;
    dnorms2[bucket] += d * d;
  }
  auto mean_error = CalculateMeanError(norms2, dnorms2);
  return mean_error;
}

LayerParam MixedBitWeightQuantization::CalculateLayerParams(const float *weights, int element_num) {
  MS_ASSERT(weights != nullptr);
  float temp_norm_tot = 0.0;
  for (int i = 0; i < element_num; i++) {
    temp_norm_tot += weights[i] * weights[i];
  }

  LayerParam ret = {std::sqrt(1.0f / temp_norm_tot), GetMinMax(weights, element_num)};
  return ret;
}

MinMax MixedBitWeightQuantization::GetMinMax(const float *arr, int arrc) {
  MS_ASSERT(arr != nullptr);
  MinMax min_max = {INFINITY, -INFINITY};
  for (int i = 0; i < arrc; i++)
    if (arr[i] > min_max.max)
      min_max.max = arr[i];
    else if (arr[i] < min_max.min)
      min_max.min = arr[i];
  return min_max;
}

BinarySearchResult MixedBitWeightQuantization::BinarySearchForQuantizationScale(float *weights, int *shape, int dims,
                                                                                int preferred_dim, int max_iters,
                                                                                float target_err, float rel_tol) {
  MS_ASSERT(weights != nullptr);
  MS_ASSERT(shape != nullptr);
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
  float right_hs_dx = curr_scale * kBinarySearchStep;
  while (MeasureQuantizationError(weights, shape, dims, preferred_dim, right_hs_dx) < target_err) {
    right_hs_dx *= kBinarySearchStep;
  }
  float left_hs_dx = curr_scale / kBinarySearchStep;
  while (MeasureQuantizationError(weights, shape, dims, preferred_dim, left_hs_dx) > target_err) {
    left_hs_dx /= kBinarySearchStep;
  }
  int iter_count = 0;
  BinarySearchResult res = {0, curr_scale};
  while (true) {
    float curr_err = MeasureQuantizationError(weights, shape, dims, preferred_dim, res.scale);
    if (std::fabs(curr_err - target_err) / target_err < rel_tol) {
      return res;
    }
    if (iter_count > max_iters) {
      if (curr_err < target_err) {
        res.status = RET_OK;
      } else {
        res.status = RET_ERROR;
      }
      return res;
    }
    if (curr_err > target_err)
      right_hs_dx = res.scale;
    else
      left_hs_dx = res.scale;
    res.scale = (left_hs_dx + right_hs_dx) / kBinarySearchStep;
    iter_count += 1;
  }
}

float MixedBitWeightQuantization::GetDx(float *weights, const int *shape, int dims, const std::string &node_name) {
  MS_ASSERT(weights != nullptr);
  MS_ASSERT(shape != nullptr);
  static std::map<std::string, LayerParam> param_map;

  int element_num = 1;
  for (int i = 0; i < dims; i++) {
    element_num *= shape[i];
  }

  LayerParam params;
  auto params_it = param_map.find(node_name);
  if (params_it == param_map.end()) {
    params = CalculateLayerParams(weights, element_num);
    param_map.insert({node_name, params});
  } else {
    params = params_it->second;
  }
  return (target_relative_err_ + target_search_tolerance_ * std::sqrt(kTwentyFour / element_num)) / params.inv_norm;
}

int MixedBitWeightQuantization::DoQuantization(float *weights, std::vector<int64_t> shape, int preferred_dim,
                                               std::vector<schema::QuantParamT> *quant_params,
                                               std::vector<int16_t> *quant_datas, const std::string &node_name,
                                               bool use_auto_tune_alg) {
  CHECK_NULL_RETURN(weights);
  CHECK_NULL_RETURN(quant_params);
  CHECK_NULL_RETURN(quant_datas);
  int weight_count = 1;
  int dims = shape.size();
  int input_shape[4] = {0, 0, 0, 0};
  MS_ASSERT(dims <= input_shape.size());
  for (int i = 0; i < dims; i++) {
    weight_count *= shape[i];
    input_shape[i] = shape[i];
  }

  float scale = 1.0;
  if (use_auto_tune_alg) {
    scale = GetDx(weights, input_shape, dims, node_name);
  } else {
    BinarySearchResult br = BinarySearchForQuantizationScale(
      weights, input_shape, dims, preferred_dim, max_search_iters_, target_relative_err_, target_search_tolerance_);
    if (br.status != RET_OK) {
      MS_LOG(WARNING) << "this layer reached max iters.";
      return RET_NO_CHANGE;
    }
    scale = br.scale;
  }

  schema::QuantParamT quant_param;
  int qr = QuantizeByScale(weights, weight_count, scale, &quant_param, quant_datas);
  if (qr != RET_OK) {
    MS_LOG(ERROR) << "quant failed.";
    return RET_ERROR;
  }
  quant_params->push_back(quant_param);
  return RET_OK;
}

int MixedBitWeightQuantization::QuantizeByScale(const float *weights, int weightsc, float scale,
                                                schema::QuantParamT *quant_params, std::vector<int16_t> *quant_datas) {
  CHECK_NULL_RETURN(weights);
  CHECK_NULL_RETURN(quant_params);
  CHECK_NULL_RETURN(quant_datas);
  MS_CHECK_GE(static_cast<int>(quant_datas->size()), weightsc, RET_ERROR);
  const float upround_offset = 0.5;
  for (int i = 0; i < weightsc; i++) {
    auto q = static_cast<int>(floorf(weights[i] / scale + upround_offset));
    quant_datas->at(i) = q;
  }
  quant_params->meanCorr = mean_corr_;
  quant_params->varCorr = var_corr_;
  quant_params->scale = scale;
  quant_params->zeroPoint = 0;
  quant_params->numBits = 0;
  quant_params->inited = true;
  return RET_OK;
}

int MixedBitWeightQuantization::QuantFilter(const PrimitivePtr &primitive, const AnfNodePtr &parameter_node,
                                            const tensor::TensorPtr &weight, int index, schema::QuantType quant_type,
                                            bool use_auto_tune_alg) {
  CHECK_NULL_RETURN(primitive);
  CHECK_NULL_RETURN(weight);
  std::vector<schema::QuantParamT> quant_params;
  int elem_count = weight->DataSize();
  auto *raw_data = static_cast<float *>(weight->data_c());
  if (raw_data == nullptr) {
    MS_LOG(ERROR) << "rawDatas is nullptr";
    return RET_ERROR;
  }

  std::vector<int16_t> quant_data(elem_count);
  auto ret = DoQuantization(static_cast<float *>(weight->data_c()), weight->shape_c(), 0, &quant_params, &quant_data,
                            parameter_node->fullname_with_scope(), use_auto_tune_alg);
  if (ret != RET_OK) {
    return ret;
  }
  ret = UpdateTensorDataAndSize(parameter_node, weight, quant_data.data(), quant_data.size() * sizeof(int16_t),
                                kNumberTypeInt16);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UpdateTensorDataAndSize error";
    return RET_ERROR;
  }

  if (quant_params.empty()) {
    MS_LOG(ERROR) << "quant_params empty";
    return RET_ERROR;
  }
  auto quant_param_holder = GetCNodeQuantHolder(primitive);
  quant_param_holder->set_input_quant_param(index, quant_params);
  quant_param_holder->set_quant_type(quant_type);
  return ret;
}
}  // namespace mindspore::lite::quant
