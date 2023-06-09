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

#include "tools/converter/quantizer/fixed_bit_weight_quantization.h"
#include <algorithm>
#include "tools/common/tensor_util.h"
#include "abstract/abstract_value.h"

namespace mindspore::lite::quant {
int FixedBitWeightQuantization::QuantFilter(const AnfNodePtr &parameter_node, const tensor::TensorPtr &weight,
                                            const PrimitivePtr &primitive, quant::QuantType quant_type, int quant_max,
                                            int quant_min, size_t bit_num, WeightQuantType weight_quant_type,
                                            TypeId quant_data_type, int preferred_dim, bool symmetric,
                                            bool narrow_range, bool bias_correction) {
  if (quant_data_type == kNumberTypeInt8) {
    return FixedBitQuantFilter<int8_t>(parameter_node, weight, primitive, quant_type, quant_max, quant_min, bit_num,
                                       weight_quant_type, quant_data_type, preferred_dim, symmetric, narrow_range,
                                       bias_correction);
  } else if (quant_data_type == kNumberTypeInt16) {
    return FixedBitQuantFilter<int16_t>(parameter_node, weight, primitive, quant_type, quant_max, quant_min, bit_num,
                                        weight_quant_type, quant_data_type, preferred_dim, symmetric, narrow_range,
                                        bias_correction);
  } else {
    MS_LOG(ERROR) << quant_data_type << " dont support.";
    return RET_ERROR;
  }
}

int FixedBitWeightQuantization::QuantBias(const ParameterPtr &weight, const ParameterPtr &bias,
                                          const std::vector<schema::QuantParamT> &active_quant_params) {
  CHECK_NULL_RETURN(weight);
  CHECK_NULL_RETURN(bias);
  MS_CHECK_TRUE_MSG(weight->has_default(), RET_ERROR, "parameter node has no default param.");
  auto weight_param = weight->default_param()->cast<tensor::TensorPtr>();
  CHECK_NULL_RETURN(weight_param);
  MS_CHECK_TRUE_MSG(bias->has_default(), RET_ERROR, "parameter node has no default param.");
  auto bias_param = bias->default_param()->cast<tensor::TensorPtr>();
  CHECK_NULL_RETURN(bias_param);

  auto quantization_params = weight_param->quant_params();
  if (quantization_params.empty()) {
    MS_LOG(ERROR) << weight->fullname_with_scope() << " quantization params empty.";
    return RET_ERROR;
  }
  auto weight_quant_params = quant::ConvertQuantizationParamToQuantParamT(quantization_params.front());
  if (active_quant_params.empty()) {
    MS_LOG(ERROR) << "active quant params empty.";
    return RET_ERROR;
  }

  std::vector<double> input_scales;
  std::vector<double> filter_scales;
  std::vector<double> bias_scales;
  size_t sizeX = active_quant_params.size();
  for (size_t i = 0; i < sizeX; i++) {
    input_scales.emplace_back(active_quant_params[i].scale);
  }
  size_t sizeY = weight_quant_params.size();
  if (sizeX != sizeY) {
    if (sizeX > 1 && sizeY > 1) {
      MS_LOG(ERROR) << "input and filter's scale count cannot match!";
      return RET_ERROR;
    }
  }
  for (size_t i = 0; i < sizeY; i++) {
    filter_scales.emplace_back(weight_quant_params[i].scale);
  }
  size_t size = std::max(sizeX, sizeY);
  for (size_t i = 0; i < size; i++) {
    auto scaleX = sizeX > 1 ? input_scales[i] : input_scales[0];
    auto scaleY = sizeY > 1 ? filter_scales[i] : filter_scales[0];
    bias_scales.push_back(scaleX * scaleY);
  }
  MS_ASSERT(!bias_scales.empty());
  size_t shape_size = bias_param->DataSize();

  // set bias quant param
  std::vector<schema::QuantParamT> bias_quant_params;
  for (double bias_scale : bias_scales) {
    schema::QuantParamT quant_param;
    if (bias_scale == 0) {
      MS_LOG(WARNING) << "bias_scale is 0, and set bias_scale to 1.";
      quant_param.scale = 1;
    } else {
      quant_param.scale = bias_scale;
    }
    quant_param.numBits = k32Bit;
    quant_param.zeroPoint = 0;
    quant_param.inited = true;
    bias_quant_params.emplace_back(quant_param);
  }
  // quant bias data
  std::vector<int32_t> quant_datas(shape_size);
  auto *raw_datas = static_cast<float *>(bias_param->data_c());
  if (ComputeBiasDataAndQuantParam(bias_scales, input_scales, raw_datas, &weight_quant_params, weight_param,
                                   &bias_quant_params, &quant_datas) != RET_OK) {
    MS_LOG(ERROR) << "compute bias data failed.";
    return RET_ERROR;
  }
  auto bias_quantization_param = quant::ConvertQuantParamTToQuantizationParam(bias_quant_params);
  CHECK_NULL_RETURN(bias_quantization_param);
  bias_param->set_quant_param(std::vector<std::shared_ptr<mindspore::QuantizationParam>>{bias_quantization_param});
  auto ret =
    UpdateTensorDataAndSize(bias, bias_param, quant_datas.data(), shape_size * sizeof(int32_t), kNumberTypeInt32);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << bias->fullname_with_scope() << " update tensor data and size failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int FixedBitWeightQuantization::ComputeBiasDataAndQuantParam(
  const std::vector<double> &bias_scales, const std::vector<double> &input_scales, const float *raw_datas,
  std::vector<schema::QuantParamT> *weight_quant_params, const tensor::TensorPtr &weight,
  std::vector<schema::QuantParamT> *bias_quant_params, std::vector<int32_t> *quant_datas) {
  CHECK_NULL_RETURN(raw_datas);
  CHECK_NULL_RETURN(weight_quant_params);
  CHECK_NULL_RETURN(bias_quant_params);
  CHECK_NULL_RETURN(quant_datas);
  double bias_scale_tmp;
  const constexpr double quanted_bias_abs_limit = 0.5 * INT32_MAX;
  auto quant_parameter_size = quant_datas->size();
  if (bias_scales.size() == quant_parameter_size) {
    for (size_t i = 0; i < quant_parameter_size; i++) {
      bias_scale_tmp = bias_scales[i];
      if (fabs(bias_scale_tmp) <= 0.0f) {
        MS_LOG(ERROR) << "divisor 'bias_scale_tmp' cannot be 0.";
        return RET_ERROR;
      }
      if (std::abs(raw_datas[i] / bias_scale_tmp) >= quanted_bias_abs_limit) {
        // update filter scale and zp
        double activate_scale = input_scales[0];
        double filter_scale = std::abs(raw_datas[i]) / (activate_scale * quanted_bias_abs_limit);
        MS_LOG(WARNING) << "quant bias over flow"
                        << " ,activation scale is:" << input_scales[i] << " and weight scale will be update from "
                        << weight_quant_params->at(i).scale << " to " << filter_scale;
        weight_quant_params->at(i).scale = filter_scale;
        weight_quant_params->at(i).zeroPoint = 0;
        auto weight_quantization = quant::ConvertQuantParamTToQuantizationParam(*weight_quant_params);
        CHECK_NULL_RETURN(weight_quantization);
        weight->set_quant_param(std::vector<std::shared_ptr<mindspore::QuantizationParam>>{weight_quantization});
        bias_scale_tmp = std::abs(raw_datas[i]) / quanted_bias_abs_limit;
        bias_quant_params->at(i).scale = bias_scale_tmp;
        MS_LOG(DEBUG) << "new filter scale: " << filter_scale;
      }
      auto quant_data = static_cast<int32_t>(std::round(raw_datas[i] / bias_scale_tmp));
      quant_datas->at(i) = quant_data;
    }
    return RET_OK;
  } else if (bias_scales.size() == 1) {
    // for fc, per tensor quant
    bias_scale_tmp = bias_quant_params->front().scale;
    float max_raw_data = 0.0f;
    for (size_t i = 0; i < quant_parameter_size; i++) {
      if (std::abs(raw_datas[i]) > max_raw_data) {
        max_raw_data = std::abs(raw_datas[i]);
      }
    }
    if (fabs(bias_scale_tmp) <= 0.0f) {
      MS_LOG(ERROR) << "divisor 'bias_scale_tmp' cannot be 0.";
      return RET_ERROR;
    }
    if (std::abs(max_raw_data / bias_scale_tmp) >= quanted_bias_abs_limit) {
      MS_LOG(WARNING) << "quanted bias over flow, maybe the scale of weight: " << weight_quant_params->front().scale
                      << " is too small, need to update";
      double activate_scale = input_scales[0];
      MS_CHECK_TRUE_MSG(activate_scale != 0, RET_ERROR, "activate_scale == 0");
      double filter_scale = std::abs(max_raw_data) / (activate_scale * quanted_bias_abs_limit);
      weight_quant_params->at(0).scale = filter_scale;
      weight_quant_params->at(0).zeroPoint = 0;
      auto weight_quantization = quant::ConvertQuantParamTToQuantizationParam(*weight_quant_params);
      CHECK_NULL_RETURN(weight_quantization);
      weight->set_quant_param(std::vector<std::shared_ptr<mindspore::QuantizationParam>>{weight_quantization});
      bias_scale_tmp = max_raw_data / quanted_bias_abs_limit;
      bias_quant_params->front().scale = bias_scale_tmp;
      MS_LOG(DEBUG) << "new filter scale: " << filter_scale;
    }
    for (size_t i = 0; i < quant_parameter_size; i++) {
      auto quant_data = (int32_t)std::round(raw_datas[i] / bias_scale_tmp);
      quant_datas->at(i) = quant_data;
    }
    return RET_OK;
  }
  MS_LOG(ERROR) << "unexpected input_scales size: " << input_scales.size()
                << " weight_scales size: " << weight_quant_params->size();
  return RET_ERROR;
}
}  // namespace mindspore::lite::quant
