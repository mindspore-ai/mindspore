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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_FIXED_BIT_WEIGHT_QUANTIZATION_WITH_HOLDER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_FIXED_BIT_WEIGHT_QUANTIZATION_WITH_HOLDER_H

#include <vector>
#include "ir/tensor.h"
#include "schema/inner/model_generated.h"
#include "src/common/log_adapter.h"
#include "src/common/quant_utils.h"
#include "tools/converter/quantizer/quant_params.h"
#include "tools/converter/quantizer/quantize_util.h"

namespace mindspore::lite::quant {
class FixedBitWeightQuantizationWithHolder {
 public:
  FixedBitWeightQuantizationWithHolder() = default;

  ~FixedBitWeightQuantizationWithHolder() = default;

  int QuantFilter(const AnfNodePtr &parameter_node, const tensor::TensorPtr &weight, const PrimitivePtr &primitive,
                  quant::QuantType quant_type, int quant_max, int quant_min, size_t bit_num,
                  WeightQuantType weight_quant_type, TypeId quant_data_type, int index, int preferred_dim,
                  bool symmetric = false, bool narrow_range = false, bool bias_correction = true);

  int StatisticsFilter(const tensor::TensorPtr &weight, const PrimitivePtr &primitive, quant::QuantType quant_type,
                       int quant_max, int quant_min, size_t bit_num, WeightQuantType weight_quant_type,
                       TypeId quant_data_type, int index, int preferred_dim, bool symmetric = false,
                       bool narrow_range = false);

  int QuantBias(const ParameterPtr &bias, const PrimitivePtr &primitive);

 private:
  int ComputeBiasDataAndQuantParam(const std::vector<double> &bias_scales, const std::vector<double> &input_scales,
                                   const float *raw_datas, const QuantParamHolderPtr &quant_param_holder,
                                   std::vector<schema::QuantParamT> *quant_params, std::vector<int32_t> *quant_datas);

  template <typename T>
  int FixedBitQuantFilter(const AnfNodePtr &parameter_node, const tensor::TensorPtr &weight,
                          const PrimitivePtr &primitive, quant::QuantType quant_type, int quant_max, int quant_min,
                          size_t bit_num, WeightQuantType weight_quant_type, TypeId quant_data_type, int index,
                          int preferred_dim, bool symmetric = false, bool narrow_range = false,
                          bool bias_correction = true) {
    size_t elem_count = weight->DataSize();
    auto *raw_data = static_cast<float *>(weight->data_c());
    if (raw_data == nullptr) {
      MS_LOG(ERROR) << "rawDatas is nullptr";
      return RET_ERROR;
    }
    std::vector<T> quant_data(elem_count);
    auto status =
      FixedBitStatisticsFilter<T>(weight, primitive, quant_type, quant_max, quant_min, bit_num, weight_quant_type,
                                  index, preferred_dim, &quant_data, symmetric, narrow_range, bias_correction);
    if (status == RET_NO_CHANGE) {
      return status;
    } else if (status != RET_OK) {
      MS_LOG(ERROR) << "FixedBitStatisticsFilter failed : " << status;
      return status;
    }
    status = UpdateTensorDataAndSize(parameter_node, weight, quant_data.data(), quant_data.size() * sizeof(T),
                                     quant_data_type);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "UpdateTensorDataAndSize error";
      return RET_ERROR;
    }
    auto quant_param_holder = GetCNodeQuantHolder(primitive);
    CHECK_NULL_RETURN(quant_param_holder);
    quant_param_holder->set_quant_type(quant_type);
    return RET_OK;
  }

  template <typename T>
  int FixedBitStatisticsFilter(const tensor::TensorPtr &weight, const PrimitivePtr &primitive,
                               quant::QuantType quant_type, int quant_max, int quant_min, size_t bit_num,
                               WeightQuantType weight_quant_type, int index, int preferred_dim,
                               std::vector<T> *quant_data, bool symmetric = false, bool narrow_range = false,
                               bool bias_correction = true) {
    MS_ASSERT(weight != nullptr);
    MS_ASSERT(primitive != nullptr);
    auto dims = weight->shape();
    if (weight_quant_type == FIXED_BIT_PER_CHANNEL) {
      if (dims.size() <= 1) {
        MS_LOG(WARNING) << "dims is " << dims.size() << " can not per_channel";
        weight_quant_type = FIXED_BIT_PER_LAYER;
      }
    }
    if (weight->data_type_c() != kNumberTypeFloat32) {
      MS_LOG(ERROR) << "data type is not Float32.";
      return RET_ERROR;
    }

    std::vector<schema::QuantParamT> quant_params;
    int ret = RET_OK;
    bool cal_gain = (quant_type == QUANT_WEIGHT) && bias_correction ? true : false;
    if (weight_quant_type == FIXED_BIT_PER_CHANNEL) {
      ret = DoPerChannelQuant<T>(static_cast<float *>(weight->data_c()), weight->DataSize(), &quant_params, quant_max,
                                 quant_min, bit_num, quant_data, ConvertShapeVectorToInt32(dims), preferred_dim,
                                 cal_gain, symmetric, narrow_range);
      if (ret == RET_NO_CHANGE) {
        return ret;
      } else if (ret != RET_OK) {
        MS_LOG(ERROR) << "Do per channel quant failed.";
        return ret;
      }
    } else if (weight_quant_type == FIXED_BIT_PER_LAYER) {
      ret = DoPerLayerQuant<T>(static_cast<float *>(weight->data_c()), weight->DataSize(), &quant_params, quant_max,
                               quant_min, bit_num, quant_data, symmetric, narrow_range, cal_gain);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Do per layer quant failed.";
        return ret;
      }
    } else {
      MS_LOG(ERROR) << "Unsupported weight quant type:" << weight_quant_type;
      return RET_ERROR;
    }
    if (quant_params.empty()) {
      MS_LOG(ERROR) << "quant_params empty";
      return RET_ERROR;
    }
    auto quant_param_holder = GetCNodeQuantHolder(primitive);
    CHECK_NULL_RETURN(quant_param_holder);
    quant_param_holder->set_input_quant_param(index, quant_params);
    return ret;
  }
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_FIXED_BIT_WEIGHT_QUANTIZATION_WITH_HOLDER_H
