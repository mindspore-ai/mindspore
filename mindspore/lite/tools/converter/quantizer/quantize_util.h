/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef QUANTIZER_UTIL_H
#define QUANTIZER_UTIL_H

#include <memory>
#include <string>
#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <limits>
#include "tools/converter/quantizer/quantizer.h"
#include "src/ops/primitive_c.h"
#include "include/errorcode.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "include/model.h"
#include "base/base.h"
#include "ir/primitive.h"
#include "abstract/dshape.h"

namespace mindspore {
namespace lite {
namespace quant {
static constexpr size_t UINT8_QUANTIZATION = 8;

/**
 * 1. when op's weight size > mWeightSize just skip
 * 2. only do conv/deconv/convdepthwise/deconvdepthwise/mul/matmul/batchmatmul quantization
 * 3. when conv/deconv/convdepthwise/deconvdepthwise ops' weight channel size > covWeightQuantChannelThreshold just skip
 * */
class QuantStrategy {
 public:
  explicit QuantStrategy(size_t weightSize, size_t covWeightQuantChannelThreshold = 16);

  ~QuantStrategy() = default;

  bool CanConvOpQuantized(const CNodePtr &node) const;
  bool CanMulOpQuantized(const CNodePtr &node) const;
  bool CanOpPostQuantized(AnfNodePtr &node) const;

 private:
  size_t mWeightSize;
  size_t mConvWeightQuantChannelThreshold;

  static const std::array<std::string, 4> mConvTypes;
  static const std::array<std::string, 4> mMulTypes;
};

STATUS CalQuantizationParams(schema::QuantParamT *quantParam, double mMin, double mMax, bool narrowRange, int quant_max,
                             int quant_min, int num_bits);

STATUS CalQuantizationParams(schema::QuantParamT *quantParam, double mMin, double mMax, bool narrowRange = false,
                             int numBits = UINT8_QUANTIZATION);

template <typename T>
T QuantizeData(const float originData, const schema::QuantParamT *quantParam) {
  MS_ASSERT(quantParam != nullptr);
  MS_ASSERT(quantParam->inited);
  const auto scale = quantParam->scale;
  const auto zeroPoint = quantParam->zeroPoint;
  const auto numBit = quantParam->numBits;
  const auto narrowRange = quantParam->narrowRange;
  double maxLimitTemp = static_cast<float>((1 << (unsigned int)numBit) - 1);
  const double maxLimit = static_cast<float>(maxLimitTemp - zeroPoint + std::numeric_limits<int8_t>::min()) * scale;
  double minLimit;
  if (narrowRange) {
    minLimit = static_cast<float>(std::numeric_limits<int8_t>::min() + 1 - zeroPoint) * scale;
  } else {
    minLimit = static_cast<float>(std::numeric_limits<int8_t>::min() - zeroPoint) * scale;
  }

  return [maxLimit, minLimit, zeroPoint, scale, narrowRange, originData] {
    double tmp = 0.0f;
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
T QuantizeData(float originData, const schema::QuantParamT &quantParam, int quant_max, int quant_min) {
  MS_ASSERT(quantParam != nullptr);
  MS_ASSERT(quantParam->inited);
  const auto scale = quantParam.scale;
  const int zeroPoint = quantParam.zeroPoint;
  const auto narrowRange = quantParam.narrowRange;
  const int maxLimit = quant_max;
  const int minLimit = quant_min;

  return [maxLimit, minLimit, zeroPoint, scale, narrowRange, originData] {
    int quant_data = std::round(originData / scale + zeroPoint);
    if (quant_data > maxLimit) {
      quant_data = maxLimit;
    } else if (quant_data < minLimit) {
      quant_data = minLimit;
    }
    return static_cast<T>(quant_data);
  }();
}
template <typename T>
STATUS QuantFilter(ParamValueLitePtr weight, std::shared_ptr<PrimitiveC> primitive_c, QuantType quantType,
                   int quant_max, int quant_min, size_t bitNum, bool per_channel, bool depth_wise) {
  auto dims = weight->tensor_shape();
  if (per_channel) {
    if (dims.size() != 4) {
      MS_LOG(ERROR) << "weight dims size error: " << dims.size() << " Back to per layer.";
      per_channel = false;
    } else {
      uint32_t channels = dims[0];
      if (channels == 0) {
        MS_LOG(ERROR) << "channels is 0";
        return RET_ERROR;
      }
    }
  }

  std::vector<schema::QuantParamT> quant_params;
  size_t elem_count = weight->tensor_shape_size();
  auto *raw_datas = static_cast<float *>(weight->tensor_addr());
  if (raw_datas == nullptr) {
    MS_LOG(ERROR) << "rawDatas is nullptr";
    return RET_ERROR;
  }
  std::vector<T> quant_datas(elem_count);

  if (per_channel) {
    // notice:
    // at now for tflite model, Conv2D's weight format is KHWC, so is DepthwiseConv2D
    // if TransWeightFormat is done before PostTraingingQuantization, the DepthwiseCon2D's weight is CHWK
    if (depth_wise) {
      // channel at last
      auto channels = dims[3];
      if (channels == 0) {
        MS_LOG(ERROR) << "channels is zero";
        return RET_ERROR;
      }
      size_t one_filter_size = elem_count / channels;

      for (int i = 0; i < channels; i++) {
        float min = FLT_MAX;
        float max = -FLT_MAX;
        // find min and max
        for (size_t j = 0; j < one_filter_size; j++) {
          auto index = i + j * channels;
          if (index >= elem_count) {
            MS_LOG(ERROR) << "over flow!";
            return RET_ERROR;
          }
          min = std::min(min, raw_datas[index]);
          max = std::max(max, raw_datas[index]);
        }
        schema::QuantParamT quant_param;
        STATUS status = CalQuantizationParams(&quant_param, min, max, false, quant_max, quant_min, bitNum);
        if (status != RET_OK) {
          MS_LOG(ERROR) << "CalQuantizationParams failed" << status;
          return status;
        }
        quant_params.emplace_back(quant_param);
        // do quantization
        for (uint32_t j = 0; j < one_filter_size; j++) {
          auto index = i + j * channels;
          if (index >= elem_count) {
            MS_LOG(ERROR) << "over flow!";
            return RET_ERROR;
          }
          float raw_data = raw_datas[index];
          auto quant_data = QuantizeData<T>(raw_data, quant_param, quant_max, quant_min);
          quant_datas[index] = quant_data;
        }
      }
      auto ret = memcpy_s(raw_datas, weight->tensor_size(), quant_datas.data(), elem_count * sizeof(T));
      if (ret != EOK) {
        MS_LOG(ERROR) << "memcpy error: " << ret;
        return RET_ERROR;
      }
      weight->set_tensor_size(elem_count * sizeof(T));
    } else {
      // channel at first
      auto channels = dims[0];
      if (channels == 0) {
        MS_LOG(ERROR) << "channels is zero";
        return RET_ERROR;
      }
      size_t one_filter_size = elem_count / channels;

      for (int i = 0; i < channels; i++) {
        float min = FLT_MAX;
        float max = -FLT_MAX;
        // find min and max
        for (size_t j = 0; j < one_filter_size; j++) {
          auto index = j + i * one_filter_size;
          if (index >= elem_count) {
            MS_LOG(ERROR) << "over flow!";
            return RET_ERROR;
          }
          min = std::min(min, raw_datas[index]);
          max = std::max(max, raw_datas[index]);
        }
        schema::QuantParamT quant_param;
        STATUS status = CalQuantizationParams(&quant_param, min, max, false, quant_max, quant_min, bitNum);
        if (status != RET_OK) {
          MS_LOG(ERROR) << "CalQuantizationParams failed" << status;
          return status;
        }
        quant_params.emplace_back(quant_param);
        // do quantization
        for (uint32_t j = 0; j < one_filter_size; j++) {
          auto index = j + i * one_filter_size;
          if (index >= elem_count) {
            MS_LOG(ERROR) << "over flow!";
            return RET_ERROR;
          }
          float raw_data = raw_datas[index];
          auto quant_data = QuantizeData<T>(raw_data, quant_param, quant_max, quant_min);
          quant_datas[index] = quant_data;
        }
      }
      auto ret = memcpy_s(raw_datas, weight->tensor_size(), quant_datas.data(), elem_count * sizeof(int8_t));
      if (ret != EOK) {
        MS_LOG(ERROR) << "memcpy error: " << ret;
        return RET_ERROR;
      }
      weight->set_tensor_size(elem_count * sizeof(T));
    }
  } else {
    // per layer
    float min = FLT_MAX;
    float max = -FLT_MIN;
    for (uint32_t i = 0; i < elem_count; i++) {
      // find max min
      min = std::min(min, raw_datas[i]);
      max = std::max(max, raw_datas[i]);
    }

    schema::QuantParamT quant_param;
    STATUS status = CalQuantizationParams(&quant_param, min, max, false, quant_max, quant_min, bitNum);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "CalQuantizationParams failed" << status;
      return status;
    }
    quant_params.emplace_back(quant_param);
    // update data and datatype
    for (uint32_t i = 0; i < elem_count; i++) {
      float raw_data = raw_datas[i];
      auto quant_data = QuantizeData<T>(raw_data, quant_param, quant_max, quant_min);
      quant_datas[i] = quant_data;
    }
    auto ret = memcpy_s(raw_datas, weight->tensor_size(), quant_datas.data(), elem_count * sizeof(int8_t));
    if (ret != EOK) {
      MS_LOG(ERROR) << "memcpy error: " << ret;
      return RET_ERROR;
    }
    weight->set_tensor_size(elem_count * sizeof(T));
  }
  if (quant_params.empty()) {
    MS_LOG(ERROR) << "quant_params empty";
    return RET_ERROR;
  }
  primitive_c->AddInputQuantParam(quant_params);
  return RET_OK;
}

STATUS PostBitPack(float *weights, size_t shapeSize, size_t bitNum = UINT8_QUANTIZATION);
}  // namespace quant
}  // namespace lite
}  // namespace mindspore
#endif
