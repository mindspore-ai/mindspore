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
#include "include/errorcode.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "include/model.h"
#include "base/base.h"
#include "ir/primitive.h"
#include "abstract/dshape.h"
#include "mindspore/lite/tools/converter/quantizer/quantizer.h"

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

STATUS CalQuantizationParams(std::unique_ptr<AnfQuantParam> &quantParam, double mMin, double mMax,
                             bool narrowRange, int quant_max, int quant_min, int num_bits);

STATUS CalQuantizationParams(schema::QuantParamT *quantParam, double mMin, double mMax,
                             bool narrowRange = false, int numBits = UINT8_QUANTIZATION);

template <typename T>
T QuantizeData(const float originData, const schema::QuantParamT *quantParam) {
  MS_ASSERT(quantParam != nullptr);
  MS_ASSERT(quantParam->inited);
  const auto scale = quantParam->scale;
  const auto zeroPoint = quantParam->zeroPoint;
  const auto numBit = quantParam->numBits;
  const auto narrowRange = quantParam->narrowRange;
  const double maxLimit = static_cast<float>((1 << (unsigned int)numBit) - 1 - zeroPoint) * scale;
  double minLimit;
  if (narrowRange) {
    minLimit = static_cast<float>(1 - zeroPoint) * scale;
  } else {
    minLimit = static_cast<float>(0 - zeroPoint) * scale;
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
    auto quantData = static_cast<T>(std::round(tmp / scale + zeroPoint));
    if (quantData == 0 && narrowRange) {
      quantData++;
    }
    return quantData;
  }();
}

template <typename T>
T QuantizeData(float originData, const AnfQuantParam *quantParam, int quant_max, int quant_min) {
  MS_ASSERT(quantParam != nullptr);
  MS_ASSERT(quantParam->inited);
  const auto scale = quantParam->scale;
  const int zeroPoint = quantParam->zeroPoint;
  const auto narrowRange = quantParam->narrowRange;
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

void CalFakeNode(const AnfNodePtr &inTensor);

STATUS QuantFilter(ParamValueLitePtr &weightPtr, QuantType quantType, int quant_max, int quant_min,
                   size_t bitNum = UINT8_QUANTIZATION, bool per_channel = false);

STATUS PostBitPack(float *weights, size_t shapeSize, size_t bitNum = UINT8_QUANTIZATION);

}  // namespace quant
}  // namespace lite
}  // namespace mindspore
#endif
