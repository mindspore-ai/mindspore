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
#include <cmath>
#include <string>
#include <algorithm>
#include <memory>
#include <vector>
#include "src/ir/primitive_t_value.h"
#include "mindspore/lite/tools/converter/quantizer/quantize_util.h"
#include "mindspore/lite/tools/converter/quantizer/general_bitpacking.h"
#include "src/common/utils.h"
#include "abstract/abstract_value.h"
#include "securec/include/securec.h"

using std::string;
using std::vector;

namespace mindspore {
namespace lite {
namespace quant {
const std::array<std::string, 4> QuantStrategy::mConvTypes = {
  {"Conv2D", "DeConv2D", "DepthwiseConv2D", "DeDepthwiseConv2D"}};
const std::array<std::string, 4> QuantStrategy::mMulTypes = {{"Mul", "MatMul", "BatchMatMul", "FullConnection"}};

QuantStrategy::QuantStrategy(size_t weightSize, size_t convWeightQuantChannelThreshold)
    : mWeightSize(weightSize), mConvWeightQuantChannelThreshold(convWeightQuantChannelThreshold) {}

bool QuantStrategy::CanConvOpQuantized(const CNodePtr &node) const {
  size_t i = 0;
  for (i = 0; i < mConvTypes.size(); i++) {
    if (node->fullname_with_scope().find(mConvTypes[i]) == 0) {
      break;
    }
  }

  if ((i == mConvTypes.size()) || (node->size() < 3)) {
    return false;
  }

  auto inputNode = node->input(2);
  if (!inputNode->isa<Parameter>()) {
    return false;
  }
  auto paramNode = inputNode->cast<ParameterPtr>();
  auto abstract_base = paramNode->abstract();
  if (abstract_base == nullptr) {
    return false;
  }

  if (!utils::isa<abstract::ShapePtr>(abstract_base->GetShapeTrack())) {
    MS_LOG(INFO) << "Shape of Abstract of parameter should be ShapePtr " << paramNode->name();
    return false;
  }
  auto weight_shape = utils::cast<abstract::ShapePtr>(abstract_base->GetShapeTrack())->shape();
  size_t shapeSize = 1;
  for (auto dim : weight_shape) {
    shapeSize = shapeSize * dim;
  }
  if (shapeSize < mWeightSize) {
    MS_LOG(INFO) << "shapeSize Invalid!" << shapeSize;
    return false;
  }
  if (weight_shape[0] <= mConvWeightQuantChannelThreshold) {
    MS_LOG(INFO) << "channel less mConvWeightQuantChannelThreshold!" << weight_shape[0];
    return false;
  }

  return true;
}

bool QuantStrategy::CanOpPostQuantized(AnfNodePtr &node) const {
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = std::dynamic_pointer_cast<CNode>(node);

  auto primitiveT_value = GetValueNode<std::shared_ptr<PrimitiveTValue>>(cnode->input(0));
  if (primitiveT_value == nullptr) {
    MS_LOG(WARNING) << "PrimitiveT_value is nullptr: " << cnode->fullname_with_scope();
    return false;
  }

  auto type = primitiveT_value->GetPrimitiveT()->value.type;
  MS_LOG(INFO) << "Primitive type: " << type;
  static const std::vector<schema::PrimitiveType> uint8OpList = {
    schema::PrimitiveType_Nchw2Nhwc, schema::PrimitiveType_Nhwc2Nchw,
    schema::PrimitiveType_Conv2D,    schema::PrimitiveType_DepthwiseConv2D,
    schema::PrimitiveType_Add,       schema::PrimitiveType_Pooling,
    schema::PrimitiveType_Concat,    /*schema::PrimitiveType_SoftMax,*/ schema::PrimitiveType_Reshape,
    schema::PrimitiveType_Activation};
  return IsContain(uint8OpList, type);
}

bool QuantStrategy::CanMulOpQuantized(const CNodePtr &node) const {
  size_t i = 0;
  for (i = 0; i < mMulTypes.size(); i++) {
    if (node->fullname_with_scope().find(mMulTypes[i]) == 0) {
      break;
    }
  }
  if (i == mMulTypes.size()) {
    return false;
  }

  if (node->size() < 3) {
    MS_LOG(INFO) << "input size less!";
    return false;
  }

  auto inputNode1 = node->input(1);
  auto inputNode2 = node->input(2);
  if (inputNode1 == nullptr || inputNode2 == nullptr) {
    MS_LOG(INFO) << "mul input is nullptr!";
    return false;
  }

  ParameterPtr paramNode = nullptr;
  if (inputNode1->isa<Parameter>()) {
    paramNode = inputNode1->cast<ParameterPtr>();
  } else if (inputNode2->isa<Parameter>()) {
    paramNode = inputNode2->cast<ParameterPtr>();
  }

  if (paramNode == nullptr) {
    MS_LOG(INFO) << "invalid paramNode!";
    return false;
  }

  auto abstract_base = paramNode->abstract();
  if (abstract_base == nullptr) {
    MS_LOG(INFO) << "abstract is nullptr";
    return false;
  }

  if (!utils::isa<abstract::ShapePtr>(abstract_base->GetShapeTrack())) {
    MS_LOG(INFO) << "Shape of Abstract of parameter should be ShapePtr " << paramNode->name();
    return false;
  }
  auto weight_shape = utils::cast<abstract::ShapePtr>(abstract_base->GetShapeTrack())->shape();
  size_t shapeSize = 1;
  for (auto dim : weight_shape) {
    shapeSize = shapeSize * dim;
  }
  if (shapeSize < mWeightSize) {
    MS_LOG(INFO) << "shapeSize Invalid!" << shapeSize;
    return false;
  }

  return true;
}

void CalFakeNode(const AnfNodePtr &inTensor) {
  // MS_ASSERT(inTensor != nullptr);
  // MS_ASSERT(inTensor->dataType == DataType_DT_FLOAT);
  // auto quantParam = GetTensorQuantParams(inTensor);
  // if (quantParam == nullptr || !quantParam->inited) {
  //   MS_LOGW("tensor quantParam has not been inited");
  //   return;
  // }

  // float quantMin = quantParam->narrowRange ? 1 : 0;
  // float quantMax = (1 << (unsigned int)(quantParam->numBits)) - 1;
  // const float scale = quantParam->scale;
  // const float nudgedMin = (quantMin - quantParam->zeroPoint) * scale;
  // const float nudgedMax = (quantMax - quantParam->zeroPoint) * scale;
  // // cal output
  // float invNudgeScale = 1.0f / scale;
  // void *inData = inTensor->data.data();
  // if(inData == nullptr) {
  //   MS_LOGE("null pointer dereferencing.");
  //   return;
  // }
  // auto *data = static_cast<float *>(inData);
  // for (size_t i = 0; i < GetShapeSize(*inTensor); i++) {
  //   float clamped = std::min(nudgedMax, std::max(nudgedMin, data[i]));
  //   float clampedShifted = clamped - nudgedMin;
  //   data[i] = std::round(clampedShifted * invNudgeScale) * scale + nudgedMin;
  // }
}

STATUS CalQuantizationParams(std::unique_ptr<AnfQuantParam> &quantParam, double mMin, double mMax, bool narrowRange,
                             int quant_max, int quant_min, int num_bits) {
  MS_ASSERT(quantParam != nullptr);
  if (mMin > 0.0f) {
    MS_LOG(ERROR) << "min " << mMin << " is bigger then 0, set to 0, this may course low precision";
    mMin = 0.0f;
  }
  if (mMax < 0.0f) {
    MS_LOG(ERROR) << "mMax " << mMax << " is smaller than 0, set to 0, this may course low precision";
    mMax = 0.0f;
  }
  if (mMin > mMax) {
    MS_LOG(ERROR) << "cal error while min" << mMin << ">" << mMax;
    return RET_PARAM_INVALID;
  }
  if (mMin == mMax) {
    if (mMin != 0.0f) {
      MS_LOG(ERROR) << "min and max should both be zero if they are equal to each other";
      return RET_ERROR;
    }
    quantParam->inited = true;
    quantParam->min = mMin;
    quantParam->max = mMax;
    quantParam->scale = 0.0f;
    quantParam->zeroPoint = 0;
    quantParam->narrowRange = narrowRange;
    quantParam->numBits = num_bits;
    return RET_OK;
  }

  auto quantMinFloat = static_cast<double>(quant_min);
  auto quantMaxFloat = static_cast<double>(quant_max);
  double scale = (mMax - mMin) / (quantMaxFloat - quantMinFloat);
  const double zeroPointFromMin = quantMinFloat - mMin / scale;
  // const double zeroPointFromMax = quantMaxFloat - mMax / scale;
  int zeroPoint = static_cast<int32_t>(std::round(zeroPointFromMin));

  // The zero point should always be in the range of quantized value,
  // [qmin, qmax].
  MS_ASSERT(zeroPoint >= quantMin);
  MS_ASSERT(zeroPoint <= quantMax);
  quantParam->inited = true;
  quantParam->min = mMin;
  quantParam->max = mMax;
  quantParam->scale = scale;
  quantParam->zeroPoint = zeroPoint;
  quantParam->narrowRange = narrowRange;
  quantParam->numBits = num_bits;

  return RET_OK;
}

STATUS CalQuantizationParams(schema::QuantParamT *quantParam, double mMin, double mMax,
                             bool narrowRange, int numBits) {
  MS_ASSERT(quantParam != nullptr);
  if (mMin > 0.0f) {
    MS_LOG(ERROR) << "min " << mMin << " is bigger then 0, set to 0, this may course low precision";
    mMin = 0.0f;
  }
  if (mMax < 0.0f) {
    MS_LOG(ERROR) << "mMax " << mMax << " is smaller than 0, set to 0, this may course low precision";
    mMax = 0.0f;
  }
  if (mMin > mMax) {
    MS_LOG(ERROR) << "cal error while min" << mMin << ">" << mMax;
    return RET_PARAM_INVALID;
  }
  if (mMin == mMax) {
    if (mMin != 0.0f) {
      MS_LOG(ERROR) << "min and max should both be zero if they are equal to each other";
      return RET_ERROR;
    }
    quantParam->inited = true;
    quantParam->min = mMin;
    quantParam->max = mMax;
    quantParam->scale = 0.0f;
    quantParam->zeroPoint = 0;
    quantParam->narrowRange = narrowRange;
    quantParam->numBits = numBits;
    return RET_OK;
  }

  int quantMin = narrowRange ? 1 : 0;
  int quantMax = (1 << (unsigned int)numBits) - 1;
  auto quantMinFloat = static_cast<double>(quantMin);
  auto quantMaxFloat = static_cast<double>(quantMax);
  double scale = (mMax - mMin) / (quantMaxFloat - quantMinFloat);
  const double zeroPointFromMin = quantMinFloat - mMin / scale;
  const double zeroPointFromMax = quantMaxFloat - mMax / scale;
  const double zpFromMinError = std::abs(quantMinFloat) + std::abs(mMin / scale);
  const double zpFromMaxError = std::abs(quantMaxFloat) + std::abs(mMax / scale);
  const double zpDouble = zpFromMinError < zpFromMaxError ? zeroPointFromMin : zeroPointFromMax;
  int zeroPoint;
  if (zpDouble < quantMinFloat) {
    zeroPoint = quantMin;
  } else if (zpDouble > quantMaxFloat) {
    zeroPoint = quantMax;
  } else {
    zeroPoint = static_cast<int32_t>(std::round(zpDouble));
  }
  // The zero point should always be in the range of quantized value,
  // [qmin, qmax].
  MS_ASSERT(zeroPoint >= quantMin);
  MS_ASSERT(zeroPoint <= quantMax);
  quantParam->inited = true;
  quantParam->min = mMin;
  quantParam->max = mMax;
  quantParam->scale = scale;
  quantParam->zeroPoint = zeroPoint;
  quantParam->narrowRange = narrowRange;
  quantParam->numBits = numBits;

  return RET_OK;
}

STATUS QuantFilter(ParamValueLitePtr &weightPtr, QuantType quantType, int quant_max, int quant_min, size_t bitNum,
                   bool per_channel) {
  auto dims = weightPtr->tensor_shape();
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

  if (per_channel) {
    // notice:
    // at now for tflite model, Conv2D's weight format is KHWC, so is DepthwiseConv2D
    // if TransWeightFormat is done before PostTraingingQuantization, the DepthwiseCon2D's weight is CHWK
    size_t shapeSize = weightPtr->tensor_shape_size();
    auto channels = dims[0];
    size_t oneFilterSize = shapeSize / channels;
    auto *rawDatas = reinterpret_cast<const float *>(weightPtr->tensor_addr());
    if (rawDatas == nullptr) {
      MS_LOG(ERROR) << "rawDatas is nullptr";
      return RET_ERROR;
    }

    float min = FLT_MAX;
    float max = -FLT_MAX;
    weightPtr->quant_param().clear();
    vector<int8_t> qDatas(shapeSize);

    for (uint32_t i = 0; i < channels; i++) {
      // find min and max
      for (uint32_t j = 0; j < oneFilterSize; j++) {
        auto index = j + i * channels;
        if (index >= shapeSize) {
          MS_LOG(ERROR) << "over flow!";
          return RET_ERROR;
        }
        min = std::min(min, rawDatas[index]);
        max = std::max(max, rawDatas[index]);
      }
      std::unique_ptr<AnfQuantParam> quantParam = std::unique_ptr<AnfQuantParam>(new AnfQuantParam);
      STATUS status = CalQuantizationParams(quantParam, min, max, false, quant_max, quant_min, bitNum);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "CalQuantizationParams failed" << status;
        return status;
      }
      // do quantization
      for (uint32_t j = 0; j < oneFilterSize; j++) {
        auto index = j + i * channels;
        if (index >= shapeSize) {
          MS_LOG(ERROR) << "over flow!";
          return RET_ERROR;
        }
        float rawData = rawDatas[index];
        auto qData = QuantizeData<int8_t>(rawData, quantParam.get(), quant_max, quant_min);
        qDatas[index] = qData;
      }
      weightPtr->set_quant_param(quantParam);
    }
    auto ret =
      memcpy_s(const_cast<float *>(rawDatas), weightPtr->tensor_size(), qDatas.data(), shapeSize * sizeof(int8_t));
    if (ret != EOK) {
      MS_LOG(ERROR) << "memcpy error: " << ret;
      return RET_ERROR;
    }
    if (quantType == QuantType_WeightQuant) {
      PostBitPack(const_cast<float *>(rawDatas), shapeSize, bitNum);
    }

    weightPtr->set_tensor_type(kNumberTypeInt8);
    weightPtr->set_tensor_size(shapeSize * sizeof(int8_t));
  } else {
    // per layer
    size_t shapeSize = weightPtr->tensor_shape_size();
    auto *rawDatas = static_cast<float *>(weightPtr->tensor_addr());
    if (rawDatas == nullptr) {
      MS_LOG(ERROR) << "rawDatas is nullptr";
      return RET_ERROR;
    }

    weightPtr->quant_param().clear();
    vector<int8_t> qDatas(shapeSize);

    float min = 0;
    float max = 0;
    for (uint32_t i = 0; i < shapeSize; i++) {
      // find max min
      min = std::min(min, rawDatas[i]);
      max = std::max(max, rawDatas[i]);
    }

    std::unique_ptr<AnfQuantParam> quantParam = std::unique_ptr<AnfQuantParam>(new AnfQuantParam);
    STATUS status = CalQuantizationParams(quantParam, min, max, false, quant_max, quant_min, bitNum);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "CalQuantizationParams failed" << status;
      return status;
    }
    // update data and datatype
    for (uint32_t i = 0; i < shapeSize; i++) {
      float rawData = rawDatas[i];
      auto quant_data = std::round(rawData / quantParam->scale + quantParam->zeroPoint);
      if (quant_data > quant_max) {
        qDatas[i] = quant_max;
      } else if (quant_data < quant_min) {
        qDatas[i] = quant_min;
      } else {
        qDatas[i] = static_cast<int8_t>(quant_data);
      }
    }

    weightPtr->set_quant_param(quantParam);
    auto ret = memcpy_s(rawDatas, weightPtr->tensor_size(), qDatas.data(), shapeSize * sizeof(int8_t));
    if (ret != EOK) {
      MS_LOG(ERROR) << "memcpy error: " << ret;
      return RET_ERROR;
    }
    if (quantType == QuantType_WeightQuant) {
      PostBitPack(rawDatas, shapeSize, bitNum);
    }

    weightPtr->set_tensor_type(kNumberTypeInt8);
    weightPtr->set_tensor_size(shapeSize * sizeof(int8_t));
  }

  return RET_OK;
}

STATUS PostBitPack(float *weight, size_t shapeSize, size_t bitNum) {
  auto *rawDatas = reinterpret_cast<uint8_t *>(weight);
  vector<uint8_t> qDatas(rawDatas, rawDatas + shapeSize);
  vector<uint8_t> qDatas_packed;
  if (bitNum < 8 && bitNum > 1) {
    BitPack weight_bitpack(bitNum);
    weight_bitpack.BitPacking(qDatas, qDatas_packed);
    if (EOK != memcpy_s(rawDatas, shapeSize, &qDatas_packed[0], shapeSize)) {
      MS_LOG(ERROR) << "PostBitPack memcpy_s qDatas_packed failed";
      return RET_ERROR;
    }
  } else if (bitNum == 8) {
    if (EOK != memcpy_s(rawDatas, shapeSize, &qDatas[0], shapeSize)) {
      MS_LOG(ERROR) << "PostBitPack memcpy_s qDatas failed";
      return RET_ERROR;
    }
  } else {
    MS_LOG(ERROR) << "bitNum must be between 0 and 8 : " << bitNum;
    return RET_ERROR;
  }

  return RET_OK;
}
}  // namespace quant
}  // namespace lite
}  // namespace mindspore
