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
#include "src/ops/primitive_c.h"
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
  if (weight_shape[0] <= static_cast<int>(mConvWeightQuantChannelThreshold)) {
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

  auto primitiveT_value = GetValueNode<std::shared_ptr<PrimitiveC>>(cnode->input(0));
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
    schema::PrimitiveType_Concat,    /*schema::PrimitiveType_SoftMax,*/
    schema::PrimitiveType_Reshape,   schema::PrimitiveType_FullConnection,
    schema::PrimitiveType_MatMul,
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

STATUS CalQuantizationParams(schema::QuantParamT *quantParam, double mMin, double mMax, bool narrowRange,
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
  int quantMax = (1 << (unsigned int) numBits) - 1;
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

STATUS QuantFilter(ParamValueLitePtr weight, std::shared_ptr<PrimitiveC> primitiveT_value, QuantType quantType,
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

  vector<schema::QuantParamT> quant_params;
  size_t elem_count = weight->tensor_shape_size();
  auto *raw_datas = static_cast<float *>(weight->tensor_addr());
  if (raw_datas == nullptr) {
    MS_LOG(ERROR) << "rawDatas is nullptr";
    return RET_ERROR;
  }
  vector<int8_t> quant_datas(elem_count);

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
          auto quant_data = QuantizeData<int8_t>(raw_data, quant_param, quant_max, quant_min);
          quant_datas[index] = quant_data;
        }
      }
      auto ret = memcpy_s(const_cast<float *>(raw_datas), weight->tensor_size(), quant_datas.data(),
                          elem_count * sizeof(int8_t));
      if (ret != EOK) {
        MS_LOG(ERROR) << "memcpy error: " << ret;
        return RET_ERROR;
      }
      if (quantType == QuantType_WeightQuant) {
        PostBitPack(const_cast<float *>(raw_datas), elem_count, bitNum);
      }

      weight->set_tensor_size(elem_count * sizeof(int8_t));
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
        for (uint32_t j = 0; j < one_filter_size; j++) {
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
          auto quant_data = QuantizeData<int8_t>(raw_data, quant_param, quant_max, quant_min);
          quant_datas[index] = quant_data;
        }
      }
      auto ret =
          memcpy_s(raw_datas, weight->tensor_size(), quant_datas.data(), elem_count * sizeof(int8_t));
      if (ret != EOK) {
        MS_LOG(ERROR) << "memcpy error: " << ret;
        return RET_ERROR;
      }
      if (quantType == QuantType_WeightQuant) {
        PostBitPack(const_cast<float *>(raw_datas), elem_count, bitNum);
      }
      weight->set_tensor_size(elem_count * sizeof(int8_t));
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
      auto quant_data = QuantizeData<int8_t>(raw_data, quant_param, quant_max, quant_min);
      quant_datas[i] = quant_data;
    }
    auto ret = memcpy_s(raw_datas, weight->tensor_size(), quant_datas.data(), elem_count * sizeof(int8_t));
    if (ret != EOK) {
      MS_LOG(ERROR) << "memcpy error: " << ret;
      return RET_ERROR;
    }
    if (quantType == QuantType_WeightQuant) {
      PostBitPack(raw_datas, elem_count, bitNum);
    }
    weight->set_tensor_size(elem_count * sizeof(int8_t));
  }
  if (quant_params.empty()) {
    MS_LOG(ERROR) << "quant_params empty";
    return RET_ERROR;
  }
  primitiveT_value->AddInputQuantParam(quant_params);
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
