/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "src/common/utils.h"
#include "tools/common/tensor_util.h"
#include "tools/common/graph_util.h"

namespace mindspore::lite {
std::unique_ptr<QuantParamT> GetTensorQuantParam(const std::unique_ptr<TensorT> &tensor) {
  MS_ASSERT(tensor != nullptr);
  auto &quantParams = tensor->quantParams;
  if (!quantParams.empty()) {
    return CopyQuantParamT(quantParams.front());
  } else {
    return nullptr;
  }
}
std::unique_ptr<schema::QuantParamT> CopyQuantParamT(const std::unique_ptr<schema::QuantParamT> &srcQuantParam) {
  MS_ASSERT(srcQuantParam != nullptr);
  std::unique_ptr<schema::QuantParamT> dstQuantParam = std::make_unique<schema::QuantParamT>();
  dstQuantParam->inited = srcQuantParam->inited;
  dstQuantParam->scale = srcQuantParam->scale;
  dstQuantParam->zeroPoint = srcQuantParam->zeroPoint;
  dstQuantParam->min = srcQuantParam->min;
  dstQuantParam->max = srcQuantParam->max;
  dstQuantParam->narrowRange = srcQuantParam->narrowRange;
  dstQuantParam->numBits = srcQuantParam->numBits;
  dstQuantParam->dstDtype = srcQuantParam->dstDtype;
  dstQuantParam->multiplier = srcQuantParam->multiplier;
  return dstQuantParam;
}

size_t GetElementSize(const TensorT &tensor) { return GetElementSize(TypeId(tensor.dataType)); }

size_t GetElementSize(const TypeId &dataType) {
  switch (dataType) {
    case kNumberTypeUInt8:
      return sizeof(uint8_t);
    case kNumberTypeInt32:
      return sizeof(int32_t);
    case kNumberTypeFloat:
      return sizeof(float);
    case kNumberTypeInt16:
      return sizeof(int16_t);
    case kNumberTypeInt8:
      return sizeof(int8_t);
    case kNumberTypeUInt32:
      return sizeof(uint32_t);
    default:
      return sizeof(float);
  }
}

size_t GetShapeSize(const TensorT &tensor) {
  auto shape = tensor.dims;
  size_t shapeSize = 1;
  for (auto dim : shape) {
    shapeSize *= dim;
  }
  return shapeSize;
}

std::unique_ptr<TensorT> CopyTensorDefT(const std::unique_ptr<TensorT> &oldTensor) {
  auto newTensor = std::unique_ptr<TensorT>(new (std::nothrow) TensorT);
  if (newTensor == nullptr) {
    MS_LOG(ERROR) << "new TensorT failed";
    return nullptr;
  }
  newTensor->dims = oldTensor->dims;
  newTensor->format = oldTensor->format;
  newTensor->dataType = oldTensor->dataType;
  newTensor->refCount = oldTensor->refCount;
  newTensor->nodeType = oldTensor->nodeType;
  newTensor->data = oldTensor->data;
  if (!oldTensor->quantParams.empty()) {
    newTensor->quantParams.emplace_back(GetTensorQuantParam(oldTensor));
  }
  return newTensor;
}

size_t GetRefCount(MetaGraphT *graphT, uint32_t tensorIdx) {
  MS_ASSERT(graphT != nullptr);
  MS_ASSERT(graphT->allTensors.size() > tensorIdx);
  size_t refCount = 0;
  for (auto &node : graphT->nodes) {
    MS_ASSERT(node != nullptr);
    if (IsContain(node->inputIndex, tensorIdx)) {
      refCount++;
    }
  }
  return refCount;
}
size_t GetShapeSize(const std::vector<int32_t> &shape) {
  size_t shapeSize = 1;
  for (auto dim : shape) {
    shapeSize *= dim;
  }
  return shapeSize;
}
}  // namespace mindspore::lite
