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

#include <cfloat>
#include "src/common/utils.h"
#include "tools/common/tensor_util.h"
#include "tools/common/graph_util.h"

namespace mindspore::lite {
std::unique_ptr<QuantParamT> GetTensorQuantParam(const std::unique_ptr<TensorT> &tensor) {
  MS_ASSERT(tensor != nullptr);
  auto &quantParams = tensor->quantParams;
  if (!quantParams.empty()) {
    return std::move(CopyQuantParamT(quantParams.front()));
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
  return std::move(dstQuantParam);
}

std::unique_ptr<QuantParamT> CopyQuantParamArrayT(const std::unique_ptr<QuantParamT> &srcQuantParamArray) {
  MS_ASSERT(srcQuantParamArray != nullptr);
  auto dstQuantParamArrayT = std::unique_ptr<QuantParamT>(new (std::nothrow) QuantParamT());
  if (dstQuantParamArrayT == nullptr) {
    // MS_LOG(ERROR)("new dstQuantParamArrayT failed");
    return nullptr;
  }
  /*
  for (size_t i = 0; i < srcQuantParamArray->param.size(); i++) {
    auto &srcQuantParam = srcQuantParamArray->param.at(i);
    MS_ASSERT(srcQuantParam != nullptr);
    std::unique_ptr<QuantParamT> dstQuantParam(new (std::nothrow) QuantParamT());
    if (dstQuantParam == nullptr) {
      //MS_LOG(ERROR)("new dstQuantParam failed");
      dstQuantParamArrayT.release();
      return nullptr;
    }
    dstQuantParam->scale = srcQuantParam->scale;
    dstQuantParam->zeroPoint = srcQuantParam->zeroPoint;
    dstQuantParam->min = srcQuantParam->min;
    dstQuantParam->max = srcQuantParam->max;
    dstQuantParam->narrowRange = srcQuantParam->narrowRange;
    dstQuantParam->numBits = srcQuantParam->numBits;
    dstQuantParamArrayT->param.emplace_back(std::move(dstQuantParam));
  }
  */
  return std::move(dstQuantParamArrayT);
}

std::unique_ptr<QuantParamT> GetInTensorQuantParamArray(const MetaGraphT &graphT, size_t tensorIdx) {
  auto preNodeIdxes = GetLinkedPreIdx(graphT, tensorIdx);
  MS_ASSERT(preNodeIdxes.size() <= 1);
  if (preNodeIdxes.empty()) {
    // MS_LOGD("the %zuth tensor has no preNode", tensorIdx);
    return nullptr;
  }
  auto preNodeIdx = preNodeIdxes.front();
  MS_ASSERT(preNodeIdx < graphT.nodes.size());
  auto &preNode = graphT.nodes.at(preNodeIdx);
  MS_ASSERT(preNode != nullptr);
  MS_ASSERT(preNode->inputIndex.size() + preNode->outputIndex.size() == preNode->quantParam.size());
  /*
  for (size_t i = 0; i < preNode->outputIndex.size(); i++) {
    if (preNode->outputIndex.at(i) == tensorIdx) {
      auto &quantPArray = preNode->quantParam.at(preNode->inputIndex.size() + i);
      MS_ASSERT(quantPArray->param.size() == 1);  // only support prelayer
      MS_ASSERT(quantPArray->param.front() != nullptr);
      if (quantPArray->param.front()->min == FLT_MAX) {
        //MS_LOGD("the %zuth tensor's preNode's relative quantParam has not be inited", tensorIdx);
        return nullptr;
      } else {
        return std::move(CopyQuantParamArrayT(quantPArray));
      }
    }
  }
  */
  MS_ASSERT(false);
  return nullptr;
}

std::unique_ptr<QuantParamT> GetOutTensorQuantParamArray(const MetaGraphT &graphT, size_t tensorIdx) {
  auto postNodeIdxes = GetLinkedPostIdx(graphT, tensorIdx);
  if (postNodeIdxes.empty()) {
    // MS_LOGD("the %zuth tensor has no postNode", tensorIdx);
    return nullptr;
  }
  // find one postNode which can give valid quantParamArray
  for (auto postNodeIdx : postNodeIdxes) {
    MS_ASSERT(postNodeIdx < graphT.nodes.size());
    auto &postNode = graphT.nodes.at(postNodeIdx);
    MS_ASSERT(postNode != nullptr);
    MS_ASSERT(postNode->inputIndex.size() + postNode->outputIndex.size() == postNode->quantParam.size());
    /*
    for (size_t i = 0; i < postNode->inputIndex.size(); i++) {
      if (postNode->inputIndex.at(i) == tensorIdx) {
        auto &quantPArray = postNode->quantParam.at(i);
        MS_ASSERT(quantPArray->param.size() == 1);  // only support prelayer
        MS_ASSERT(quantPArray->param.front() != nullptr);
        // check if postNode has valid quantParam
        if (quantPArray->param.front()->min == FLT_MAX) {
          continue;
        }
        MS_ASSERT(graphT.allTensors.size() > postNode->inputIndex.at(i));
        auto &tensor = graphT.allTensors.at(postNode->inputIndex.at(i));
        MS_ASSERT(tensor != nullptr);
        if (tensor->refCount == schema::NodeType_ValueNode) {
          continue;
        }
        // find valid quantParam return
        auto paramArray = CopyQuantParamArrayT(quantPArray);
        if (paramArray == nullptr) {
          //MS_LOG(ERROR)("CopyQuantParamArrayT return nullptr");
          return nullptr;
        }
        return std::move(paramArray);
      }
    }*/
  }
  return nullptr;
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
    // MS_LOG(ERROR)("new TensorT failed");
    return nullptr;
  }
  newTensor->dims = oldTensor->dims;
  newTensor->format = oldTensor->format;
  newTensor->dataType = oldTensor->dataType;
  newTensor->refCount = oldTensor->refCount;
  newTensor->nodeType = oldTensor->nodeType;
  newTensor->data = oldTensor->data;
  if (!oldTensor->quantParams.empty()) {
    newTensor->quantParams.emplace_back(std::move(GetTensorQuantParam(oldTensor)));
  }
  return std::move(newTensor);
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
