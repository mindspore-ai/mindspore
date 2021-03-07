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

#include "tools/converter/legacy_optimizer/graph/dtype_trans_pass.h"
#include <string>
#include <set>
#include "tools/common/node_util.h"
#include "tools/converter/converter_context.h"
#include "src/common/common.h"
#include "src/common/utils.h"

namespace mindspore {
namespace lite {
#define kMinInputNum 1
#define kOutputNum 1

STATUS DTypeTransPass::Run(schema::MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);

  auto status = DoModelInputDTypeTrans(graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "DoModelInputDTypeTrans error: " << status;
    return status;
  }

  status = DoModelOutputDTypeTrans(graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "DoModelOutputDTypeTrans error: " << status;
    return status;
  }

  status = DoNodeInoutDTypeTrans(graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "DoNodeInoutDTypeTrans error: " << status;
    return status;
  }
  return RET_OK;
}

STATUS DTypeTransPass::DoModelInputDTypeTrans(schema::MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  auto &graphInIdxes = graph->inputIndex;
  if (this->inputDataDType != TypeId::kNumberTypeFloat32 && this->inputDataDType != TypeId::kNumberTypeUInt8 &&
      this->inputDataDType != TypeId::kNumberTypeInt8 && this->inputDataDType != TypeId::kTypeUnknown) {
    MS_LOG(ERROR) << "Invalid inputDataType: " << this->inputDataDType;
    return RET_ERROR;
  }
  for (auto graphInIdx : graphInIdxes) {
    MS_ASSERT(graphInIdx < graph->allTensors.size());
    auto &tensor = graph->allTensors.at(graphInIdx);
    if (tensor->quantParams.empty() || !tensor->quantParams.front()->inited) {
      continue;
    }
    int32_t tensorDataType = this->inputDataDType != TypeId::kTypeUnknown
                               ? this->inputDataDType
                               : TensorDataType::GetInstance()->GetTensorType(graphInIdx);
    for (auto iter = graph->nodes.begin(); iter != graph->nodes.end(); iter++) {
      auto nodeName = (*iter)->name;
      for (size_t inputIndexIdx = 0; inputIndexIdx < (*iter)->inputIndex.size(); inputIndexIdx++) {
        if ((*iter)->inputIndex.at(inputIndexIdx) == graphInIdx) {
          STATUS status = RET_OK;

          // insert dtype cast node between input tensor and input node
          if (tensorDataType != tensor->dataType && tensorDataType != kTypeUnknown) {
            iter = InsertDTypeTransNode(graph, iter, kBefore, inputIndexIdx, tensorDataType, tensor->dataType, &status);
          }

          if (status != RET_OK) {
            MS_LOG(ERROR) << "InsertDTypeTransNode before " << nodeName.c_str() << " failed";
            return status;
          }
        }
      }
    }
  }
  return RET_OK;
}

STATUS DTypeTransPass::DoModelOutputDTypeTrans(schema::MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  if (this->outputDataDType != TypeId::kNumberTypeFloat32 && this->outputDataDType != TypeId::kNumberTypeUInt8 &&
      this->outputDataDType != TypeId::kNumberTypeInt8 && this->outputDataDType != TypeId::kTypeUnknown) {
    MS_LOG(ERROR) << "Invalid outputDataType: " << this->outputDataDType;
    return RET_ERROR;
  }
  auto &graphOutIdxes = graph->outputIndex;
  for (auto graphOutIdx : graphOutIdxes) {
    MS_ASSERT(graphOutIdx < graph->allTensors.size());
    auto &tensor = graph->allTensors.at(graphOutIdx);
    if (tensor->quantParams.empty() || !tensor->quantParams.front()->inited) {
      continue;
    }
    int32_t tensorDataType = this->outputDataDType != TypeId::kTypeUnknown
                               ? this->outputDataDType
                               : TensorDataType::GetInstance()->GetTensorType(graphOutIdx);
    for (auto iter = graph->nodes.begin(); iter != graph->nodes.end(); iter++) {
      auto nodeName = (*iter)->name;
      MS_ASSERT(node != nullptr);
      for (size_t outputIndexIdx = 0; outputIndexIdx < (*iter)->outputIndex.size(); outputIndexIdx++) {
        if ((*iter)->outputIndex.at(outputIndexIdx) == graphOutIdx) {
          // insert transNode
          STATUS status = RET_OK;
          if (tensorDataType != tensor->dataType && tensorDataType != kTypeUnknown) {
            iter = InsertDTypeTransNode(graph, iter, kAfter, outputIndexIdx, tensor->dataType, tensorDataType, &status);
          }
          if (status != RET_OK) {
            MS_LOG(ERROR) << "InsertDTypeTransNode after " << nodeName.c_str() << " failed";
            return status;
          }
          break;
        }
      }
    }
  }
  return RET_OK;
}

STATUS DTypeTransPass::DoNodeInoutDTypeTrans(schema::MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  // insert transNode before and after existNode
  for (auto iter = graph->nodes.begin(); iter != graph->nodes.end(); iter++) {
    if (IsContain(GetInt8OpList(), GetCNodeTType(**iter)) || (*iter)->quantType != QuantType_AwareTraining) {
      continue;
    }
    auto nodeName = (*iter)->name;
    if ((*iter)->inputIndex.empty()) {
      MS_LOG(ERROR) << "Op " << nodeName.c_str() << " should have " << kMinInputNum << " input tensor at least";
      return RET_ERROR;
    }
    STATUS status;
    // insert pre
    for (size_t i = 0; i < (*iter)->inputIndex.size(); i++) {
      MS_ASSERT(graph->allTensors.size() > (*iter)->inputIndex.at(i));
      auto &preTensor = graph->allTensors.at((*iter)->inputIndex.at(i));
      if (preTensor->dataType != TypeId::kNumberTypeInt8) {
        continue;
      }
      iter = InsertDTypeTransNode(graph, iter, kBefore, i, kNumberTypeInt8, kNumberTypeFloat32, &status);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "InsertInt8ToFloat32Node before " << nodeName.c_str() << " failed";
        return RET_ERROR;
      }
    }

    // insert post
    for (size_t i = 0; i < (*iter)->outputIndex.size(); i++) {
      auto &postTensor = graph->allTensors.at((*iter)->outputIndex.at(i));
      if (postTensor->dataType != TypeId::kNumberTypeInt8) {
        continue;
      }
      iter = InsertDTypeTransNode(graph, iter, kAfter, i, kNumberTypeFloat32, kNumberTypeUInt8, &status);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "InsertFloat32ToUint8Node after " << nodeName.c_str() << " failed";
        return RET_ERROR;
      }
    }
    (*iter)->quantType = QuantType_QUANT_NONE;
  }

  return RET_OK;
}

NodeIter DTypeTransPass::InsertDTypeTransNode(schema::MetaGraphT *graph, NodeIter existNodeIter, InsertPlace place,
                                              size_t inoutIdx, int32_t inputDataType, int32_t outputDataType,
                                              STATUS *errorCode) {
  MS_ASSERT((*existNodeIter) != nullptr);
  auto existNodeName = (*existNodeIter)->name;
  std::string tileName;
  if (place == kBefore) {
    tileName = existNodeName + "_pre";
  } else {
    tileName = existNodeName + "_post";
  }
  auto transNode = std::unique_ptr<CNodeT>(new (std::nothrow) CNodeT);
  if (transNode == nullptr) {
    MS_LOG(ERROR) << "new TransNode failed";
    *errorCode = RET_ERROR;
    return graph->nodes.end();
  }
  auto quantDTypeCastParam = new (std::nothrow) QuantDTypeCastT;
  if (quantDTypeCastParam == nullptr) {
    MS_LOG(ERROR) << "new quantDTypeCastParam failed";
    *errorCode = RET_ERROR;
    return graph->nodes.end();
  }
  transNode->primitive = std::make_unique<schema::PrimitiveT>();
  transNode->primitive->value.value = quantDTypeCastParam;
  transNode->primitive->value.type = PrimitiveType_QuantDTypeCast;
  transNode->quantType = QuantType_AwareTraining;
  quantDTypeCastParam->src_t = inputDataType;
  quantDTypeCastParam->dst_t = outputDataType;
  if (inputDataType == TypeId::kNumberTypeInt8 && outputDataType == TypeId::kNumberTypeFloat32) {
    transNode->name = "int8toft32_" + tileName + std::to_string(id++);
  } else if (inputDataType == TypeId::kNumberTypeFloat32 && outputDataType == TypeId::kNumberTypeInt8) {
    transNode->name = "ft32toint8_" + tileName + std::to_string(id++);
  } else if (inputDataType == TypeId::kNumberTypeUInt8 && outputDataType == TypeId::kNumberTypeInt8) {
    transNode->name = "uint8toint8_" + tileName + std::to_string(id++);
  } else if (inputDataType == TypeId::kNumberTypeInt8 && outputDataType == TypeId::kNumberTypeUInt8) {
    transNode->name = "int8touint8_" + tileName + std::to_string(id++);
  }
  transNode->primitive->value.value = quantDTypeCastParam;
  int insert_num = 0;
  return InsertNode(graph, existNodeIter, place, inoutIdx, std::move(transNode), errorCode, &insert_num, castOpCopyer);
}

void DTypeTransPass::SetInputDataDType(TypeId dataType) { this->inputDataDType = dataType; }

void DTypeTransPass::SetOutputDataDType(TypeId dataType) { this->outputDataDType = dataType; }

}  // namespace lite
}  // namespace mindspore
