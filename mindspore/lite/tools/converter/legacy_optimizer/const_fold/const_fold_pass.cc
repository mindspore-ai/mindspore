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

#include "tools/converter/legacy_optimizer/const_fold/const_fold_pass.h"
#include <vector>
#include "utils/log_adapter.h"
#include "converter/common/graph_util.h"

namespace mindspore {
namespace lite {
STATUS ConstFoldPass::Run(GraphNode *graphNode) {
  MS_ASSERT(graphNode != nullptr);
  auto subGraph = graphNode->subGraph;
  auto node = graphNode->opDef;
  MS_ASSERT(subGraph != nullptr);
  MS_ASSERT(node != nullptr);
  if (GetOpType(*node) != opType) {
    return RET_OK;
  }
  if (!IsFoldable(subGraph, node)) {
    MS_LOGD("All input should be ConstTensor, node : %s");
    return RET_OK;
  }

  for (uint32_t i : node->inputIndex) {
    TensorDefT *tensorDefT = subGraph->allTensors.at(i).get();
    MS_ASSERT(tensorDefT != nullptr);
    auto tensor = CopyTensorDefT2Tensor(tensorDefT);
    if (tensor == nullptr) {
      MS_LOGE("Pack TensorDefT return nullptr");
      FreeTensors();
      return RET_ERROR;
    }
    inputs.emplace_back(tensor);
  }
  for (uint32_t i : node->outputIndex) {
    TensorDefT *tensorDefT = subGraph->allTensors.at(i).get();
    MS_ASSERT(tensorDefT != nullptr);
    auto tensor = CopyTensorDefT2Tensor(tensorDefT, false);
    if (tensor == nullptr) {
      MS_LOGE("Pack TensorDefT return nullptr");
      FreeTensors();
      return RET_ERROR;
    }
    outputs.emplace_back(tensor);
  }

  auto status = CreateOp(subGraph, node);
  if (status != RET_OK) {
    MS_LOGE("CreateOp error: %d, node: %s", status, node->name.c_str());
    FreeTensors();
    return status;
  }
  for (auto &outputTensor : outputs) {
    auto statusTmp = outputTensor->MallocData();
    if (statusTmp != RET_OK) {
      MS_LOGE("OutTensor MallocData error: %d, nodeName: %s", statusTmp, node->name.c_str());
      FreeTensors();
      return RET_ERROR;
    }
  }
  status = DoFold(subGraph, node);
  if (status != RET_OK) {
    MS_LOGE("DoFold error: %d, node: %s", status, node->name.c_str());
    FreeTensors();
    return status;
  }

  if (this->outputTensor->data.empty()) {
    MS_LOGI("outputTensor's data has not been set, node : %s", node->name.c_str());
    FreeTensors();
    return RET_OK;
  }
  this->outputTensor->refCount = schema::NodeType_ValueNode;
  bool isSubNode = false;
  for (auto &inNode : subGraph->nodes) {
    if (inNode->name == node->name) {
      isSubNode = true;
      break;
    }
  }
  if (!isSubNode) {
    MS_LOGE("Node %s is not in subGraph %s", node->name.c_str(), subGraph->name.c_str());
    return RET_PARAM_INVALID;
  } else {
    status = RemoveTensor(subGraph, node->inputIndex);
    if (status != RET_OK) {
      MS_LOGE("RemoveTensor failed, node : %s", node->name.c_str());
      FreeTensors();
      return status;
    }
    // we can not erase nodes in iter loop, so just isolate the node
    node->inputIndex.clear();
    node->outputIndex.clear();
  }

  FreeTensors();
  return RET_OK;
}

OpDef *ConstFoldPass::PackOpDefT(const OpDefT *opDefT) {
  flatbuffers::FlatBufferBuilder builder(1024);
  auto offset = OpDef::Pack(builder, opDefT);
  builder.Finish(offset);
  auto buf = builder.GetBufferPointer();
  auto opDef = flatbuffers::GetRoot<mindspore::predict::OpDef>(buf);
  return const_cast<mindspore::predict::OpDef *>(opDef);
}

Tensor *ConstFoldPass::CopyTensorDefT2Tensor(const TensorDefT *tensorDefT, bool needCopyData) {
  if (tensorDefT == nullptr) {
    MS_LOGE("tensorDefT is null");
    return nullptr;
  }
  std::vector<int64_t> dims;
  for (size_t i = 0; i < tensorDefT->dims.size(); i++) {
    dims.emplace_back(tensorDefT->dims.at(i));
  }

  auto tensor = new (std::nothrow) Tensor(tensorDefT->dataType, dims, tensorDefT->format, nullptr);
  if (tensor == nullptr) {
    MS_LOGE("new tensor error");
    return nullptr;
  }
  if (needCopyData) {
    auto status = tensor->MallocData();
    if (status != RET_OK) {
      MS_LOGE("malloc tensor data error: %d", status);
      delete (tensor);
      return nullptr;
    }
    size_t dataLength = tensor->GetDataSize();
    status = ::memcpy_s(tensor->GetData(), dataLength, tensorDefT->data.data(), dataLength);
    if (status != 0) {
      MS_LOGE("memcpy_s error: %d", status);
      delete (tensor);
      return nullptr;
    }
  }
  return tensor;
}

STATUS ConstFoldPass::CopyTensor2TensorDefT(const Tensor *tensor, TensorDefT *tensorDefT) {
  MS_ASSERT(tensorDefT != nullptr);
  if (tensor == nullptr) {
    MS_LOGE("tensor is null");
    return RET_ERROR;
  }

  tensorDefT->dims.clear();
  for (size_t i = 0; i < tensor->GetNDim(); i++) {
    tensorDefT->dims.emplace_back(tensor->GetDims().at(i));
  }
  tensorDefT->dataType = tensor->GetDataType();
  tensorDefT->format = tensor->GetFormat();
  size_t dataLength = tensor->GetDataSize();
  tensorDefT->data.resize(dataLength);
  auto ret = ::memcpy_s(tensorDefT->data.data(), dataLength, tensor->GetData(), dataLength);
  if (ret != 0) {
    MS_LOGE("memcpy_s error: %d", ret);
    return RET_ERROR;
  }
  return RET_OK;
}

bool ConstFoldPass::IsFoldable(SubGraphDefT *subGraph, OpDefT *node) {
  bool isFoldable = true;
  for (auto tensorIdx : node->inputIndex) {
    auto &tensor = subGraph->allTensors.at(tensorIdx);
    if (tensor->refCount != schema::NodeType_ValueNode || tensor->data.empty()) {
      isFoldable = false;
      break;
    }
  }
  return isFoldable;
}

void ConstFoldPass::FreeTensors() {
  for (auto tensor : inputs) {
    if (tensor != nullptr) {
      delete (tensor);
    }
  }
  inputs.clear();
  for (auto tensor : outputs) {
    if (tensor != nullptr) {
      delete (tensor);
    }
  }
  outputs.clear();
}
}  // namespace lite
}  // namespace mindspore

