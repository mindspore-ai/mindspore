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

#include <algorithm>
#include <string>
#include <memory>
#include <utility>
#include "tools/converter/legacy_optimizer/graph/format_trans_pass.h"
#include "tools/common/node_util.h"
#include "src/common/log_adapter.h"
#include "src/common/common.h"
#include "src/common/utils.h"

namespace mindspore {
namespace lite {
#define kMinInputNum 1
#define kOutputNum 1

STATUS FormatTransPass::Run(schema::MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  auto status = DoModelInputFormatTrans(graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "DoModelInputFormatTrans failed : " << status;
    return status;
  }
  status = DoNodeInoutFormatTrans(graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "DoNodeInoutFormatTrans failed : " << status;
    return status;
  }
  return RET_OK;
}

STATUS FormatTransPass::GetInsertFormatTrans(const schema::CNodeT &node, FormatTransNodeType *beforeNodeType,
                                             FormatTransNodeType *afterNodeType) {
  if (fmkType == converter::FmkType_TFLITE) {  // inference by nhwc
    return RET_NO_CHANGE;
  } else if (fmkType == converter::FmkType_CAFFE) {  // inference by nchw
    if (!IsContain(GetNhwcOpList(), GetCNodeTType(node))) {
      return RET_NO_CHANGE;
    }
    *beforeNodeType = kNCHW2NHWC;
    *afterNodeType = kNHWC2NCHW;
    return RET_OK;
  } else if (fmkType == converter::FmkType_MS) {
    if (!IsContain(GetNhwcOpList(), GetCNodeTType(node))) {
      return RET_NO_CHANGE;
    }
    *beforeNodeType = kNCHW2NHWC;
    *afterNodeType = kNHWC2NCHW;
    return RET_OK;
  } else if (fmkType == converter::FmkType_ONNX) {
    if (!IsContain(GetNhwcOpList(), GetCNodeTType(node))) {
      return RET_NO_CHANGE;
    }
    *beforeNodeType = kNCHW2NHWC;
    *afterNodeType = kNHWC2NCHW;
    return RET_OK;
  } else if (fmkType == converter::FmkType_TF) {
    if (IsContain(GetNhwcOpList(), GetCNodeTType(node)) && GetFormat(node) == schema::Format_NCHW) {
      *beforeNodeType = kNCHW2NHWC;
      *afterNodeType = kNHWC2NCHW;
      return RET_OK;
    }
    return RET_NO_CHANGE;
  } else {
    MS_LOG(ERROR) << "Unsupported fmk: " << fmkType;
    return RET_ERROR;
  }
}

STATUS FormatTransPass::DoModelInputFormatTrans(schema::MetaGraphT *graph) {
  if (fmkType == converter::FmkType_TF || fmkType == converter::FmkType_TFLITE) {
    return RET_OK;
  }
  MS_ASSERT(graph != nullptr);
  // insert trans node in model input tensor
  if (graph->nodes.empty()) {
    return RET_OK;
  }
  // onnx input format may be nhwc
  if (fmkType == converter::FmkType_ONNX && graph->inputIndex.size() == 1) {
    auto &input_tensor = graph->allTensors.at(graph->inputIndex[0]);
    auto &input_dims = input_tensor->dims;
    if (input_dims.size() == 4 && input_dims[3] != -1 && input_dims[1] == -1) {
      return RET_OK;
    }
  }
  auto graphInputIdxes = graph->inputIndex;
  for (size_t i = 0; i < graphInputIdxes.size(); i++) {
    bool transed = false;
    auto inputIdx = graphInputIdxes.at(i);
    MS_ASSERT(inputIdx < subGraph->allTensors.size());
    auto &tensor = graph->allTensors.at(inputIdx);
    if (tensor->dims.size() != kNCHWDimNumber) {
      continue;
    }

    for (auto iter = graph->nodes.begin(); iter != graph->nodes.end(); iter++) {
      for (size_t inputIndexIdx = 0; inputIndexIdx < (*iter)->inputIndex.size(); inputIndexIdx++) {
        if ((*iter)->inputIndex.at(inputIndexIdx) == inputIdx) {
          STATUS status = RET_OK;
          iter = InsertFormatTransNode(graph, iter, kBefore, inputIndexIdx, kNHWC2NCHW, &status);
          if (status != RET_OK) {
            MS_LOG(ERROR) << "InsertNhwc2NchwNode before " << (*iter)->name << " failed";
            return status;
          }
          // set first tensor format to nhwc
          auto &transNode = *(iter - 1);
          MS_ASSERT(transNode != nullptr);
          MS_ASSERT(transNode->inputIndex.size() == 1);
          MS_ASSERT(subGraph->allTensors.size() > transNode->inputIndex.front());
          auto &graphInTensor = graph->allTensors.at(transNode->inputIndex.front());
          graphInTensor->format = schema::Format::Format_NHWC;
          // assume parser not reformat shape
          auto oldDims = graphInTensor->dims;
          if (!transed) {
            graphInTensor->dims = {oldDims[NCHW_N], oldDims[NCHW_H], oldDims[NCHW_W], oldDims[NCHW_C]};
            transed = true;
          }
        }
      }
    }
  }
  return RET_OK;
}

// inference needed inputFormat:
//           conv     deconv     depth     dedepth
// fp32      NCHW     NCHW       NCHW      NCHW
// uint8     NCHW      ?         NCHW        ?
STATUS FormatTransPass::DoNodeInoutFormatTrans(schema::MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  // insert before and after the op cal by nchw/nc4hw4
  for (auto iter = graph->nodes.begin(); iter != graph->nodes.end(); iter++) {
    FormatTransNodeType beforeNodeType = kNCHW2NHWC;
    FormatTransNodeType afterNodeType = kNHWC2NCHW;
    STATUS status = RET_OK;
    status = GetInsertFormatTrans(**iter, &beforeNodeType, &afterNodeType);
    if (status == RET_NO_CHANGE) {
      continue;
    }
    if (status != RET_OK) {
      return status;
    }
    auto &node = *iter;
    auto nodeName = node->name;
    if (node->inputIndex.size() < kMinInputNum) {
      MS_LOG(ERROR) << "Op should have " << kMinInputNum << " input tensor at least";
      return RET_ERROR;
    }
    if (node->outputIndex.size() < kOutputNum) {
      MS_LOG(ERROR) << "Op should have " << kOutputNum << " output tensor";
      return RET_ERROR;
    }
    void *attr = node->primitive->value.value;
    if (node->primitive->value.type == schema::PrimitiveType_SpaceToDepth) {
      reinterpret_cast<schema::SpaceToDepthT *>(attr)->format = schema::Format_NHWC;
    }
    if (node->primitive->value.type == schema::PrimitiveType_DepthToSpace) {
      reinterpret_cast<schema::DepthToSpaceT *>(attr)->format = schema::Format_NHWC;
    }
#ifdef SUPPORT_TRAIN
    if (IsContain(GetNhwcAllInputOpList(), GetCNodeTType(**iter))) {
      int idx_num = node->inputIndex.size();
      if (GetCNodeTType(**iter) == schema::PrimitiveType_BNGrad) idx_num = 2;
      for (int i = 0; i < idx_num; i++) {
        iter = InsertFormatTransNode(graph, iter, kBefore, i, beforeNodeType, &status);
        if (status != RET_OK) {
          MS_LOG(ERROR) << "InsertNchw2NhwcNode before " << nodeName << "failed";
          return RET_ERROR;
        }
      }
    } else {
      int idx = 0;
      if (GetCNodeTType(**iter) == schema::PrimitiveType_ApplyMomentum) idx = 3;
      if (GetCNodeTType(**iter) == schema::PrimitiveType_Sgd) idx = 1;
      if (GetCNodeTType(**iter) == schema::PrimitiveType_Adam) idx = 9;
      iter = InsertFormatTransNode(graph, iter, kBefore, idx, beforeNodeType, &status);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "InsertNhwc2NchwNode after " << nodeName << "failed";
        return RET_ERROR;
      }
    }
#else
    iter = InsertFormatTransNode(graph, iter, kBefore, 0, beforeNodeType, &status);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "InsertNhwc2NchwNode after " << nodeName << "failed";
      return RET_ERROR;
    }
#endif
    iter = InsertFormatTransNode(graph, iter, kAfter, 0, afterNodeType, &status);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "InsertNhwc2NchwNode after " << nodeName << "failed";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

NodeIter FormatTransPass::InsertFormatTransNode(schema::MetaGraphT *graph, NodeIter existNodeIter, InsertPlace place,
                                                size_t inoutIdx, FormatTransNodeType nodeType, STATUS *errorCode) {
  MS_ASSERT((*existNodeIter) != nullptr);
  MS_ASSERT(graph != nullptr);
  auto existNodeName = (*existNodeIter)->name;
  std::string tileName;
  if (place == kBefore) {
    tileName = existNodeName + "_pre";
  } else {
    tileName = existNodeName + "_post";
  }
  auto transNode = std::make_unique<schema::CNodeT>();
  transNode->primitive = std::make_unique<schema::PrimitiveT>();
  transNode->primitive->value.type = schema::PrimitiveType_Transpose;
  auto attr = new (std::nothrow) schema::TransposeT();

  if (nodeType == kNCHW2NHWC) {
    transNode->name = "nchw2nhwc_" + tileName + std::to_string(id++);
    attr->perm = {0, 2, 3, 1};
  } else {
    transNode->name = "nhwc2nchw_" + tileName + std::to_string(id++);
    attr->perm = {0, 3, 1, 2};
  }
  transNode->primitive->value.value = attr;

  OpDefCopyer TransposeOpCopyer = [](CNodeT *inOpDef) -> std::unique_ptr<CNodeT> {
    auto newOpDef = std::make_unique<schema::CNodeT>();
    if (newOpDef == nullptr) {
      MS_LOG(ERROR) << "new CNodeT failed";
      return nullptr;
    }
    newOpDef->name = inOpDef->name;
    newOpDef->quantType = inOpDef->quantType;
    newOpDef->primitive = std::make_unique<schema::PrimitiveT>();
    if (newOpDef->primitive == nullptr) {
      MS_LOG(ERROR) << "new PrimitiveT failed";
      return nullptr;
    }
    newOpDef->primitive->value.type = schema::PrimitiveType_Transpose;
    auto transposeParam = new (std::nothrow) TransposeT;
    if (transposeParam == nullptr) {
      MS_LOG(ERROR) << "new transposeParam failed";
      return nullptr;
    }
    auto inParam = inOpDef->primitive->value.AsTranspose();
    MS_ASSERT(inParam != nullptr);
    transposeParam->perm.resize(inParam->perm.size());
    std::transform(inParam->perm.begin(), inParam->perm.end(), transposeParam->perm.begin(),
                   [](const int32_t ele) { return ele; });
    MS_ASSERT(newOpDef->primitive != nullptr);
    newOpDef->primitive->value.value = transposeParam;
    return newOpDef;
  };

  return InsertNode(graph, existNodeIter, place, inoutIdx, std::move(transNode), errorCode, TransposeOpCopyer);
}

void FormatTransPass::SetQuantType(QuantType quantType) { this->quantType = quantType; }

void FormatTransPass::SetFmk(converter::FmkType fmkType) { this->fmkType = fmkType; }

int FormatTransPass::GetFormat(const schema::CNodeT &node) {
  switch (node.primitive->value.type) {
    case schema::PrimitiveType_Conv2D:
      return node.primitive->value.AsConv2D()->format;
    case schema::PrimitiveType_DeConv2D:
      return node.primitive->value.AsDeConv2D()->format;
    case schema::PrimitiveType_DeDepthwiseConv2D:
      return node.primitive->value.AsDeDepthwiseConv2D()->format;
    case schema::PrimitiveType_DepthwiseConv2D:
      return node.primitive->value.AsDepthwiseConv2D()->format;
    case schema::PrimitiveType_Pooling:
      return node.primitive->value.AsPooling()->format;
    default:
      return schema::Format_NHWC;
  }
}

}  // namespace lite
}  // namespace mindspore
