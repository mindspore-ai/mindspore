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

#include <algorithm>
#include <string>
#include <memory>
#include <utility>
#include <vector>
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
  if (fmk_type_ == converter::FmkType_TFLITE) {  // inference by nhwc
    if (!IsContain(GetNchwOpList(), GetCNodeTType(node))) {
      return RET_NO_CHANGE;
    }
    *beforeNodeType = kNHWC2NCHW;
    *afterNodeType = kNCHW2NHWC;
    return RET_OK;
  } else if (fmk_type_ == converter::FmkType_CAFFE || fmk_type_ == converter::FmkType_MS ||
             fmk_type_ == converter::FmkType_ONNX) {
    if (!IsContain(GetNhwcOpList(), GetCNodeTType(node))) {
      return RET_NO_CHANGE;
    }
    *beforeNodeType = kNCHW2NHWC;
    *afterNodeType = kNHWC2NCHW;
    return RET_OK;
  } else if (fmk_type_ == converter::FmkType_TF) {
    if (IsContain(GetNhwcOpList(), GetCNodeTType(node)) && GetFormat(node) == schema::Format_NCHW) {
      *beforeNodeType = kNCHW2NHWC;
      *afterNodeType = kNHWC2NCHW;
      return RET_OK;
    }
    if (IsContain(GetNchwOpList(), GetCNodeTType(node))) {
      *beforeNodeType = kNHWC2NCHW;
      *afterNodeType = kNCHW2NHWC;
      return RET_OK;
    }
    return RET_NO_CHANGE;
  }
  MS_LOG(ERROR) << "Unsupported fmk: " << fmk_type_;
  return RET_ERROR;
}

STATUS FormatTransPass::DoModelInputFormatTrans(schema::MetaGraphT *graph) {
  if (fmk_type_ == converter::FmkType_TF || fmk_type_ == converter::FmkType_TFLITE) {
    return RET_OK;
  }
  MS_ASSERT(graph != nullptr);
  // insert trans node in model input tensor
  if (graph->nodes.empty()) {
    return RET_OK;
  }
  // onnx input format may be nhwc
  if (fmk_type_ == converter::FmkType_ONNX && graph->inputIndex.size() == 1) {
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
    auto specInsertIndexes = GetExtNhwcIndexes();
    auto opType = GetCNodeTType(**iter);
    if (specInsertIndexes.find(opType) != specInsertIndexes.end()) {
      for (auto insert_index : specInsertIndexes[opType]) {
        iter = InsertFormatTransNode(graph, iter, kBefore, insert_index, beforeNodeType, &status);
        if (status != RET_OK) {
          MS_LOG(ERROR) << "InsertNchw2NhwcNode before " << nodeName << "failed";
          return RET_ERROR;
        }
      }
    } else if (IsContain(GetNhwcAllInputOpList(), opType)) {
      auto input_size = node->inputIndex.size();
      if (GetCNodeTType(**iter) == schema::PrimitiveType_ResizeGrad) {
        if ((**iter).primitive->value.AsResizeGrad()->method == schema::ResizeMethod_NEAREST) {
          input_size = 1;
        }
      }
      for (size_t i = 0; i < input_size; i++) {
        iter = InsertFormatTransNode(graph, iter, kBefore, i, beforeNodeType, &status);
        if (status != RET_OK) {
          MS_LOG(ERROR) << "InsertNchw2NhwcNode before " << nodeName << "failed";
          return RET_ERROR;
        }
      }
    } else {
      iter = InsertFormatTransNode(graph, iter, kBefore, 0, beforeNodeType, &status);
    }
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
  auto perm_tensor = std::make_unique<schema::TensorT>();
  perm_tensor->dataType = kNumberTypeInt32;
  perm_tensor->dims = {4};
  std::vector<int> perm;
  if (nodeType == kNCHW2NHWC) {
    transNode->name = "nchw2nhwc_" + tileName + std::to_string(id_++);
    perm = {0, 2, 3, 1};
  } else {
    transNode->name = "nhwc2nchw_" + tileName + std::to_string(id_++);
    perm = {0, 3, 1, 2};
  }
  size_t bytes = perm.size() * sizeof(int);
  perm_tensor->data.resize(bytes);
  if (memcpy_s(perm_tensor->data.data(), bytes, perm.data(), bytes) != EOK) {
    MS_LOG(ERROR) << "memcpy data failed.";
  }
  perm_tensor->name = transNode->name + "_perm";

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
    return newOpDef;
  };
  int insert_num = 0;
  auto iter =
    InsertNode(graph, existNodeIter, place, inoutIdx, std::move(transNode), errorCode, &insert_num, TransposeOpCopyer);
  size_t index = graph->allTensors.size();
  graph->allTensors.push_back(std::move(perm_tensor));
  for (int i = insert_num; i > 0; --i) {
    (*(iter - i))->inputIndex.push_back(index);
  }
  return iter;
}

int FormatTransPass::GetFormat(const schema::CNodeT &node) {
  switch (node.primitive->value.type) {
    case schema::PrimitiveType_Conv2DFusion:
      return node.primitive->value.AsConv2DFusion()->format;
    case schema::PrimitiveType_Conv2dTransposeFusion:
      return node.primitive->value.AsConv2dTransposeFusion()->format;
    case schema::PrimitiveType_AvgPoolFusion:
      return node.primitive->value.AsAvgPoolFusion()->format;
    case schema::PrimitiveType_MaxPoolFusion:
      return node.primitive->value.AsMaxPoolFusion()->format;
    default:
      return schema::Format_NHWC;
  }
}

STATUS FormatTransPass::ChangeOpAxis(schema::MetaGraphT *graph, const std::unique_ptr<schema::CNodeT> &node) {
  MS_ASSERT(node->primitive != nullptr);
  auto type = node->primitive->value.type;
  auto input1_ndim = graph->allTensors.at(node->inputIndex[0])->dims.size();
  if (input1_ndim != 4) {
    if (node->inputIndex.size() > 1) {
      auto input2_ndim = graph->allTensors.at(node->inputIndex[1])->dims.size();
      if (input2_ndim != 4 && input2_ndim != 0) {
        MS_LOG(ERROR) << "change op axis only support 4 dims";
        return RET_NOT_SUPPORT;
      }
    } else {
      MS_LOG(DEBUG) << "change op axis only support 4 dims";
      return RET_NOT_SUPPORT;
    }
  }
  if (type == schema::PrimitiveType_Concat) {
    MS_ASSERT(node->primitive->value.AsConcat() != nullptr);
    auto origin_axis = node->primitive->value.AsConcat()->axis;
    auto axis_map = GetNc2NhAxisMap();
    if (node->primitive->value.AsConcat() == nullptr) {
      MS_LOG(ERROR) << "node->primitive->value.AsConcat() is nullptr";
      return RET_NULL_PTR;
    }
    node->primitive->value.AsConcat()->axis = axis_map[origin_axis < 0 ? origin_axis + 4 : origin_axis];
  }
  if (type == schema::PrimitiveType_Split) {
    MS_ASSERT(node->primitive->value.AsSplit() != nullptr);
    auto origin_axis = node->primitive->value.AsSplit()->axis;
    auto axis_map = GetNc2NhAxisMap();
    if (node->primitive->value.AsSplit() == nullptr) {
      MS_LOG(ERROR) << "node->primitive->value.AsSplit() is nullptr";
      return RET_NULL_PTR;
    }
    node->primitive->value.AsSplit()->axis = axis_map[origin_axis];
  }
  if (type == schema::PrimitiveType_Crop) {
    MS_ASSERT(node->primitive->value.AsCrop() != nullptr);
    auto origin_axis = node->primitive->value.AsCrop()->axis;
    auto offsets = node->primitive->value.AsCrop()->offsets;
    auto axis_map = GetNc2NhAxisMap();
    if (node->primitive->value.AsCrop() == nullptr) {
      MS_LOG(ERROR) << "node->primitive->value.AsCrop() is nullptr";
      return RET_NULL_PTR;
    }
    // nchw->nhwc,offsets need pad 0;
    if (axis_map[origin_axis] == 0) {
      offsets = {offsets[0], offsets[2], offsets[3], offsets[1]};
    } else if (axis_map[origin_axis] == 1 || axis_map[origin_axis] == 2) {
      // orgin_axis = 2 or orgin_axis = 3
      offsets.push_back(0);
    } else if (axis_map[origin_axis] == -1) {
      // origin_axis = 1
      offsets = {offsets[1], offsets[2], offsets[0]};
    } else {
      // axis error
      MS_LOG(ERROR) << "Crop error";
      return RET_ERROR;
    }
    node->primitive->value.AsCrop()->offsets = offsets;
  }
  if (type == schema::PrimitiveType_SliceFusion || type == schema::PrimitiveType_StridedSlice) {
    return ChangeOpSliceAndStridedSlice(graph, node);
  }
  return RET_OK;
}

void FormatTransPass::TransformAttrByAxes(int *origin_attr, int *axes, int element_size) {
  if (origin_attr == nullptr || axes == nullptr || element_size == 0) {
    return;
  }
  auto axis_map = GetNc2NhAxisMap();
  std::vector<int> cur_attr;
  for (int dim = 0; dim < 4; ++dim) {
    for (int index = 0; index < element_size; ++index) {
      int nhwc_dim = axis_map[axes[index] < 0 ? axes[index] + 4 : axes[index]];
      if (nhwc_dim == dim || (nhwc_dim + 4) == dim) {
        cur_attr.push_back(origin_attr[index]);
      }
    }
  }
  for (int index = 0; index < element_size; ++index) {
    origin_attr[index] = cur_attr[index];
  }
}

void FormatTransPass::TransformOpAxisAttr(int *origin_axis, int element_size) {
  if (origin_axis == nullptr || element_size == 0) {
    return;
  }
  auto axis_map = GetNc2NhAxisMap();
  std::vector<int> new_axis;
  for (int i = 0; i < element_size; ++i) {
    int axis = axis_map[origin_axis[i]];
    axis = axis < 0 ? axis + 4 : axis;
    new_axis.push_back(axis);
  }
  std::sort(new_axis.begin(), new_axis.end());
  for (int i = 0; i < element_size; ++i) {
    origin_axis[i] = new_axis[i];
  }
}

STATUS FormatTransPass::ChangeOpSlice(schema::MetaGraphT *graph, const std::unique_ptr<schema::CNodeT> &node) {
  auto attr = node->primitive->value.AsSliceFusion();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "node->primitive->value.AsSliceFusion() is nullptr.";
    return RET_NULL_PTR;
  }
  // transform attr
  if (node->inputIndex.size() < 2) {
    MS_LOG(ERROR) << "slice input is error";
    return RET_ERROR;
  }
  for (size_t index = 1; index < node->inputIndex.size(); ++index) {
    if (graph->allTensors[node->inputIndex[index]]->data.data() == nullptr) {
      return RET_NOT_SUPPORT;
    }
  }
  int element_num = graph->allTensors[node->inputIndex[1]]->dims[0];
  std::vector<int> axes;
  auto axes_attr = attr->axes;
  if (axes_attr.empty()) {
    for (int index = 0; index < element_num; ++index) {
      axes.push_back(index);
    }
  } else {
    std::transform(axes_attr.begin(), axes_attr.end(), std::back_inserter(axes),
                   [](int64_t val) { return static_cast<int>(val); });
  }
  for (size_t index = 1; index < node->inputIndex.size(); ++index) {
    TransformAttrByAxes(reinterpret_cast<int *>(graph->allTensors[node->inputIndex[index]]->data.data()),
                        reinterpret_cast<int *>(axes.data()), element_num);
  }
  TransformOpAxisAttr(axes.data(), element_num);
  attr->axes.clear();
  for (int i = 0; i < element_num; ++i) {
    attr->axes.push_back(static_cast<int64_t>(axes[i]));
  }
  return RET_OK;
}

STATUS FormatTransPass::ChangeOpStridedSlice(schema::MetaGraphT *graph, const std::unique_ptr<schema::CNodeT> &node) {
  // onnx input size is equal to 5 always.
  if (node->inputIndex.size() != 5) {
    return RET_NOT_SUPPORT;
  }
  if (node->inputIndex.size() == 5) {
    for (int index = 1; index < 5; ++index) {
      if (graph->allTensors[node->inputIndex[index]]->data.data() == nullptr) {
        return RET_NOT_SUPPORT;
      }
    }
    int element_num = graph->allTensors[node->inputIndex[1]]->dims[0];
    auto axes = graph->allTensors[node->inputIndex[3]]->data;
    for (int index = 1; index < 5; ++index) {
      if (index == 3) {
        continue;
      }
      TransformAttrByAxes(reinterpret_cast<int *>(graph->allTensors[node->inputIndex[index]]->data.data()),
                          reinterpret_cast<int *>(axes.data()), element_num);
    }
    TransformOpAxisAttr(reinterpret_cast<int *>(graph->allTensors[node->inputIndex[3]]->data.data()), element_num);
  }
  return RET_OK;
}

STATUS FormatTransPass::ChangeOpSliceAndStridedSlice(schema::MetaGraphT *graph,
                                                     const std::unique_ptr<schema::CNodeT> &node) {
  auto type = node->primitive->value.type;
  if (type == schema::PrimitiveType_StridedSlice) {
    return ChangeOpStridedSlice(graph, node);
  }
  if (type == schema::PrimitiveType_SliceFusion) {
    return ChangeOpSlice(graph, node);
  }
  return RET_ERROR;
}
}  // namespace lite
}  // namespace mindspore
