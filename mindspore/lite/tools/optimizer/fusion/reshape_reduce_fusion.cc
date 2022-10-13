/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#define USE_DEPRECATED_API
#include "tools/optimizer/fusion/reshape_reduce_fusion.h"
#include <set>
#include "ops/op_name.h"
#include "tools/lite_exporter/fetch_content.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace opt {
bool ReshapeReduceFusion::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    MS_LOG(ERROR) << "The manager of this graph is a nullptr.";
    return false;
  }
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    MS_CHECK_TRUE_RET(node != nullptr, false);
    if (!utils::isa<CNode>(node)) {
      continue;
    }
    if (!CheckPrimitiveType(node, prim::kPrimReduceFusion)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (IsMarkedTrainOp(cnode)) {
      continue;
    }
    if (!CheckCanFusion(func_graph, cnode)) {
      continue;
    }
    FuseReshapeWithReduce(func_graph, cnode);
  }
  return true;
}

bool ReshapeReduceFusion::CheckCanFusion(const FuncGraphPtr &func_graph, const CNodePtr &reduce) {
  MS_ASSERT(func_graph != nullptr && reduce != nullptr);
  if (!CheckReduce(reduce)) {
    return false;
  }
  auto reshape_node = reduce->input(1);
  if (!CheckPrimitiveType(reshape_node, prim::kPrimReshape)) {
    return false;
  }
  reshape_ = reshape_node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reshape_ != nullptr, false);
  if (IsMarkedTrainOp(reshape_) || IsMultiOutputTensors(func_graph, reshape_)) {
    return false;
  }
  if (reshape_->size() != C3NUM || utils::isa<CNode>(reshape_->input(C2NUM))) {
    return false;
  }
  lite::DataInfo data_info;
  if (lite::FetchConstData(reshape_, C2NUM, converter::kFmkTypeMs, &data_info, true) != lite::RET_OK) {
    return false;
  }
  if ((data_info.data_type_ != kNumberTypeInt && data_info.data_type_ != kNumberTypeInt32) ||
      data_info.data_.size() / C4NUM == 0) {
    return false;
  }
  shape_.resize(data_info.data_.size() / C4NUM);
  if (memcpy_s(shape_.data(), shape_.size() * sizeof(int), data_info.data_.data(), data_info.data_.size()) != EOK) {
    return false;
  }
  axis_ = axis_ < 0 ? axis_ + static_cast<int>(shape_.size()) : axis_;
  MS_CHECK_TRUE_RET(axis_ >= 0 && axis_ < static_cast<int>(shape_.size()), false);
  if (shape_[axis_] != 1) {
    return false;
  }
  if (!keep_dim_) {
    shape_.erase(shape_.begin() + axis_);
  }
  return true;
}

bool ReshapeReduceFusion::CheckReduce(const CNodePtr &reduce) {
  MS_ASSERT(reduce != nullptr);
  auto reduce_prim = GetCNodePrimitive(reduce);
  MS_CHECK_TRUE_RET(reduce_prim != nullptr, false);
  if (!IsReduceModeMeetOutEqualIn(reduce_prim)) {
    return false;
  }
  auto attr = reduce_prim->GetAttr(ops::kReduceToEnd);
  auto reduce_to_end = attr != nullptr && GetValue<bool>(attr);
  if (reduce_to_end) {
    return false;
  }
  attr = reduce_prim->GetAttr(ops::kCoeff);
  if (attr != nullptr && fabs(GetValue<float>(attr) - 1.f) > FLT_EPSILON) {
    return false;
  }
  attr = reduce_prim->GetAttr(ops::kKeepDims);
  keep_dim_ = attr != nullptr && GetValue<bool>(attr);
  if (reduce->size() != C3NUM || utils::isa<CNode>(reduce->input(C2NUM))) {
    return false;
  }
  lite::DataInfo data_info;
  if (lite::FetchConstData(reduce, C2NUM, converter::kFmkTypeMs, &data_info, true) != lite::RET_OK) {
    return false;
  }
  if ((data_info.data_type_ != kNumberTypeInt && data_info.data_type_ != kNumberTypeInt32) ||
      data_info.data_.size() != C4NUM) {
    return false;
  }
  axis_ = *reinterpret_cast<int *>(data_info.data_.data());
  return true;
}

void ReshapeReduceFusion::FuseReshapeWithReduce(const FuncGraphPtr &func_graph, const CNodePtr &reduce) {
  MS_ASSERT(reduce != nullptr);
  auto param_node = BuildIntVecParameterNode(func_graph, shape_, reshape_->input(C2NUM)->fullname_with_scope());
  if (param_node == nullptr) {
    return;
  }
  if (reduce->abstract() != nullptr) {
    reshape_->set_abstract(reduce->abstract()->Clone());
  }
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  manager->SetEdge(reshape_, C2NUM, param_node);
  (void)manager->Replace(reduce, reshape_);
}
}  // namespace opt
}  // namespace mindspore
