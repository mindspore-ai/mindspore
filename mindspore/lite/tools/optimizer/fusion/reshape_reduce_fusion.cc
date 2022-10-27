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
#include <queue>
#include "ops/op_name.h"
#include "tools/lite_exporter/fetch_content.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace opt {
bool CheckIsClosedCycle(const FuncGraphPtr &func_graph, const CNodePtr &in_node, const CNodePtr &out_node) {
  std::set<CNodePtr> ops{in_node};
  std::queue<CNodePtr> link_ops;
  link_ops.push(out_node);
  int max_depth = 10;
  while (max_depth > 0 && !link_ops.empty()) {
    --max_depth;
    auto cur_node = link_ops.front();
    link_ops.pop();
    for (size_t i = 0; i < cur_node->size(); ++i) {
      if (utils::isa<CNode>(cur_node->input(i))) {
        auto in_cnode = cur_node->input(i)->cast<CNodePtr>();
        if (ops.find(in_cnode) != ops.end()) {
          continue;
        }
        link_ops.push(in_cnode);
        ops.insert(in_cnode);
      }
      if (utils::isa<Parameter>(cur_node->input(i))) {
        auto in_param = cur_node->input(i)->cast<ParameterPtr>();
        if (!in_param->has_default()) {
          return false;
        }
      }
    }
  }
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, false);
  return link_ops.empty() && std::all_of(ops.begin(), ops.end(), [&ops, &manager, &out_node](const CNodePtr &cnode) {
           auto node_users = manager->node_users()[cnode];
           for (auto &node_user : node_users) {
             auto post_node = node_user.first;
             if (!utils::isa<CNode>(post_node)) {
               return false;
             }
             auto post_cnode = post_node->cast<CNodePtr>();
             if (post_cnode != out_node && ops.find(post_cnode) == ops.end()) {
               return false;
             }
           }
           return true;
         });
}
bool ReshapeReduceFusion::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  (void)preprocessor_.Run(func_graph);
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
    if (!CheckReduce(cnode)) {
      continue;
    }
    if (CheckReshapeReduceFusion(func_graph, cnode)) {
      FuseReshapeWithReduce(func_graph, cnode);
    } else if (CheckReduceReshapeFusion(func_graph, cnode)) {
      FuseReduceWithReshape(func_graph, cnode);
    }
  }
  return true;
}

bool ReshapeReduceFusion::CheckReshapeReduceFusion(const FuncGraphPtr &func_graph, const CNodePtr &reduce) {
  MS_ASSERT(func_graph != nullptr && reduce != nullptr);
  auto reshape_node = reduce->input(1);
  if (!CheckPrimitiveType(reshape_node, prim::kPrimReshape)) {
    return false;
  }
  reshape_ = reshape_node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reshape_ != nullptr, false);
  if (IsMultiOutputTensors(func_graph, reshape_)) {
    return false;
  }
  if (!CheckReshape(reshape_)) {
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

bool ReshapeReduceFusion::CheckReduceReshapeFusion(const FuncGraphPtr &func_graph, const CNodePtr &reduce) {
  MS_ASSERT(func_graph != nullptr && reduce != nullptr);
  if (keep_dim_) {
    return false;
  }
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, false);
  std::vector<CNodePtr> reshape_ops;
  auto node_users = manager->node_users()[reduce];
  for (auto &node_user : node_users) {
    if (utils::isa<CNode>(node_user.first) && CheckPrimitiveType(node_user.first, prim::kPrimReshape)) {
      reshape_ops.push_back(node_user.first->cast<CNodePtr>());
    }
  }
  if (reshape_ops.size() != 1) {
    return false;
  }
  reshape_ = reshape_ops.front();
  if (!CheckIsClosedCycle(func_graph, reduce, reshape_)) {
    return false;
  }
  if (!CheckReshape(reshape_)) {
    return false;
  }
  const auto &shape_container = preprocessor_.GetShapeContainer();
  if (shape_container.find(reduce) == shape_container.end()) {
    return false;
  }
  if (shape_container.at(reduce).first.empty() || shape_container.at(reduce).second.empty()) {
    return false;
  }
  auto reduce_in = shape_container.at(reduce).first.front();
  axis_ = axis_ < 0 ? axis_ + static_cast<int>(reduce_in.size()) : axis_;
  auto reduce_out = shape_container.at(reduce).second.front();
  if (axis_ < 0 || axis_ > static_cast<int>(reduce_out.size())) {
    return false;
  }
  (void)reduce_out.insert(reduce_out.begin() + axis_, 1);
  if (std::count_if(reduce_in.begin(), reduce_in.end(), [](int64_t val) { return val < 0; }) > 1) {
    return false;
  }
  return reduce_out == ShapeVector(shape_.begin(), shape_.end());
}

bool ReshapeReduceFusion::CheckReduce(const CNodePtr &reduce) {
  MS_ASSERT(reduce != nullptr);
  if (IsMarkedTrainOp(reduce)) {
    return false;
  }
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
  if (IsQuantParameterNode(reduce_prim)) {
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

bool ReshapeReduceFusion::CheckReshape(const CNodePtr &reshape) {
  MS_ASSERT(reshape != nullptr);
  MS_CHECK_TRUE_RET(reshape != nullptr, false);
  if (IsMarkedTrainOp(reshape)) {
    return false;
  }
  if (reshape->size() != C3NUM) {
    return false;
  }
  auto prim = GetCNodePrimitive(reshape);
  MS_CHECK_TRUE_RET(prim != nullptr, false);
  if (IsQuantParameterNode(prim)) {
    return false;
  }
  const auto &shape_container = preprocessor_.GetShapeContainer();
  if (shape_container.find(reshape) != shape_container.end()) {
    const auto &reshape_infos = shape_container.at(reshape);
    if (reshape_infos.second.size() != 1) {
      return false;
    }
    const auto &out_shape = reshape_infos.second.front();
    shape_ = std::vector<int>(out_shape.begin(), out_shape.end());
    return true;
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

void ReshapeReduceFusion::FuseReduceWithReshape(const FuncGraphPtr &func_graph, const CNodePtr &reduce) {
  MS_ASSERT(reduce != nullptr);
  auto reduce_prim = GetCNodePrimitive(reduce);
  MS_ASSERT(reduce_prim != nullptr);
  (void)reduce_prim->AddAttr(ops::kKeepDims, MakeValue(true));
  if (reshape_->abstract() != nullptr) {
    reduce->set_abstract(reshape_->abstract()->Clone());
  }
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  (void)manager->Replace(reshape_, reduce);
}
}  // namespace opt
}  // namespace mindspore
