/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "backend/optimizer/ascend/ir_fission/cdist_fission.h"
#include <algorithm>
#include <memory>
#include <vector>
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kCdistInputNum = 2;
constexpr size_t kCdistGradInputNum = 4;
constexpr int64_t kInputXDimP = -1;
constexpr int64_t kInputYDimR = -2;

std::vector<size_t> CalCdistBroadCastShape(std::vector<size_t> x_shape, std::vector<size_t> y_shape) {
  (void)x_shape.insert(x_shape.end() + kInputXDimP, 1);
  (void)y_shape.insert(y_shape.end() + kInputYDimR, 1);
  if (x_shape.size() != y_shape.size()) {
    MS_EXCEPTION(ValueError) << "For Cdist, input_x and input_y should have the same rank.";
  }
  if (x_shape == y_shape) {
    return x_shape;
  }
  auto length = x_shape.size();
  std::vector<size_t> broadcast_shape;
  std::copy(x_shape.begin(), x_shape.end() - SizeToLong(length), std::back_inserter(broadcast_shape));
  for (size_t i = length; i > 0; --i) {
    if (x_shape[length - i] == 1) {
      broadcast_shape.push_back(y_shape[length - i]);
    } else if (y_shape[length - i] == 1) {
      broadcast_shape.push_back(x_shape[length - i]);
    } else if (x_shape[length - i] == y_shape[length - i]) {
      broadcast_shape.push_back(x_shape[length - i]);
    } else {
      MS_EXCEPTION(ValueError) << "The two input shape can not broadcast, x_shape: " << x_shape << ", y_shape"
                               << y_shape;
    }
  }
  return broadcast_shape;
}

AnfNodePtr AddBroadCastToNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node, int64_t dim,
                              const std::vector<size_t> &need_shape) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(input_node);
  // Add ExpandDims Node
  std::vector<AnfNodePtr> expand_dims_inputs = {
    NewValueNode(std::make_shared<Primitive>(prim::kPrimExpandDims->name())), input_node};
  auto expand_dims = func_graph->NewCNode(expand_dims_inputs);
  auto dtype = AnfAlgo::GetOutputInferDataType(input_node, 0);
  auto expand_shape = AnfAlgo::GetOutputInferShape(input_node, 0);
  (void)expand_shape.insert(expand_shape.end() + dim, 1);
  AnfAlgo::SetOutputInferTypeAndShape({dtype}, {expand_shape}, expand_dims.get());
  AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(dim), expand_dims);
  AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), expand_dims);
  // Add BroadCastTo Node
  std::vector<AnfNodePtr> broadcast_to_inputs = {
    NewValueNode(std::make_shared<Primitive>(prim::kPrimBroadcastTo->name())), expand_dims};
  auto broadcast_to = func_graph->NewCNode(broadcast_to_inputs);
  AnfAlgo::SetOutputInferTypeAndShape({dtype}, {need_shape}, broadcast_to.get());
  std::vector<int64_t> shape;
  (void)std::transform(need_shape.begin(), need_shape.end(), std::back_inserter(shape), LongToSize);
  AnfAlgo::SetNodeAttr(kAttrShape, MakeValue(shape), broadcast_to);
  AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), broadcast_to);
  return broadcast_to;
}
}  // namespace

const BaseRef CdistFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto cdist_prim = std::make_shared<Primitive>(prim::kPrimCdist->name());
  return VectorRef({cdist_prim, Xs});
}

const BaseRef CdistGradFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto cdist_grad_prim = std::make_shared<Primitive>(prim::kPrimCdistGrad->name());
  return VectorRef({cdist_grad_prim, Xs});
}

const AnfNodePtr CdistFission::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cdist_cnode = node->cast<CNodePtr>();
  if (GetBoolAttr(cdist_cnode, kAttrVisited)) {
    return nullptr;
  }
  AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);
  MS_EXCEPTION_IF_NULL(cdist_cnode);
  if (cdist_cnode->size() != kCdistInputNum + 1) {
    MS_LOG(INFO) << "The node " << cdist_cnode->DebugString() << " is not equal to " << cdist_cnode << " inputs";
    return nullptr;
  }
  const auto &cdist_inputs = cdist_cnode->inputs();
  auto x_shape = AnfAlgo::GetOutputInferShape(cdist_inputs[kDim1], 0);
  auto y_shape = AnfAlgo::GetOutputInferShape(cdist_inputs[kDim2], 0);
  auto broadcast_to_shape = CalCdistBroadCastShape(x_shape, y_shape);
  auto broadcast_input_x = AddBroadCastToNode(graph, cdist_inputs[kDim1], kInputXDimP, broadcast_to_shape);
  auto broadcast_input_y = AddBroadCastToNode(graph, cdist_inputs[kDim2], kInputYDimR, broadcast_to_shape);
  std::vector<AnfNodePtr> new_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimCdist->name())),
                                     broadcast_input_x, broadcast_input_y};
  CNodePtr new_cnode = graph->NewCNode(new_inputs);
  MS_EXCEPTION_IF_NULL(new_cnode);
  new_cnode->set_abstract(cdist_cnode->abstract());
  new_cnode->set_scope(cdist_cnode->scope());
  AnfAlgo::CopyNodeAttrs(cdist_cnode, new_cnode);
  return new_cnode;
}

const AnfNodePtr CdistGradFission::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cdist_grad_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cdist_grad_cnode);
  if (GetBoolAttr(cdist_grad_cnode, kAttrVisited)) {
    return nullptr;
  }
  AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);
  if (cdist_grad_cnode->size() != kCdistGradInputNum + 1) {
    MS_LOG(INFO) << "The node " << cdist_grad_cnode->DebugString() << " is not equal to " << cdist_grad_cnode
                 << " inputs";
    return nullptr;
  }
  const auto &cdist_grad_inputs = cdist_grad_cnode->inputs();
  auto x_shape = AnfAlgo::GetOutputInferShape(cdist_grad_inputs[kDim2], 0);
  auto y_shape = AnfAlgo::GetOutputInferShape(cdist_grad_inputs[kDim3], 0);
  auto broadcast_to_shape = CalCdistBroadCastShape(x_shape, y_shape);
  auto broadcast_grad = AddBroadCastToNode(graph, cdist_grad_inputs[kDim1], 0, broadcast_to_shape);
  auto broadcast_input_x = AddBroadCastToNode(graph, cdist_grad_inputs[kDim2], kInputXDimP, broadcast_to_shape);
  auto broadcast_input_y = AddBroadCastToNode(graph, cdist_grad_inputs[kDim3], kInputYDimR, broadcast_to_shape);
  auto broadcast_out = AddBroadCastToNode(graph, cdist_grad_inputs[kDim4], 0, broadcast_to_shape);
  std::vector<AnfNodePtr> new_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimCdistGrad->name())),
                                     broadcast_grad, broadcast_input_x, broadcast_input_y, broadcast_out};
  CNodePtr new_cnode = graph->NewCNode(new_inputs);
  MS_EXCEPTION_IF_NULL(new_cnode);
  new_cnode->set_abstract(cdist_grad_cnode->abstract());
  new_cnode->set_scope(cdist_grad_cnode->scope());
  AnfAlgo::CopyNodeAttrs(cdist_grad_cnode, new_cnode);
  return new_cnode;
}
}  // namespace opt
}  // namespace mindspore
