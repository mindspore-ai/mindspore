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
#include "plugin/device/ascend/optimizer/ir_fission/bce_with_logits_loss_fission.h"
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"
#include "backend/common/optimizer/helper.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "utils/trace_base.h"
#include "abstract/dshape.h"

namespace mindspore {
namespace opt {
void ReduceSuitForDynamicShape(const FuncGraphPtr &func_graph, const CNodePtr &reduce_node) {
  MS_EXCEPTION_IF_NULL(reduce_node);
  if (!common::AnfAlgo::IsDynamicShape(reduce_node)) {
    MS_LOG(DEBUG) << "Current node is not dynamic shape, skip!";
    return;
  }
  // Add the IOName:
  auto prim = common::AnfAlgo::GetCNodePrimitive(reduce_node);
  MS_EXCEPTION_IF_NULL(prim);
  std::vector<std::string> input_names = {"x", "axis"};
  (void)prim->AddAttr("input_names", MakeValue(input_names));
  (void)prim->AddAttr("output_names", MakeValue("y"));
  // When axis is empty, some additional processing needs to be done for the axis
  // under dynamic shape (Refer to the ReduceSumOptimizer pass):
  auto x_node = reduce_node->inputs()[1];
  auto x_base_shape = common::AnfAlgo::GetOutputDetailShape(x_node, 0);
  MS_EXCEPTION_IF_NULL(x_base_shape);
  if (x_base_shape->IsDimUnknown()) {
    MS_LOG(EXCEPTION) << "Can not support the case that input is dim unknown and axis is empty. node: "
                      << trace::DumpSourceLines(reduce_node);
  }
  auto x_shape = x_base_shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(x_shape);
  std::vector<int64_t> axis_vec;
  for (size_t i = 0; i < x_shape->shape().size(); ++i) {
    axis_vec.push_back(i);
  }
  // Convert axis from attr to input:
  auto axis_value = MakeValue<std::vector<int64_t>>(axis_vec);
  auto axis_tensor = CreateTupleTensor(axis_value->cast<ValueTuplePtr>());
  MS_EXCEPTION_IF_NULL(axis_tensor);
  auto axis_node = std::make_shared<ValueNode>(axis_tensor);
  MS_EXCEPTION_IF_NULL(axis_node);
  axis_node->set_abstract(axis_tensor->ToAbstract());
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  axis_node = kernel_graph->NewValueNode(axis_node);
  MS_EXCEPTION_IF_NULL(axis_node);
  kernel_graph->AddValueNodeToGraph(axis_node);
  axis_node->set_scope(reduce_node->scope());
  reduce_node->add_input(axis_node);
  common::AnfAlgo::EraseNodeAttr(kAttrAxis, reduce_node);
}

AnfNodePtr BCEWithLogitsLossFission::AddReduceNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // Copy a new sigmoid node, shape of output is the same as input
  std::vector<AnfNodePtr> new_simoid_inputs = {
    NewValueNode(std::make_shared<Primitive>(prim::kPrimBCEWithLogitsLoss->name()))};
  (void)new_simoid_inputs.insert(new_simoid_inputs.cend(), cnode->inputs().cbegin() + 1, cnode->inputs().cend());
  CNodePtr new_cnode = NewCNode(new_simoid_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(new_cnode);
  auto predict_input = cnode->inputs()[kIndex1];
  auto new_node_dtype = {common::AnfAlgo::GetOutputInferDataType(predict_input, 0)};
  auto new_node_shape = {common::AnfAlgo::GetOutputDetailShape(predict_input, 0)};
  // The kAttrReduction is necessary for InferShape of BCEWithLogitsLoss op
  common::AnfAlgo::SetNodeAttr(kAttrReduction, MakeValue("none"), new_cnode);
  common::AnfAlgo::SetOutputTypeAndDetailShape(new_node_dtype, new_node_shape, new_cnode.get());

  // Add reduce node
  string reduction = common::AnfAlgo::GetNodeAttr<std::string>(node, kAttrReduction);
  MS_LOG(INFO) << "Create reduce node, reduction attr is: " << reduction;
  std::vector<AnfNodePtr> reduce_inputs;
  if (reduction == "sum") {
    reduce_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimReduceSum->name())), new_cnode};
  } else if (reduction == "mean") {
    reduce_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimReduceMean->name())), new_cnode};
  } else {
    MS_LOG(INFO) << "Reduction attr is not mean or sum, can not do fission.";
    return nullptr;
  }
  auto reduce_node = NewCNode(reduce_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(reduce_node);
  auto shape = {common::AnfAlgo::GetOutputDetailShape(node, 0)};
  auto type = common::AnfAlgo::GetOutputInferDataType(node, 0);
  if (type == kNumberTypeFloat16) {
    common::AnfAlgo::SetOutputTypeAndDetailShape({kNumberTypeFloat32}, shape, reduce_node.get());
  } else {
    common::AnfAlgo::SetOutputTypeAndDetailShape({type}, shape, reduce_node.get());
  }

  common::AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(std::vector<int64_t>{}), reduce_node);
  common::AnfAlgo::SetNodeAttr("keep_dims", MakeValue(false), reduce_node);
  common::AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), reduce_node);
  reduce_node->set_scope(cnode->scope());

  ReduceSuitForDynamicShape(func_graph, reduce_node);

  if (type == kNumberTypeFloat16) {
    std::vector<AnfNodePtr> cast_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimCast->name())),
                                           reduce_node};
    auto cast_node = NewCNode(cast_inputs, func_graph);
    common::AnfAlgo::SetOutputTypeAndDetailShape({kNumberTypeFloat16}, shape, cast_node.get());
    common::AnfAlgo::SetNodeAttr(kAttrDstType, TypeIdToType(kNumberTypeFloat16), cast_node);
    cast_node->set_scope(reduce_node->scope());
    return cast_node;
  }
  return reduce_node;
}

const BaseRef BCEWithLogitsLossFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  MS_EXCEPTION_IF_NULL(Xs);
  return VectorRef({prim::kPrimBCEWithLogitsLoss, Xs});
}

// The corresponding op implementation of BCEWithLogitsLoss does not include the reduce implementation,
// so the reduce operator needs to be added when the reduction attr is sum or mean.
const AnfNodePtr BCEWithLogitsLossFission::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                   const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (GetBoolAttr(cnode, kAttrVisited)) {
    return nullptr;
  }
  common::AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);
  if (cnode->inputs().size() == 0) {
    return nullptr;
  }
  if (!common::AnfAlgo::HasNodeAttr("reduction", cnode)) {
    MS_LOG(INFO) << "Has no reduction attr.";
    return nullptr;
  }
  return AddReduceNode(func_graph, node);
}
}  // namespace opt
}  // namespace mindspore
