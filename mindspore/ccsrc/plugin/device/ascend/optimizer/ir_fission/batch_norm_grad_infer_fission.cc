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
#include "plugin/device/ascend/optimizer/ir_fission/batch_norm_grad_infer_fission.h"
#include <vector>
#include "include/backend/optimizer/helper.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kBatchNormGradInferOutputNum = 3;
bool CheckOutputsIndex(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (manager->node_users().find(node) == manager->node_users().end()) {
    MS_LOG(DEBUG) << "The node " << node->DebugString() << " should have some outputs";
    return false;
  }
  for (const auto &node_index : manager->node_users()[node]) {
    const AnfNodePtr &output = node_index.first;
    MS_EXCEPTION_IF_NULL(output);
    if (!IsPrimitiveCNode(output, prim::kPrimTupleGetItem)) {
      continue;
    }
    auto tuple_getiterm_cnode = output->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(tuple_getiterm_cnode);
    auto index_node = tuple_getiterm_cnode->input(kInputNodeOutputIndexInTupleGetItem);
    MS_EXCEPTION_IF_NULL(index_node);
    auto value_node = index_node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    auto index = GetValue<int64_t>(value_node->value());
    auto output_num = SizeToLong(kBatchNormGradInferOutputNum);
    if (index == output_num || index == output_num + 1) {
      MS_LOG(DEBUG) << "The output " << index << " of node " << node->DebugString() << " is not null, no need change";
      return false;
    }
  }
  return true;
}
}  // namespace

AnfNodePtr BatchNormGradInferFission::CreateBNInferGrad(const FuncGraphPtr &func_graph, const AnfNodePtr &bn_grad,
                                                        const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(bn_grad);
  MS_EXCEPTION_IF_NULL(equiv);
  // Set inputs
  Equiv::const_iterator iter_input0 = (*equiv).find(input0_var_);
  if (iter_input0 == (*equiv).cend()) {
    MS_LOG(EXCEPTION) << "The equiv map is expected to contains the input0 var after matched."
                      << trace::DumpSourceLines(bn_grad);
  }
  Equiv::const_iterator iter_input2 = (*equiv).find(input2_var_);
  if (iter_input2 == (*equiv).cend()) {
    MS_LOG(EXCEPTION) << "The equiv map is expected to contains the input2 var after matched."
                      << trace::DumpSourceLines(bn_grad);
  }
  Equiv::const_iterator iter_input4 = (*equiv).find(input4_var_);
  if (iter_input4 == (*equiv).cend()) {
    MS_LOG(EXCEPTION) << "The equiv map is expected to contains the input4 var after matched."
                      << trace::DumpSourceLines(bn_grad);
  }
  std::vector<AnfNodePtr> bn_infer_grad_inputs = {
    NewValueNode(std::make_shared<Primitive>(kBNInferGradOpName)), utils::cast<AnfNodePtr>(iter_input0->second),
    utils::cast<AnfNodePtr>(iter_input2->second), utils::cast<AnfNodePtr>(iter_input4->second)};
  auto bn_infer_grad = NewCNode(bn_infer_grad_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(bn_infer_grad);
  // Set abstract, the output of new node is taking the place of the 0th output of bn_grad.
  auto bn_grad_abstract_tuple = dyn_cast<abstract::AbstractTuple>(bn_grad->abstract());
  MS_EXCEPTION_IF_NULL(bn_grad_abstract_tuple);
  if (bn_grad_abstract_tuple->elements().empty()) {
    MS_LOG(EXCEPTION) << "The abstract tuple of node " << bn_grad->DebugString() << "should not be empty"
                      << trace::DumpSourceLines(bn_grad);
  }
  bn_infer_grad->set_abstract(bn_grad_abstract_tuple->elements()[0]);
  common::AnfAlgo::CopyNodeAttr(kAttrEpsilon, bn_grad, bn_infer_grad);
  bn_infer_grad->set_scope(bn_grad->scope());
  return bn_infer_grad;
}

AnfNodePtr BatchNormGradInferFission::CreateBNTrainingUpdateGrad(const FuncGraphPtr &func_graph,
                                                                 const AnfNodePtr &bn_grad,
                                                                 const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(bn_grad);
  MS_EXCEPTION_IF_NULL(equiv);
  // Set inputs
  Equiv::const_iterator iter_input0 = (*equiv).find(input0_var_);
  if (iter_input0 == (*equiv).cend()) {
    MS_LOG(EXCEPTION) << "The equiv map is expected to contains the input0 var after matched."
                      << trace::DumpSourceLines(bn_grad);
  }
  Equiv::const_iterator iter_input1 = (*equiv).find(input1_var_);
  if (iter_input1 == (*equiv).cend()) {
    MS_LOG(EXCEPTION) << "The equiv map is expected to contains the input1 var after matched."
                      << trace::DumpSourceLines(bn_grad);
  }
  Equiv::const_iterator iter_input3 = (*equiv).find(input3_var_);
  if (iter_input3 == (*equiv).cend()) {
    MS_LOG(EXCEPTION) << "The equiv map is expected to contains the input3 var after matched."
                      << trace::DumpSourceLines(bn_grad);
  }
  Equiv::const_iterator iter_input4 = (*equiv).find(input4_var_);
  if (iter_input4 == (*equiv).cend()) {
    MS_LOG(EXCEPTION) << "The equiv map is expected to contains the input4 var after matched."
                      << trace::DumpSourceLines(bn_grad);
  }
  std::vector<AnfNodePtr> bn_training_update_grad_inputs = {
    NewValueNode(std::make_shared<Primitive>(kBNTrainingUpdateGradOpName)),
    utils::cast<AnfNodePtr>(iter_input0->second), utils::cast<AnfNodePtr>(iter_input1->second),
    utils::cast<AnfNodePtr>(iter_input3->second), utils::cast<AnfNodePtr>(iter_input4->second)};
  auto bn_training_update_grad = NewCNode(bn_training_update_grad_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(bn_training_update_grad);
  // Set abstract, the outputs of new node are taking the place of the 1st and 2nd outputs of bn_grad.
  auto bn_grad_abstract_tuple = dyn_cast<abstract::AbstractTuple>(bn_grad->abstract());
  MS_EXCEPTION_IF_NULL(bn_grad_abstract_tuple);
  if (bn_grad_abstract_tuple->elements().size() < kBatchNormGradInferOutputNum) {
    MS_LOG(EXCEPTION) << "The abstract tuple of node " << bn_grad->DebugString() << "should not be less than 3"
                      << trace::DumpSourceLines(bn_grad);
  }
  std::vector<AbstractBasePtr> abstract_list{bn_grad_abstract_tuple->elements()[kIndex1],
                                             bn_grad_abstract_tuple->elements()[kIndex2]};
  auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(abstract_list);
  bn_training_update_grad->set_abstract(abstract_tuple);
  common::AnfAlgo::CopyNodeAttr(kAttrEpsilon, bn_grad, bn_training_update_grad);
  bn_training_update_grad->set_scope(bn_grad->scope());
  return bn_training_update_grad;
}

const BaseRef BatchNormGradInferFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimBatchNormGrad, input0_var_, input1_var_, input2_var_, input3_var_, input4_var_, Xs});
}

const AnfNodePtr BatchNormGradInferFission::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                    const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  if (!common::AnfAlgo::HasNodeAttr(kAttrIsTraining, node->cast<CNodePtr>())) {
    MS_LOG(DEBUG) << "The BatchNormGrad " << node->DebugString() << " has no is_training attr, should not be changed";
    return nullptr;
  }
  if (common::AnfAlgo::GetNodeAttr<bool>(node, kAttrIsTraining)) {
    MS_LOG(DEBUG) << "The is_training attr value of " << node->DebugString() << " is true, no need change";
    return nullptr;
  }
  if (!CheckOutputsIndex(func_graph, node)) {
    MS_LOG(DEBUG) << "The output 3 or 4 of BatchNormGrad is not null, no need change";
    return nullptr;
  }
  AnfNodePtr bn_infer_grad = CreateBNInferGrad(func_graph, node, equiv);
  AnfNodePtr bn_training_update_grad = CreateBNTrainingUpdateGrad(func_graph, node, equiv);
  std::vector<AnfNodePtr> bn_training_update_grad_outputs;
  CreateMultipleOutputsOfAnfNode(func_graph, bn_training_update_grad, kBNTrainingUpdateGradOutputNum,
                                 &bn_training_update_grad_outputs);
  if (bn_training_update_grad_outputs.size() != kBNTrainingUpdateGradOutputNum) {
    MS_LOG(EXCEPTION) << "The output size of " << bn_training_update_grad << " should be "
                      << kBNTrainingUpdateGradOutputNum << ", but it is " << bn_training_update_grad_outputs.size()
                      << trace::DumpSourceLines(node);
  }
  std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple), bn_infer_grad,
                                               bn_training_update_grad_outputs[0], bn_training_update_grad_outputs[1]};
  auto make_tuple = func_graph->NewCNode(make_tuple_inputs);
  MS_EXCEPTION_IF_NULL(make_tuple);
  return make_tuple;
}
}  // namespace opt
}  // namespace mindspore
