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
#include "pre_activate/ascend/ir_fusion/fused_batch_norm_fusion.h"
#include <memory>
#include <algorithm>
#include "pre_activate/common/helper.h"
#include "session/anf_runtime_algorithm.h"
#include "utils/utils.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kReplaceOutputIndex0 = 3;
constexpr size_t kReplaceOutputIndex1 = 4;
bool IsC(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    AnfNodePtr in = utils::cast<AnfNodePtr>(n);
    MS_EXCEPTION_IF_NULL(in);
    return in->isa<ValueNode>();
  }
  return false;
}

void GetBNOutput(const FuncGraphPtr &func_graph, const AnfNodePtr &bn, std::vector<AnfNodePtr> *bn_outputs) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(bn);
  MS_EXCEPTION_IF_NULL(bn_outputs);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (manager->node_users().find(bn) == manager->node_users().end()) {
    MS_LOG(EXCEPTION) << "The bn node " << bn->DebugString() << " should has some outputs";
  }
  for (const auto &node_index : manager->node_users()[bn]) {
    AnfNodePtr output = node_index.first;
    MS_EXCEPTION_IF_NULL(output);
    bn_outputs->push_back(output);
  }
}
}  // namespace

const BaseRef FusedBatchNormFusion::DefinePattern() const {
  std::shared_ptr<Var> Xs = std::make_shared<SeqVar>();
  VarPtr index0 = std::make_shared<CondVar>(IsC);
  VarPtr index1 = std::make_shared<CondVar>(IsC);
  VarPtr index2 = std::make_shared<CondVar>(IsC);
  VectorRef batch_norm = VectorRef({batch_norm_var_, data_input0_var_, data_input1_var_, data_input2_var_, Xs});
  VectorRef tuple_getitem0 = VectorRef({prim::kPrimTupleGetItem, batch_norm, index0});
  VectorRef tuple_getitem1 = VectorRef({prim::kPrimTupleGetItem, batch_norm, index1});
  VectorRef tuple_getitem2 = VectorRef({prim::kPrimTupleGetItem, batch_norm, index2});
  VectorRef sub0 = VectorRef({prim::kPrimSub, variable_input0_var_, tuple_getitem1});
  VectorRef sub1 = VectorRef({prim::kPrimSub, variable_input1_var_, tuple_getitem2});
  VectorRef mul0 = VectorRef({prim::kPrimMul, sub0, constant_input0_var_});
  VectorRef mul1 = VectorRef({prim::kPrimMul, sub1, constant_input1_var_});
  VectorRef assign_sub0 = VectorRef({prim::kPrimAssignSub, variable_input0_var_, mul0});
  VectorRef assign_sub1 = VectorRef({prim::kPrimAssignSub, variable_input1_var_, mul1});
  VectorRef depend0 = VectorRef({prim::kPrimDepend, tuple_getitem0, assign_sub0});
  return VectorRef({prim::kPrimDepend, depend0, assign_sub1});
}

ValuePtr FusedBatchNormFusion::GetFactor(const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(equiv);
  auto iter_constant_input0 = (*equiv).find(constant_input0_var_);
  if (iter_constant_input0 == (*equiv).end()) {
    MS_LOG(EXCEPTION) << "The equiv map is expected to contains the constant_input0 var after matched.";
  }
  auto constant_input = utils::cast<AnfNodePtr>(iter_constant_input0->second);
  MS_EXCEPTION_IF_NULL(constant_input);
  if (!constant_input->isa<ValueNode>()) {
    return nullptr;
  }
  auto value_node = constant_input->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  if (!value->isa<tensor::Tensor>()) {
    return nullptr;
  }
  auto tensor_ptr = value->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  if (tensor_ptr->data_type() == kNumberTypeFloat16) {
    auto *half_data = static_cast<const Eigen::half *>(tensor_ptr->data_c());
    MS_EXCEPTION_IF_NULL(half_data);
    float float_data = Eigen::half_impl::half_to_float(half_data[0]);
    return MakeValue(float_data);
  } else if (tensor_ptr->data_type() == kNumberTypeFloat32) {
    auto *tensor_data = static_cast<const float *>(tensor_ptr->data_c());
    MS_EXCEPTION_IF_NULL(tensor_data);
    return MakeValue(tensor_data[0]);
  } else {
    MS_LOG(WARNING) << "The factor data type of value node " << value_node->DebugString() << " is not fp16 or fp32";
    return nullptr;
  }
}

AnfNodePtr FusedBatchNormFusion::CreateBNTrainingReduce(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                        const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);
  // Set input to create node
  auto iter_data_input0 = (*equiv).find(data_input0_var_);
  if (iter_data_input0 == (*equiv).end()) {
    MS_LOG(EXCEPTION) << "The equiv map is expected to contains the data_input0 var after matched.";
  }
  std::vector<AnfNodePtr> bn_training_reduce_inputs = {
    NewValueNode(std::make_shared<Primitive>(kBNTrainingReduceOpName)),
    utils::cast<AnfNodePtr>(iter_data_input0->second)};
  auto bn_training_reduce = func_graph->NewCNode(bn_training_reduce_inputs);
  MS_EXCEPTION_IF_NULL(bn_training_reduce);
  bn_training_reduce->set_scope(node->scope());
  // Set abstract
  auto iter_data_input1 = (*equiv).find(data_input1_var_);
  if (iter_data_input1 == (*equiv).end()) {
    MS_LOG(EXCEPTION) << "The equiv map is expected to contains the data_input1 var after matched.";
  }
  auto data_input1 = utils::cast<AnfNodePtr>(iter_data_input1->second);
  MS_EXCEPTION_IF_NULL(data_input1);
  auto iter_data_input2 = (*equiv).find(data_input2_var_);
  if (iter_data_input2 == (*equiv).end()) {
    MS_LOG(EXCEPTION) << "The equiv map is expected to contains the data_input2 var after matched.";
  }
  auto data_input2 = utils::cast<AnfNodePtr>(iter_data_input2->second);
  MS_EXCEPTION_IF_NULL(data_input2);
  AbstractBasePtrList abstract_list{data_input1->abstract(), data_input2->abstract()};
  auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(abstract_list);
  bn_training_reduce->set_abstract(abstract_tuple);
  return bn_training_reduce;
}

void FusedBatchNormFusion::GetBNTrainingUpdateInputs(const EquivPtr &equiv,
                                                     const std::vector<AnfNodePtr> &bn_training_reduce_outputs,
                                                     std::vector<AnfNodePtr> *bn_training_update_inputs) const {
  MS_EXCEPTION_IF_NULL(equiv);
  MS_EXCEPTION_IF_NULL(bn_training_update_inputs);
  auto iter_data_input0 = (*equiv).find(data_input0_var_);
  if (iter_data_input0 == (*equiv).end()) {
    MS_LOG(EXCEPTION) << "The equiv map is expected to contains the data_input0 var after matched.";
  }
  auto iter_data_input1 = (*equiv).find(data_input1_var_);
  if (iter_data_input1 == (*equiv).end()) {
    MS_LOG(EXCEPTION) << "The equiv map is expected to contains the data_input1 var after matched.";
  }
  auto iter_data_input2 = (*equiv).find(data_input2_var_);
  if (iter_data_input2 == (*equiv).end()) {
    MS_LOG(EXCEPTION) << "The equiv map is expected to contains the data_input2 var after matched.";
  }
  auto iter_variable_input0 = (*equiv).find(variable_input0_var_);
  if (iter_variable_input0 == (*equiv).end()) {
    MS_LOG(EXCEPTION) << "The equiv map is expected to contains the variable_input0 var after matched.";
  }
  auto iter_variable_input1 = (*equiv).find(variable_input1_var_);
  if (iter_variable_input1 == (*equiv).end()) {
    MS_LOG(EXCEPTION) << "The equiv map is expected to contains the variable_input1 var after matched.";
  }
  if (bn_training_reduce_outputs.size() != kBNTrainingReduceOutputNum) {
    MS_LOG(EXCEPTION) << "The output size of node bn_training_reduce must be " << kBNTrainingReduceOutputNum
                      << ", but it is " << bn_training_reduce_outputs.size();
  }
  *bn_training_update_inputs = {
    NewValueNode(std::make_shared<Primitive>(kBNTrainingUpdateOpName)),
    utils::cast<AnfNodePtr>(iter_data_input0->second),
    bn_training_reduce_outputs[0],
    bn_training_reduce_outputs[1],
    utils::cast<AnfNodePtr>(iter_data_input1->second),
    utils::cast<AnfNodePtr>(iter_data_input2->second),
    utils::cast<AnfNodePtr>(iter_variable_input0->second),
    utils::cast<AnfNodePtr>(iter_variable_input1->second),
  };
}

void FusedBatchNormFusion::GetBNTrainingUpdateAbstractList(const EquivPtr &equiv, const AnfNodePtr &bn,
                                                           std::vector<AbstractBasePtr> *abstract_list) const {
  MS_EXCEPTION_IF_NULL(equiv);
  MS_EXCEPTION_IF_NULL(bn);
  MS_EXCEPTION_IF_NULL(abstract_list);
  auto bn_abstract_tuple = dyn_cast<abstract::AbstractTuple>(bn->abstract());
  MS_EXCEPTION_IF_NULL(bn_abstract_tuple);
  if (bn_abstract_tuple->elements().size() < kBnOutputNum) {
    MS_LOG(EXCEPTION) << "The abstract size of node bn must not be less than " << kBnOutputNum << ", but it is "
                      << bn_abstract_tuple->elements().size();
  }
  auto iter_variable_input0 = (*equiv).find(variable_input0_var_);
  if (iter_variable_input0 == (*equiv).end()) {
    MS_LOG(EXCEPTION) << "The equiv map is expected to contains the variable_input0 var after matched.";
  }
  auto variable_input0 = utils::cast<AnfNodePtr>(iter_variable_input0->second);
  MS_EXCEPTION_IF_NULL(variable_input0);
  auto iter_variable_input1 = (*equiv).find(variable_input1_var_);
  if (iter_variable_input1 == (*equiv).end()) {
    MS_LOG(EXCEPTION) << "The equiv map is expected to contains the variable_input1 var after matched.";
  }
  auto variable_input1 = utils::cast<AnfNodePtr>(iter_variable_input1->second);
  MS_EXCEPTION_IF_NULL(variable_input1);
  *abstract_list = {bn_abstract_tuple->elements()[0], variable_input0->abstract(), variable_input1->abstract(),
                    bn_abstract_tuple->elements()[1], bn_abstract_tuple->elements()[2]};
}

AnfNodePtr FusedBatchNormFusion::CreateBNTrainingUpdate(
  const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &equiv,
  const std::vector<AnfNodePtr> &bn_training_reduce_outputs) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);
  // Set input
  std::vector<AnfNodePtr> bn_training_update_inputs;
  GetBNTrainingUpdateInputs(equiv, bn_training_reduce_outputs, &bn_training_update_inputs);
  auto bn_training_update = func_graph->NewCNode(bn_training_update_inputs);
  MS_EXCEPTION_IF_NULL(bn_training_update);
  // Set abstract
  auto iter_batch_norm = (*equiv).find(batch_norm_var_);
  if (iter_batch_norm == (*equiv).end()) {
    MS_LOG(EXCEPTION) << "The equiv map is expected to contains the batch_norm var after matched.";
  }
  AnfNodePtr bn = utils::cast<AnfNodePtr>(iter_batch_norm->second);
  MS_EXCEPTION_IF_NULL(bn);
  AbstractBasePtrList abstract_list;
  GetBNTrainingUpdateAbstractList(equiv, bn, &abstract_list);
  auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(abstract_list);
  bn_training_update->set_abstract(abstract_tuple);
  AnfAlgo::CopyNodeAttr(kAttrEpsilon, bn, bn_training_update);
  ValuePtr factor = GetFactor(equiv);
  if (factor == nullptr) {
    return nullptr;
  }
  AnfAlgo::SetNodeAttr(kAttrFactor, factor, bn_training_update);
  AnfAlgo::SetNodeAttr(kAttrIsRef, MakeValue(true), bn_training_update);
  bn_training_update->set_scope(node->scope());
  return bn_training_update;
}

const AnfNodePtr FusedBatchNormFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                               const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(equiv);
  MS_EXCEPTION_IF_NULL(node);
  AnfNodePtr bn_training_reduce = CreateBNTrainingReduce(func_graph, node, equiv);
  std::vector<AnfNodePtr> bn_training_reduce_outputs;
  CreateMultipleOutputsOfAnfNode(func_graph, bn_training_reduce, kBNTrainingReduceOutputNum,
                                 &bn_training_reduce_outputs);
  AnfNodePtr bn_training_update = CreateBNTrainingUpdate(func_graph, node, equiv, bn_training_reduce_outputs);
  if (bn_training_update == nullptr) {
    MS_LOG(DEBUG) << "Create BNTrainingUpdate failed for bn node " << node->DebugString();
    return nullptr;
  }
  std::vector<AnfNodePtr> bn_training_update_outputs;
  CreateMultipleOutputsOfAnfNode(func_graph, bn_training_update, kBNTrainingUpdateOutputNum,
                                 &bn_training_update_outputs);
  if (bn_training_update_outputs.size() < kBNTrainingUpdateOutputNum) {
    MS_LOG(EXCEPTION) << "The output size of node bn must be " << kBNTrainingUpdateOutputNum << ", but it is "
                      << bn_training_update_outputs.size();
  }
  // Replace old bn outputs with new outputs
  auto iter_batch_norm = (*equiv).find(batch_norm_var_);
  if (iter_batch_norm == (*equiv).end()) {
    MS_LOG(EXCEPTION) << "The equiv map is expected to contains the batch_norm var after matched.";
  }
  AnfNodePtr bn = utils::cast<AnfNodePtr>(iter_batch_norm->second);
  std::vector<AnfNodePtr> bn_outputs;
  GetBNOutput(func_graph, bn, &bn_outputs);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (const auto &output : bn_outputs) {
    MS_EXCEPTION_IF_NULL(output);
    if (!IsPrimitiveCNode(output, prim::kPrimTupleGetItem)) {
      continue;
    }
    auto tuple_getitem_cnode = output->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(tuple_getitem_cnode);
    AnfNodePtr index_node = tuple_getitem_cnode->input(kInputNodeOutputIndexInTupleGetItem);
    MS_EXCEPTION_IF_NULL(index_node);
    auto value_node = index_node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    int index = GetValue<int>(value_node->value());
    if (index == kReplaceOutputIndex0 || index == kReplaceOutputIndex1) {
      (void)manager->Replace(output, bn_training_update_outputs[index]);
    }
  }
  return bn_training_update_outputs[0];
}

const BaseRef FusedBatchNormMixPrecisionFusion::DefinePattern() const {
  std::shared_ptr<Var> Xs = std::make_shared<SeqVar>();
  VarPtr index0 = std::make_shared<CondVar>(IsC);
  VarPtr index1 = std::make_shared<CondVar>(IsC);
  VarPtr index2 = std::make_shared<CondVar>(IsC);
  VectorRef batch_norm = VectorRef({batch_norm_var_, data_input0_var_, data_input1_var_, data_input2_var_, Xs});
  VectorRef tuple_getitem0 = VectorRef({prim::kPrimTupleGetItem, batch_norm, index0});
  VectorRef tuple_getitem1 = VectorRef({prim::kPrimTupleGetItem, batch_norm, index1});
  VectorRef tuple_getitem2 = VectorRef({prim::kPrimTupleGetItem, batch_norm, index2});
  VectorRef cast_variable_input0 = VectorRef({prim::kPrimCast, variable_input0_var_});
  VectorRef cast_variable_input1 = VectorRef({prim::kPrimCast, variable_input1_var_});
  VectorRef sub0 = VectorRef({prim::kPrimSub, cast_variable_input0, tuple_getitem1});
  VectorRef sub1 = VectorRef({prim::kPrimSub, cast_variable_input1, tuple_getitem2});
  VectorRef mul0 = VectorRef({prim::kPrimMul, sub0, constant_input0_var_});
  VectorRef mul1 = VectorRef({prim::kPrimMul, sub1, constant_input1_var_});
  VectorRef cast2 = VectorRef({prim::kPrimCast, mul0});
  VectorRef cast3 = VectorRef({prim::kPrimCast, mul1});
  VectorRef assign_sub0 = VectorRef({prim::kPrimAssignSub, variable_input0_var_, cast2});
  VectorRef assign_sub1 = VectorRef({prim::kPrimAssignSub, variable_input1_var_, cast3});
  VectorRef depend0 = VectorRef({prim::kPrimDepend, tuple_getitem0, assign_sub0});
  return VectorRef({prim::kPrimDepend, depend0, assign_sub1});
}
}  // namespace opt
}  // namespace mindspore
