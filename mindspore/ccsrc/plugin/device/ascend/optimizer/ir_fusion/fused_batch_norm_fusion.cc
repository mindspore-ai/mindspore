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
#include "plugin/device/ascend/optimizer/ir_fusion/fused_batch_norm_fusion.h"
#include <memory>
#include <algorithm>
#include "backend/common/optimizer/helper.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "utils/trace_base.h"

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
    MS_LOG(EXCEPTION) << "The bn node " << bn->DebugString() << " should has some outputs"
                      << trace::DumpSourceLines(bn);
  }
  for (const auto &node_index : manager->node_users()[bn]) {
    const AnfNodePtr &output = node_index.first;
    MS_EXCEPTION_IF_NULL(output);
    bn_outputs->push_back(output);
  }
}
}  // namespace

ValuePtr FusedBatchNormFusion::GetFactor(const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(equiv);
  auto constant_input = GetAnfNodeByVar(equiv, constant_input0_var_);
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
    auto *half_data = static_cast<const float16 *>(tensor_ptr->data_c());
    MS_EXCEPTION_IF_NULL(half_data);
    float float_data = half_to_float(half_data[0]);
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
  std::vector<AnfNodePtr> bn_training_reduce_inputs = {
    NewValueNode(std::make_shared<Primitive>(kBNTrainingReduceOpName)), GetAnfNodeByVar(equiv, data_input0_var_)};
  auto bn_training_reduce = NewCNode(bn_training_reduce_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(bn_training_reduce);
  bn_training_reduce->set_scope(node->scope());
  // Set abstract
  auto data_input1 = GetAnfNodeByVar(equiv, data_input1_var_);
  MS_EXCEPTION_IF_NULL(data_input1);
  auto data_input2 = GetAnfNodeByVar(equiv, data_input2_var_);
  MS_EXCEPTION_IF_NULL(data_input2);
  AbstractBasePtrList abstract_list{data_input1->abstract(), data_input2->abstract()};
  auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(abstract_list);
  bn_training_reduce->set_abstract(abstract_tuple);
  return bn_training_reduce;
}

void FusedBatchNormFusion::GetBNTrainingUpdateInputs(const EquivPtr &equiv,
                                                     const std::vector<AnfNodePtr> &bn_training_reduce_outputs,
                                                     std::vector<AnfNodePtr> *const bn_training_update_inputs) const {
  MS_EXCEPTION_IF_NULL(equiv);
  MS_EXCEPTION_IF_NULL(bn_training_update_inputs);
  *bn_training_update_inputs = {
    NewValueNode(std::make_shared<Primitive>(kBNTrainingUpdateOpName)),
    utils::cast<AnfNodePtr>(GetAnfNodeByVar(equiv, data_input0_var_)),
    bn_training_reduce_outputs[0],
    bn_training_reduce_outputs[1],
    GetAnfNodeByVar(equiv, data_input1_var_),
    GetAnfNodeByVar(equiv, data_input2_var_),
    GetAnfNodeByVar(equiv, variable_input0_var_),
    GetAnfNodeByVar(equiv, variable_input1_var_),
  };
}

void FusedBatchNormFusion::GetBNTrainingUpdateAbstractList(const EquivPtr &equiv, const AnfNodePtr &bn,
                                                           std::vector<AbstractBasePtr> *const abstract_list) const {
  MS_EXCEPTION_IF_NULL(equiv);
  MS_EXCEPTION_IF_NULL(bn);
  MS_EXCEPTION_IF_NULL(abstract_list);
  auto bn_abstract_tuple = dyn_cast<abstract::AbstractTuple>(bn->abstract());
  MS_EXCEPTION_IF_NULL(bn_abstract_tuple);
  if (bn_abstract_tuple->elements().size() < kBnOutputNum) {
    MS_LOG(EXCEPTION) << "The abstract size of node bn must not be less than " << kBnOutputNum << ", but it is "
                      << bn_abstract_tuple->elements().size() << trace::DumpSourceLines(bn);
  }
  auto variable_input0 = GetAnfNodeByVar(equiv, variable_input0_var_);
  auto variable_input1 = GetAnfNodeByVar(equiv, variable_input1_var_);
  MS_EXCEPTION_IF_NULL(variable_input0);
  MS_EXCEPTION_IF_NULL(variable_input1);
  *abstract_list = {bn_abstract_tuple->elements()[kIndex0], variable_input0->abstract(), variable_input1->abstract(),
                    bn_abstract_tuple->elements()[kIndex1], bn_abstract_tuple->elements()[kIndex2]};
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
  auto bn_training_update = NewCNode(bn_training_update_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(bn_training_update);
  // Set abstract
  AnfNodePtr bn = GetAnfNodeByVar(equiv, batch_norm_var_);
  AbstractBasePtrList abstract_list;
  GetBNTrainingUpdateAbstractList(equiv, bn, &abstract_list);
  auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(abstract_list);
  bn_training_update->set_abstract(abstract_tuple);
  common::AnfAlgo::CopyNodeAttr(kAttrEpsilon, bn, bn_training_update);
  ValuePtr factor = GetFactor(equiv);
  if (factor == nullptr) {
    return nullptr;
  }
  common::AnfAlgo::SetNodeAttr(kAttrFactor, factor, bn_training_update);
  common::AnfAlgo::SetNodeAttr(kAttrIsRef, MakeValue(true), bn_training_update);
  bn_training_update->set_scope(node->scope());
  return bn_training_update;
}

void FusedBatchNormFusion::EliminateMonadNodes(const FuncGraphPtr &func_graph, const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(equiv);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto assign_sub1 = GetAnfNodeByVar(equiv, assign_sub1_var_);
  MS_EXCEPTION_IF_NULL(assign_sub1);
  auto users = manager->node_users()[assign_sub1];
  for (const auto &node_index : users) {
    const AnfNodePtr &output = node_index.first;
    MS_EXCEPTION_IF_NULL(output);
    if (common::AnfAlgo::CheckPrimitiveType(output, prim::kPrimUpdateState)) {
      (void)manager->Replace(output, GetAnfNodeByVar(equiv, monad0_var_));
      break;
    }
  }
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
                      << bn_training_update_outputs.size() << trace::DumpSourceLines(node);
  }
  // Replace old bn outputs with new outputs
  std::vector<AnfNodePtr> bn_outputs;
  GetBNOutput(func_graph, GetAnfNodeByVar(equiv, batch_norm_var_), &bn_outputs);
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
    auto value_index = GetValue<int64_t>(value_node->value());
    if (value_index < 0) {
      MS_LOG(EXCEPTION) << "Error value index: " << value_index;
    }
    auto index = LongToSize(value_index);
    if (index == kReplaceOutputIndex0 || index == kReplaceOutputIndex1) {
      (void)manager->Replace(output, bn_training_update_outputs[index]);
    }
  }
  (void)manager->Replace(node, bn_training_update_outputs[0]);
  EliminateMonadNodes(func_graph, equiv);
  return nullptr;
}

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
  VectorRef assign_sub0 = VectorRef({assign_sub0_var_, variable_input0_var_, mul0, monad0_var_});
  VectorRef assign_sub1 = VectorRef({assign_sub1_var_, variable_input1_var_, mul1, monad1_var_});
  VectorRef depend0 = VectorRef({prim::kPrimDepend, tuple_getitem0, assign_sub0});
  return VectorRef({prim::kPrimDepend, depend0, assign_sub1});
}

const BaseRef FusedBatchNormMixPrecisionFusion0::DefinePattern() const {
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
  VectorRef assign_sub0 = VectorRef({assign_sub0_var_, variable_input0_var_, cast2, monad0_var_});
  VectorRef assign_sub1 = VectorRef({assign_sub1_var_, variable_input1_var_, cast3, monad1_var_});
  VectorRef depend0 = VectorRef({prim::kPrimDepend, tuple_getitem0, assign_sub0});
  return VectorRef({prim::kPrimDepend, depend0, assign_sub1});
}

const BaseRef FusedBatchNormMixPrecisionFusion1::DefinePattern() const {
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
  VectorRef cast0 = VectorRef({prim::kPrimCast, sub0});
  VectorRef cast1 = VectorRef({prim::kPrimCast, sub1});
  VectorRef mul0 = VectorRef({prim::kPrimMul, cast0, constant_input0_var_});
  VectorRef mul1 = VectorRef({prim::kPrimMul, cast1, constant_input1_var_});
  VectorRef assign_sub0 = VectorRef({assign_sub0_var_, variable_input0_var_, mul0, monad0_var_});
  VectorRef assign_sub1 = VectorRef({assign_sub1_var_, variable_input1_var_, mul1, monad1_var_});
  VectorRef depend0 = VectorRef({prim::kPrimDepend, tuple_getitem0, assign_sub0});
  return VectorRef({prim::kPrimDepend, depend0, assign_sub1});
}
}  // namespace opt
}  // namespace mindspore
