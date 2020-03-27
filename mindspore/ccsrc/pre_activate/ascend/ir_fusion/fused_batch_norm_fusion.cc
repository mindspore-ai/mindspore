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
bool IsC(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    AnfNodePtr in = utils::cast<AnfNodePtr>(n);
    MS_EXCEPTION_IF_NULL(in);
    return in->isa<ValueNode>();
  }
  return false;
}

AnfNodePtr GetBatchNormNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto depend_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(depend_cnode);
  CheckCNodeInputSize(depend_cnode, kDependInputNum);
  AnfNodePtr assign_sub = depend_cnode->input(2);
  MS_EXCEPTION_IF_NULL(assign_sub);
  auto assign_sub_cnode = assign_sub->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(assign_sub_cnode);
  CheckCNodeInputSize(assign_sub_cnode, kAssignSubInputNum);
  AnfNodePtr mul = assign_sub_cnode->input(2);
  MS_EXCEPTION_IF_NULL(mul);
  auto mul_cnode = mul->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(mul_cnode);
  CheckCNodeInputSize(mul_cnode, kMulInputNum);
  AnfNodePtr sub = mul_cnode->input(1);
  MS_EXCEPTION_IF_NULL(sub);
  auto sub_cnode = sub->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(sub_cnode);
  CheckCNodeInputSize(sub_cnode, kSubInputNum);
  AnfNodePtr tuple_getitem = sub_cnode->input(2);
  MS_EXCEPTION_IF_NULL(tuple_getitem);
  auto tuple_getitem_cnode = tuple_getitem->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(tuple_getitem_cnode);
  CheckCNodeInputSize(tuple_getitem_cnode, kTupleGetitemInputNum);
  return tuple_getitem_cnode->input(1);
}

bool CompareTupleGetitem(const AnfNodePtr &n1, const AnfNodePtr &n2) {
  MS_EXCEPTION_IF_NULL(n1);
  MS_EXCEPTION_IF_NULL(n2);
  auto n1_cnode = n1->cast<CNodePtr>();
  auto n2_cnode = n2->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(n1_cnode);
  MS_EXCEPTION_IF_NULL(n2_cnode);
  auto index_input1 = n1_cnode->input(kInputNodeOutputIndexInTupleGetItem);
  MS_EXCEPTION_IF_NULL(index_input1);
  auto value_node1 = index_input1->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node1);
  auto index_input2 = n2_cnode->input(kInputNodeOutputIndexInTupleGetItem);
  MS_EXCEPTION_IF_NULL(index_input2);
  auto value_node2 = index_input2->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node2);
  return GetValue<int>(value_node1->value()) < GetValue<int>(value_node2->value());
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
  sort(bn_outputs->begin(), bn_outputs->end(), CompareTupleGetitem);
}
}  // namespace

const BaseRef FusedBatchNormFusion::DefinePattern() const {
  const auto prim_batch_norm = std::make_shared<Primitive>(kBatchNormOpName);
  std::shared_ptr<Var> Xs = std::make_shared<SeqVar>();
  VarPtr index0 = std::make_shared<CondVar>(IsC);
  VarPtr index1 = std::make_shared<CondVar>(IsC);
  VarPtr index2 = std::make_shared<CondVar>(IsC);
  VectorRef batch_norm = VectorRef({prim_batch_norm, data_input_var0_, data_input_var1_, data_input_var2_, Xs});
  VectorRef tuple_getitem0 = VectorRef({prim::kPrimTupleGetItem, batch_norm, index0});
  VectorRef tuple_getitem1 = VectorRef({prim::kPrimTupleGetItem, batch_norm, index1});
  VectorRef tuple_getitem2 = VectorRef({prim::kPrimTupleGetItem, batch_norm, index2});
  VectorRef sub0 = VectorRef({prim::kPrimSub, variable_input_var0_, tuple_getitem1});
  VectorRef sub1 = VectorRef({prim::kPrimSub, variable_input_var1_, tuple_getitem2});
  VectorRef mul0 = VectorRef({prim::kPrimMul, sub0, constant_input_var0_});
  VectorRef mul1 = VectorRef({prim::kPrimMul, sub1, constant_input_var1_});
  VectorRef assign_sub0 = VectorRef({prim::kPrimAssignSub, variable_input_var0_, mul0});
  VectorRef assign_sub1 = VectorRef({prim::kPrimAssignSub, variable_input_var1_, mul1});
  VectorRef depend0 = VectorRef({prim::kPrimDepend, tuple_getitem0, assign_sub0});
  return VectorRef({prim::kPrimDepend, depend0, assign_sub1});
}

abstract::AbstractTuplePtr FusedBatchNormFusion::CreateAbstractOfFusedBatchNorm(const EquivPtr &equiv,
                                                                                const AnfNodePtr &bn) const {
  MS_EXCEPTION_IF_NULL(equiv);
  MS_EXCEPTION_IF_NULL(bn);
  auto variable_input0 = utils::cast<AnfNodePtr>((*equiv)[variable_input_var0_]);
  MS_EXCEPTION_IF_NULL(variable_input0);
  auto variable_input1 = utils::cast<AnfNodePtr>((*equiv)[variable_input_var1_]);
  MS_EXCEPTION_IF_NULL(variable_input1);
  auto bn_abstract_tuple = dyn_cast<abstract::AbstractTuple>(bn->abstract());
  MS_EXCEPTION_IF_NULL(bn_abstract_tuple);
  if (bn_abstract_tuple->elements().size() != kBnOutputNum) {
    MS_LOG(EXCEPTION) << "The abstract size of node bn must be " << kBnOutputNum << ", but it is "
                      << bn_abstract_tuple->elements().size();
  }
  AbstractBasePtrList fused_bn_abstract_list{bn_abstract_tuple->elements()[0], variable_input0->abstract(),
                                             variable_input1->abstract(), bn_abstract_tuple->elements()[3],
                                             bn_abstract_tuple->elements()[4]};
  auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(fused_bn_abstract_list);
  return abstract_tuple;
}

ValuePtr FusedBatchNormFusion::GetFactor(const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(equiv);
  auto constant_input = utils::cast<AnfNodePtr>((*equiv)[constant_input_var0_]);
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
  auto *tensor_data = static_cast<float *>(tensor_ptr->data_c());
  MS_EXCEPTION_IF_NULL(tensor_data);
  return MakeValue(tensor_data[0]);
}

const AnfNodePtr FusedBatchNormFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                               const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(equiv);
  // Set inputs
  auto data_input0 = utils::cast<AnfNodePtr>((*equiv)[data_input_var0_]);
  MS_EXCEPTION_IF_NULL(data_input0);
  auto data_input1 = utils::cast<AnfNodePtr>((*equiv)[data_input_var1_]);
  MS_EXCEPTION_IF_NULL(data_input1);
  auto data_input2 = utils::cast<AnfNodePtr>((*equiv)[data_input_var2_]);
  MS_EXCEPTION_IF_NULL(data_input2);
  auto variable_input0 = utils::cast<AnfNodePtr>((*equiv)[variable_input_var0_]);
  MS_EXCEPTION_IF_NULL(variable_input0);
  auto variable_input1 = utils::cast<AnfNodePtr>((*equiv)[variable_input_var1_]);
  MS_EXCEPTION_IF_NULL(variable_input1);
  std::vector<AnfNodePtr> fused_bn_inputs = {
    NewValueNode(prim::kPrimFusedBatchNorm), data_input0, data_input1, data_input2, variable_input0, variable_input1};
  auto fused_bn = func_graph->NewCNode(fused_bn_inputs);
  fused_bn->set_scope(node->scope());
  MS_EXCEPTION_IF_NULL(fused_bn);
  // Set abstract
  AnfNodePtr bn = GetBatchNormNode(node);
  fused_bn->set_abstract(CreateAbstractOfFusedBatchNorm(equiv, bn));
  // Set attr
  AnfAlgo::CopyNodeAttr(kAttrEpsilon, bn, fused_bn);
  ValuePtr factor = GetFactor(equiv);
  if (factor == nullptr) {
    return nullptr;
  }
  AnfAlgo::SetNodeAttr(kAttrMomentum, factor, fused_bn);
  // Replace old nodes with outputs of fused_bn
  std::vector<AnfNodePtr> fused_bn_outputs;
  CreateMultipleOutputsOfAnfNode(func_graph, fused_bn, kBnOutputNum, &fused_bn_outputs);
  if (fused_bn_outputs.size() != kBnOutputNum) {
    MS_LOG(EXCEPTION) << "The output size of node bn must be " << kBnOutputNum << ", but it is "
                      << fused_bn_outputs.size();
  }
  std::vector<AnfNodePtr> bn_outputs;
  GetBNOutput(func_graph, bn, &bn_outputs);
  if (bn_outputs.size() != kBnOutputNum) {
    MS_LOG(EXCEPTION) << "The output size of node bn must be " << kBnOutputNum << ", but it is " << bn_outputs.size();
  }
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  (void)manager->Replace(bn_outputs[3], fused_bn_outputs[3]);
  (void)manager->Replace(bn_outputs[4], fused_bn_outputs[4]);
  return fused_bn_outputs[0];
}
}  // namespace opt
}  // namespace mindspore
