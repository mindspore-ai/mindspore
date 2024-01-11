/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/mindir/adam_weight_decay_unify_mindir.h"
#include "ops/nn_optimizer_ops.h"
#include "ops/math_ops.h"
#include "ops/framework_ops.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kAdamWeightDecayInputNum = 12;
const std::vector<size_t> kdamWeightDecayIndexMapping = {9, 3, 2, 1, 4, 5, 12, 6, 13, 8, 7};

ValueNodePtr CreateValueNode(const FuncGraphPtr &graph, double value) {
  auto tensor = std::make_shared<tensor::Tensor>(value);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  ValueNodePtr value_node = kernel_graph->NewValueNode(tensor->ToAbstract(), tensor);
  return value_node;
}

AnfNodePtr CreateCastNode(const FuncGraphPtr &graph, const AnfNodePtr &input, const TypeId dst_type) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(input);
  if (common::AnfAlgo::GetOutputInferDataType(input, 0) != dst_type) {
    AnfNodePtr cast = graph->NewCNode({NewValueNode(std::make_shared<Primitive>(kCastOpName)), input});
    MS_EXCEPTION_IF_NULL(cast);
    common::AnfAlgo::SetOutputTypeAndDetailShape({dst_type}, {AnfAlgo::GetOutputDetailShape(input, 0)}, cast.get());
    common::AnfAlgo::SetNodeAttr(kAttrDstType, TypeIdToType(dst_type), cast);
    cast->set_scope(input->scope());
    return cast;
  }
  return input;
}

AnfNodePtr CreateSubCNode(const FuncGraphPtr &graph, const AnfNodePtr &src_node, const AnfNodePtr &dst_node) {
  MS_EXCEPTION_IF_NULL(src_node);
  MS_EXCEPTION_IF_NULL(dst_node);
  std::vector<AnfNodePtr> new_node_inputs = {NewValueNode(std::make_shared<Primitive>(kSubOpName)), src_node, dst_node};
  return CreateNodeBase(graph, new_node_inputs, dst_node);
}

AnfNodePtr CreateAssignCNode(const FuncGraphPtr &graph, const AnfNodePtr &src_node, const AnfNodePtr &dst_node) {
  MS_EXCEPTION_IF_NULL(src_node);
  MS_EXCEPTION_IF_NULL(dst_node);
  std::vector<AnfNodePtr> new_node_inputs = {NewValueNode(std::make_shared<Primitive>(kAssignOpName)), src_node,
                                             dst_node};
  return CreateNodeBase(graph, new_node_inputs, dst_node);
}
}  // namespace

const BaseRef AdamWeightDecayUnifyMindIR::DefinePattern() const {
  auto Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimAdamWeightDecay, Xs});
}

const AnfNodePtr AdamWeightDecayUnifyMindIR::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                     const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto input_list = cnode->inputs();
  if (input_list.size() != kAdamWeightDecayInputNum) {
    MS_LOG(EXCEPTION) << "AdamWeightDecay's input size must be " << kAdamWeightDecayInputNum << ", but got "
                      << input_list.size();
  }

  // Create New node
  PrimitivePtr prim = std::make_shared<Primitive>(kAdamApplyOneWithDecayOpName);
  std::vector<AnfNodePtr> new_node_inputs = {NewValueNode(prim)};

  auto num_one = CreateValueNode(func_graph, 1.0);
  // 1 - beta1
  auto beta1_sub = CreateSubCNode(func_graph, num_one, input_list[kIndex5]);
  input_list.push_back(beta1_sub);
  // 1 - beta2
  auto beta2_sub = CreateSubCNode(func_graph, num_one, input_list[kIndex6]);
  input_list.push_back(beta2_sub);
  // Cast
  auto ori_param = input_list[kIndex1];
  auto param_fp32 = CreateCastNode(func_graph, input_list[kIndex1], kNumberTypeFloat32);
  input_list[kIndex1] = param_fp32;
  auto grad_fp32 = CreateCastNode(func_graph, input_list[kIndex9], kNumberTypeFloat32);
  input_list[kIndex9] = grad_fp32;

  // Mapping ms index to ge index.
  for (size_t i = 0; i < kdamWeightDecayIndexMapping.size(); ++i) {
    const auto &cur_node = input_list[kdamWeightDecayIndexMapping[i]];
    (void)new_node_inputs.emplace_back(cur_node);
  }

  // Create New AdamApplyOneWithDecay with three outputs.
  auto new_cnode = NewCNode(new_node_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(new_cnode);
  new_cnode->set_scope(node->scope());
  AbstractBasePtrList new_node_abstract_list;
  new_node_abstract_list.push_back(input_list[kIndex3]->abstract());
  new_node_abstract_list.push_back(input_list[kIndex2]->abstract());
  new_node_abstract_list.push_back(param_fp32->abstract());
  auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(new_node_abstract_list);
  new_cnode->set_abstract(abstract_tuple);
  std::vector<AnfNodePtr> new_cnode_outputs;
  CreateMultipleOutputsOfAnfNode(func_graph, new_cnode, kAdamApplyOneOutputNum, &new_cnode_outputs);
  if (new_cnode_outputs.size() != kAdamApplyOneOutputNum) {
    MS_LOG(INTERNAL_EXCEPTION) << "The output size of node " << new_cnode->DebugString() << " should be "
                               << kAdamApplyOneOutputNum << trace::DumpSourceLines(node);
  }

  // Create assign.
  auto update_param =
    CreateCastNode(func_graph, new_cnode_outputs[kIndex2], common::AnfAlgo::GetOutputInferDataType(ori_param, 0));
  auto assign_param = CreateAssignCNode(func_graph, ori_param, update_param);
  auto assign_m = CreateAssignCNode(func_graph, input_list[kIndex2], new_cnode_outputs[kIndex1]);
  auto assign_v = CreateAssignCNode(func_graph, input_list[kIndex3], new_cnode_outputs[kIndex0]);
  return CreateMakeTupleNode(func_graph, std::vector<AnfNodePtr>{assign_param, assign_m, assign_v});
}
}  // namespace opt
}  // namespace mindspore
