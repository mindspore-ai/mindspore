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
#include "plugin/device/ascend/optimizer/ir_fission/lamb_fission.h"
#include <memory>
#include <vector>
#include <string>
#include "include/common/utils/anfalgo.h"
#include "mindspore/core/ops/core_ops.h"
#include "backend/common/optimizer/optimizer.h"
#include "backend/common/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
namespace {
// Lamb's inputs: param, m, v, lr, beta1, beta2, eps, weight_decay, global_step, gradient
constexpr size_t kParamIndex = 1;
constexpr size_t kMIndex = 2;
constexpr size_t kVIndex = 3;
constexpr size_t kLearningRateIndex = 4;
constexpr size_t kBeta1Index = 5;
constexpr size_t kBeta2Index = 6;
constexpr size_t kEpsilonIndex = 7;
constexpr size_t kWeightDecayIndex = 8;
constexpr size_t kGlobalStepIndex = 9;
constexpr size_t kGradientIndex = 10;
constexpr size_t kUMonadIndex = 11;
constexpr size_t kLambInputNum = 10;
constexpr size_t kLambInputNumWithUMonad = 11;
constexpr size_t kLambApplyOptimizerAssignOutputNum = 3;
constexpr size_t kLambApplyOptimizerAssignUpdateIndex = 0;

AnfNodePtr CreateCastNode(const FuncGraphPtr &graph, const AnfNodePtr &input, const TypeId dst_type) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(input);
  if (common::AnfAlgo::GetOutputInferDataType(input, 0) != dst_type) {
    AnfNodePtr cast = graph->NewCNode({NewValueNode(std::make_shared<Primitive>(kCastOpName)), input});
    common::AnfAlgo::SetOutputTypeAndDetailShape({dst_type}, {AnfAlgo::GetOutputDetailShape(input, 0)}, cast.get());
    common::AnfAlgo::SetNodeAttr(kAttrDstType, MakeValue(static_cast<size_t>(dst_type)), cast);
    cast->set_scope(input->scope());
    return cast;
  }
  return input;
}

AnfNodePtr CreateNodeOfBinaryOp(const FuncGraphPtr &graph, const string &op_name, const AnfNodePtr &node1,
                                const AnfNodePtr &node2, const AnfNodePtr &node3) {
  std::vector<AnfNodePtr> new_node_inputs = {NewValueNode(std::make_shared<Primitive>(op_name)), node1, node2};
  return CreateNodeBase(graph, new_node_inputs, node3);
}

AnfNodePtr CreateUpdateStateNode(const FuncGraphPtr &graph, const bool is_need_update_state, const AnfNodePtr &node1,
                                 const AnfNodePtr &node2) {
  if (!is_need_update_state) {
    return nullptr;
  }

  std::vector<AnfNodePtr> new_node_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimUpdateState->name())),
                                             node1, node2};
  auto update_state_node = NewCNode(new_node_inputs, graph);
  MS_EXCEPTION_IF_NULL(update_state_node);

  update_state_node->set_kernel_info(std::make_shared<device::KernelInfo>());
  update_state_node->set_scope(node1->scope());
  update_state_node->set_abstract(node1->abstract());
  return update_state_node;
}

ValueNodePtr CreateValueNode(const FuncGraphPtr &graph, double value) {
  auto tensor = std::make_shared<tensor::Tensor>(value);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  ValueNodePtr value_node = kernel_graph->NewValueNode(tensor->ToAbstract(), tensor);
  kernel_graph->AddValueNodeToGraph(value_node);
  return value_node;
}

AnfNodePtr CreateLambApplyOptimizerAssignNode(const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &ori_inputs,
                                              const AnfNodePtr &param_fp32, const AnfNodePtr &gradient_fp32,
                                              const AnfNodePtr &new_global_step, const AnfNodePtr &weight_decay_flag,
                                              const AnfNodePtr &sub_beta1, const AnfNodePtr &sub_beta2,
                                              const bool is_exist_umonad_node, const AnfNodePtr &update_state_node) {
  std::vector<AnfNodePtr> new_node_inputs = {
    NewValueNode(std::make_shared<Primitive>(prim::kLambApplyOptimizerAssign->name())),
    gradient_fp32,
    ori_inputs[kVIndex],
    ori_inputs[kMIndex],
    param_fp32,
    ori_inputs[kBeta1Index],
    sub_beta1,
    ori_inputs[kBeta2Index],
    sub_beta2,
    ori_inputs[kEpsilonIndex],
    new_global_step,
    weight_decay_flag,
    ori_inputs[kWeightDecayIndex]};

  if (is_exist_umonad_node) {
    (void)new_node_inputs.emplace_back(update_state_node);
  }

  auto new_node = NewCNode(new_node_inputs, graph);
  MS_EXCEPTION_IF_NULL(new_node);

  new_node->set_kernel_info(std::make_shared<device::KernelInfo>());
  new_node->set_scope(ori_inputs[kMIndex]->scope());

  auto types = {common::AnfAlgo::GetOutputInferDataType(ori_inputs[kMIndex], 0),
                common::AnfAlgo::GetOutputInferDataType(ori_inputs[kGradientIndex], 0),
                common::AnfAlgo::GetOutputInferDataType(ori_inputs[kGradientIndex], 0)};
  auto shapes = {common::AnfAlgo::GetOutputInferShape(ori_inputs[kMIndex], 0),
                 common::AnfAlgo::GetOutputInferShape(ori_inputs[kGradientIndex], 0),
                 common::AnfAlgo::GetOutputInferShape(ori_inputs[kGradientIndex], 0)};
  common::AnfAlgo::SetOutputInferTypeAndShape(types, shapes, new_node.get());
  std::vector<AnfNodePtr> lamb_assign_outputs;
  CreateMultipleOutputsOfAnfNode(graph, new_node, kLambApplyOptimizerAssignOutputNum, &lamb_assign_outputs);
  if (lamb_assign_outputs.size() != kLambApplyOptimizerAssignOutputNum) {
    MS_LOG(EXCEPTION) << "The input tensor size[" << lamb_assign_outputs.size()
                      << "] of node [" + new_node->DebugString() + "] is not equal to "
                      << kLambApplyOptimizerAssignOutputNum << trace::DumpSourceLines(new_node);
  }
  return lamb_assign_outputs[kLambApplyOptimizerAssignUpdateIndex];
}

AnfNodePtr CreateLayerNormNode(const FuncGraphPtr &graph, const AnfNodePtr &input_node) {
  // Calc the sum of square
  const std::vector<AnfNodePtr> square_sum_node_inputs = {
    NewValueNode(std::make_shared<Primitive>(prim::kSquareSumV1->name())), input_node};
  auto square_sum_node = NewCNode(square_sum_node_inputs, graph);
  MS_EXCEPTION_IF_NULL(square_sum_node);

  std::vector<int64_t> axis;
  common::AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(axis), square_sum_node);
  common::AnfAlgo::SetNodeAttr(kAttrKeepDims, MakeValue(false), square_sum_node);
  square_sum_node->set_scope(input_node->scope());
  square_sum_node->set_abstract(input_node->abstract());

  auto types = {common::AnfAlgo::GetOutputInferDataType(input_node, 0)};
  ShapeVector shape = {1};
  common::AnfAlgo::SetOutputInferTypeAndShape(types, {shape}, square_sum_node.get());

  // Calc sqrt of the sum of square
  const std::vector<AnfNodePtr> sqrt_node_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimSqrt->name())),
                                                    square_sum_node};
  auto sqrt_node = NewCNode(sqrt_node_inputs, graph);
  MS_EXCEPTION_IF_NULL(sqrt_node);
  sqrt_node->set_scope(square_sum_node->scope());
  sqrt_node->set_abstract(square_sum_node->abstract());

  common::AnfAlgo::SetOutputInferTypeAndShape(types, {shape}, sqrt_node.get());
  return sqrt_node;
}

AnfNodePtr CreateLambApplyWeightAssignNode(const FuncGraphPtr &graph, const AnfNodePtr &w_norm,
                                           const AnfNodePtr &g_norm, const AnfNodePtr &lr, const AnfNodePtr &update,
                                           const AnfNodePtr &param, const bool is_exist_umonad_node,
                                           const AnfNodePtr &update_state_node) {
  std::vector<AnfNodePtr> new_node_inputs = {
    NewValueNode(std::make_shared<Primitive>(prim::kLambApplyWeightAssign->name())), w_norm, g_norm, lr, update, param};

  if (is_exist_umonad_node) {
    (void)new_node_inputs.emplace_back(update_state_node);
  }

  return CreateNodeBase(graph, new_node_inputs, param);
}
}  // namespace

const BaseRef LambFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimLamb, Xs});
}

const AnfNodePtr LambFission::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto lamb_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(lamb_cnode);
  auto real_input_num = common::AnfAlgo::GetInputNum(lamb_cnode);
  if (real_input_num < kLambInputNum) {
    MS_LOG(EXCEPTION) << "The input tensor size[" << real_input_num
                      << "] of node [" + lamb_cnode->DebugString() + "] is not equal to " << kLambInputNum
                      << trace::DumpSourceLines(lamb_cnode);
  }

  bool is_exist_umonad_node = false;
  const auto ori_inputs = lamb_cnode->inputs();
  AnfNodePtr param_node = nullptr;
  AnfNodePtr global_step_node = nullptr;
  AnfNodePtr update_state_load_node = nullptr;
  if (real_input_num == kLambInputNumWithUMonad && HasAbstractUMonad(ori_inputs[kUMonadIndex])) {
    is_exist_umonad_node = true;

    // param is a side-effect operator parameter, need load with UMonad
    param_node = CreateNodeOfBinaryOp(graph, prim::kPrimLoad->name(), ori_inputs[kParamIndex], ori_inputs[kUMonadIndex],
                                      ori_inputs[kParamIndex]);
    global_step_node = CreateNodeOfBinaryOp(graph, prim::kPrimLoad->name(), ori_inputs[kGlobalStepIndex],
                                            ori_inputs[kUMonadIndex], ori_inputs[kGlobalStepIndex]);

    // For multiple load scenarios, MakeTuple needs to be executed as the input parameter of UpdateState
    std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple), param_node, global_step_node};
    auto make_tuple_node = NewCNode(make_tuple_inputs, graph);
    MS_EXCEPTION_IF_NULL(make_tuple_node);

    // graph mode need umonad and update-state function to keep order
    update_state_load_node =
      CreateUpdateStateNode(graph, is_exist_umonad_node, ori_inputs[kUMonadIndex], make_tuple_node);
  } else {
    param_node = ori_inputs[kParamIndex];
    global_step_node = ori_inputs[kGlobalStepIndex];
  }

  // cast param to float32
  auto param_fp32 = CreateCastNode(graph, param_node, kNumberTypeFloat32);
  // cast grad to float32
  auto gradient_fp32 = CreateCastNode(graph, ori_inputs[kGradientIndex], kNumberTypeFloat32);
  // cast global stpe to float32
  auto new_global_step = CreateCastNode(graph, global_step_node, kNumberTypeFloat32);

  // cast delay flag to float32
  auto weight_decay_flag = CreateValueNode(graph, 1.0);

  auto num_one = CreateValueNode(graph, 1.0);
  // create 1-beta1
  auto sub_beta1 = CreateNodeOfBinaryOp(graph, kSubOpName, num_one, ori_inputs[kBeta1Index], ori_inputs[kBeta1Index]);
  // create 1-beta2
  auto sub_beta2 = CreateNodeOfBinaryOp(graph, kSubOpName, num_one, ori_inputs[kBeta2Index], ori_inputs[kBeta2Index]);

  auto update =
    CreateLambApplyOptimizerAssignNode(graph, ori_inputs, param_fp32, gradient_fp32, new_global_step, weight_decay_flag,
                                       sub_beta1, sub_beta2, is_exist_umonad_node, update_state_load_node);
  auto update_state_opt_assign_node =
    CreateUpdateStateNode(graph, is_exist_umonad_node, update_state_load_node, update);

  // create w_norm = op_norm(param_fp32)
  auto w_norm = CreateLayerNormNode(graph, param_fp32);
  // create g_norm = op_norm(update)
  auto g_norm = CreateLayerNormNode(graph, update);

  // param = op_lamb_apply_weight_assign(w_norm, g_norm, lr, update, param)
  auto lamb_node = CreateLambApplyWeightAssignNode(graph, w_norm, g_norm, ori_inputs[kLearningRateIndex], update,
                                                   param_node, is_exist_umonad_node, update_state_opt_assign_node);
  auto update_state_weight_assign_node =
    CreateUpdateStateNode(graph, is_exist_umonad_node, update_state_opt_assign_node, lamb_node);

  if (is_exist_umonad_node) {
    auto depend_node =
      CreateNodeOfBinaryOp(graph, prim::kPrimDepend->name(), lamb_node, update_state_weight_assign_node, lamb_node);
    return depend_node;
  }
  return lamb_node;
}
}  // namespace opt
}  // namespace mindspore
