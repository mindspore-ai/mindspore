/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fission/adam_weight_decay_fission.h"
#include <memory>
#include <string>
#include <vector>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/backend/optimizer/helper.h"
#include "include/common/utils/anfalgo.h"
#include "ops/array_op_name.h"
#include "ops/math_op_name.h"
#include "ops/nn_optimizer_ops.h"
#include "ops/sequence_ops.h"

namespace mindspore {
namespace opt {
namespace {
// AdamWeightDecay's inputs: param, m, v, lr, beta1, beta2, eps, weight_decay, gradient
constexpr size_t kIdxParam = 1;
constexpr size_t kIdxM = 2;
constexpr size_t kIdxV = 3;
constexpr size_t kIdxLr = 4;
constexpr size_t kIdxBeta1 = 5;
constexpr size_t kIdxBeta2 = 6;
constexpr size_t kIdxEps = 7;
constexpr size_t kIdxWeightDecay = 8;
constexpr size_t kIdxGradient = 9;
constexpr size_t kAamWeightDecayInputNum = 9;

AnfNodePtr CreateNodeOfBinaryOp(const FuncGraphPtr &graph, const string &op_name, const AnfNodePtr &node1,
                                const AnfNodePtr &node2) {
  std::vector<AnfNodePtr> new_node_inputs = {NewValueNode(std::make_shared<Primitive>(op_name)), node1, node2};
  return CreateNodeBase(graph, new_node_inputs, node2);
}

AnfNodePtr CreateNodeOfUnaryOp(const FuncGraphPtr &graph, const string &op_name, const AnfNodePtr &node) {
  std::vector<AnfNodePtr> new_node_inputs = {NewValueNode(std::make_shared<Primitive>(op_name)), node};
  return CreateNodeBase(graph, new_node_inputs, node);
}

ValueNodePtr CreateValueNode(const FuncGraphPtr &graph, double value) {
  auto tensor = std::make_shared<tensor::Tensor>(value);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  ValueNodePtr value_node = kernel_graph->NewValueNode(tensor->ToAbstract(), tensor);
  kernel_graph->AddValueNodeToGraph(value_node);
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
}  // namespace

const BaseRef AdamWeightDecayFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimAdamWeightDecay, Xs});
}

const AnfNodePtr AdamWeightDecayFission::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                 const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto adam_weight_decay_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(adam_weight_decay_cnode);
  CheckCNodeInputSize(adam_weight_decay_cnode, kAamWeightDecayInputNum);
  if (common::AnfAlgo::IsDynamicShape(adam_weight_decay_cnode)) {
    MS_LOG(EXCEPTION) << "AdamWeightDecay don't support dynamic shape, node: "
                      << adam_weight_decay_cnode->fullname_with_scope();
  }

  const auto ori_inputs = adam_weight_decay_cnode->inputs();

  // cast param to float32
  auto param_fp32 = CreateCastNode(graph, ori_inputs[kIdxParam], kNumberTypeFloat32);
  auto m_fp32 = CreateCastNode(graph, ori_inputs[kIdxM], kNumberTypeFloat32);
  auto v_fp32 = CreateCastNode(graph, ori_inputs[kIdxV], kNumberTypeFloat32);
  auto grad_fp32 = CreateCastNode(graph, ori_inputs[kIdxGradient], kNumberTypeFloat32);

  // create beta1 * m
  auto mul_1 = CreateNodeOfBinaryOp(graph, kMulOpName, ori_inputs[kIdxBeta1], m_fp32);
  // create 1-beta1
  auto num_one = CreateValueNode(graph, 1.0);
  auto sub_1 = CreateNodeOfBinaryOp(graph, kSubOpName, num_one, ori_inputs[kIdxBeta1]);
  // create (1-beta1) * gradient
  auto mul_2 = CreateNodeOfBinaryOp(graph, kMulOpName, sub_1, grad_fp32);
  // create next_m = beta1 * m + (1 - beat1) * gradient
  auto add_1 = CreateNodeOfBinaryOp(graph, kTensorAddOpName, mul_1, mul_2);

  // create beta2 * v
  auto mul_3 = CreateNodeOfBinaryOp(graph, kMulOpName, ori_inputs[kIdxBeta2], v_fp32);
  // create gradient^2
  auto square = CreateNodeOfUnaryOp(graph, kSquareOpName, grad_fp32);
  // create 1-beta2
  auto sub_2 = CreateNodeOfBinaryOp(graph, kSubOpName, num_one, ori_inputs[kIdxBeta2]);
  // create (1-beta2) * gradient^2
  auto mul_4 = CreateNodeOfBinaryOp(graph, kMulOpName, sub_2, square);
  // create next_v = beta2 * v + (1 - beta2) * gradient^2
  auto add_2 = CreateNodeOfBinaryOp(graph, kTensorAddOpName, mul_3, mul_4);

  // create sqrt(next_v)
  auto sqrt = CreateNodeOfUnaryOp(graph, kSqrtOpName, add_2);
  // create eps + sqrt(next_v)
  auto add_3 = CreateNodeOfBinaryOp(graph, kTensorAddOpName, ori_inputs[kIdxEps], sqrt);
  // create update = next_m / (eps + sqrt(next_v))
  auto real_div = CreateNodeOfBinaryOp(graph, kRealDivOpName, add_1, add_3);
  // create weight_decay * param
  auto mul_5 = CreateNodeOfBinaryOp(graph, kMulOpName, ori_inputs[kIdxWeightDecay], param_fp32);
  // create update <== weight_decay * param + update
  auto add_4 = CreateNodeOfBinaryOp(graph, kTensorAddOpName, mul_5, real_div);
  // create update_with_lr = lr * update
  auto mul_6 = CreateNodeOfBinaryOp(graph, kMulOpName, ori_inputs[kIdxLr], add_4);
  // create param - update_with_lr
  auto sub_3 = CreateNodeOfBinaryOp(graph, kSubOpName, param_fp32, mul_6);

  // create param = param - update_with_lr
  auto assign_1 = CreateNodeOfBinaryOp(graph, prim::kPrimAssign->name(), param_fp32, sub_3);
  // create m = next_m
  auto assign_2 = CreateNodeOfBinaryOp(graph, prim::kPrimAssign->name(), m_fp32, add_1);
  // create v = next_v
  auto assign_3 = CreateNodeOfBinaryOp(graph, prim::kPrimAssign->name(), v_fp32, add_2);

  return CreateMakeTupleNode(graph, std::vector<AnfNodePtr>{assign_1, assign_2, assign_3});
}
}  // namespace opt
}  // namespace mindspore
