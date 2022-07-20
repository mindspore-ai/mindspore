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

#include "frontend/optimizer/irpass/ge/lamb_split.h"

#include "pybind_api/pybind_patch.h"
#include "pybind_api/ir/tensor_py.h"
#include "pipeline/pynative/base.h"
#include "pipeline/jit/static_analysis/prim.h"
#include "include/common/utils/python_adapter.h"
#include "mindspore/core/mindapi/ir/common.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"

namespace mindspore {
namespace opt {
namespace irpass {
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
constexpr char kOpsFunctionName[] = "mindspore.ops";
constexpr char kOpsOtherFunctionName[] = "mindspore.ops.operations.other_ops";
constexpr char kOpsInnerFunctionName[] = "mindspore.ops.operations.inner_ops";

void CreateMultiOutputsOfAnfNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, size_t output_num,
                                 std::vector<AnfNodePtr> *outputs) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(outputs);
  auto type_ptr = node->Type();
  auto shape_ptr = node->Shape();
  for (size_t i = 0; i < output_num; i++) {
    int64_t temp = SizeToLong(i);
    auto idx = NewValueNode(temp);
    MS_EXCEPTION_IF_NULL(idx);
    auto imm = std::make_shared<Int64Imm>(temp);
    auto abstract_scalar = std::make_shared<abstract::AbstractScalar>(imm);
    idx->set_abstract(abstract_scalar);
    auto tuple_getitem = func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), node, idx});
    tuple_getitem->set_abstract(idx->abstract());
    MS_EXCEPTION_IF_NULL(tuple_getitem);
    common::AnfAlgo::SetOutputInferTypeAndShape({common::AnfAlgo::GetOutputInferDataType(type_ptr, i)},
                                                {common::AnfAlgo::GetOutputInferShape(node, shape_ptr, i)},
                                                tuple_getitem.get());
    outputs->push_back(tuple_getitem);
  }
}

PrimitivePtr GetPrimitiveFromPyAdapter(const py::object &prim_obj) {
  const auto &adapter = py::cast<PrimitivePyAdapterPtr>(prim_obj);
  MS_EXCEPTION_IF_NULL(adapter);
  auto attached_prim = adapter->attached_primitive();
  if (attached_prim == nullptr) {
    attached_prim = std::make_shared<PrimitivePy>(prim_obj, adapter);
    adapter->set_attached_primitive(attached_prim);
  }
  return attached_prim->cast<PrimitivePtr>();
}

AnfNodePtr CreateCastNode(const FuncGraphPtr &graph, const AnfNodePtr &input, const TypeId dst_type) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(input);
  if (common::AnfAlgo::GetOutputInferDataType(input, 0) != dst_type) {
    auto op_obj = python_adapter::GetPyFn(kOpsFunctionName, kCastOpName)();
    auto op_prim = GetPrimitiveFromPyAdapter(op_obj);
    op_prim->set_attr(kAttrDstType, MakeValue(static_cast<size_t>(dst_type)));
    AnfNodePtr cast = graph->NewCNode(op_prim, {input});
    common::AnfAlgo::SetOutputTypeAndDetailShape({dst_type}, {common::AnfAlgo::GetOutputDetailShape(input, 0)},
                                                 cast.get());
    return cast;
  }
  return input;
}

AnfNodePtr CreateNodeOfBinaryOp(const FuncGraphPtr &graph, const string &op_name, const AnfNodePtr &node1,
                                const AnfNodePtr &node2, const AnfNodePtr &node3) {
  MS_EXCEPTION_IF_NULL(graph);
  auto op_obj = python_adapter::GetPyFn(kOpsFunctionName, op_name)();
  auto op_prim = GetPrimitiveFromPyAdapter(op_obj);
  auto new_node = graph->NewCNode(op_prim, {node1, node2});
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_abstract(node3->abstract());
  return new_node;
}

AnfNodePtr CreateUpdateStateNode(const FuncGraphPtr &graph, const bool is_need_update_state, const AnfNodePtr &node1,
                                 const AnfNodePtr &node2) {
  MS_EXCEPTION_IF_NULL(graph);
  if (!is_need_update_state) {
    return nullptr;
  }
  auto op_prim = NewValueNode(prim::kPrimUpdateState);
  auto update_state_node = graph->NewCNode({op_prim, node1, node2});
  MS_EXCEPTION_IF_NULL(update_state_node);
  update_state_node->set_abstract(node1->abstract());
  return update_state_node;
}

ValueNodePtr CreateValueNode(const ValuePtr &value_ptr, TypeId output_type) {
  MS_EXCEPTION_IF_NULL(value_ptr);
  auto new_node = std::make_shared<ValueNode>(value_ptr);
  MS_EXCEPTION_IF_NULL(new_node);
  auto value_abstract = value_ptr->ToAbstract();
  new_node->set_abstract(value_abstract);
  return new_node;
}

AnfNodePtr CreateLambApplyOptimizerAssignNode(const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &ori_inputs,
                                              const AnfNodePtr &param_fp32, const AnfNodePtr &gradient_fp32,
                                              const AnfNodePtr &new_global_step, const AnfNodePtr &weight_decay_flag,
                                              const AnfNodePtr &sub_beta1, const AnfNodePtr &sub_beta2,
                                              const bool is_exist_umonad_node, const AnfNodePtr &update_state_node) {
  MS_EXCEPTION_IF_NULL(graph);
  auto op_obj = python_adapter::GetPyFn(kOpsInnerFunctionName, prim::kLambApplyOptimizerAssign->name())();
  auto op_prim = GetPrimitiveFromPyAdapter(op_obj);
  std::vector<AnfNodePtr> new_node_inputs = {gradient_fp32,
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
  auto new_node = graph->NewCNode(op_prim, new_node_inputs);
  MS_EXCEPTION_IF_NULL(new_node);

  auto types = {common::AnfAlgo::GetOutputInferDataType(ori_inputs[kMIndex], 0),
                common::AnfAlgo::GetOutputInferDataType(ori_inputs[kGradientIndex], 0),
                common::AnfAlgo::GetOutputInferDataType(ori_inputs[kGradientIndex], 0)};
  auto shapes = {common::AnfAlgo::GetOutputInferShape(ori_inputs[kMIndex], 0),
                 common::AnfAlgo::GetOutputInferShape(ori_inputs[kGradientIndex], 0),
                 common::AnfAlgo::GetOutputInferShape(ori_inputs[kGradientIndex], 0)};
  common::AnfAlgo::SetOutputInferTypeAndShape(types, shapes, new_node.get());
  std::vector<AnfNodePtr> lamb_assign_outputs;
  (void)CreateMultiOutputsOfAnfNode(graph, new_node, kLambApplyOptimizerAssignOutputNum, &lamb_assign_outputs);
  if (lamb_assign_outputs.size() != kLambApplyOptimizerAssignOutputNum) {
    MS_LOG(EXCEPTION) << "The input tensor size[" << lamb_assign_outputs.size()
                      << "] of node [" + new_node->DebugString() + "] is not equal to "
                      << kLambApplyOptimizerAssignOutputNum << trace::DumpSourceLines(new_node);
  }
  return lamb_assign_outputs[kLambApplyOptimizerAssignUpdateIndex];
}

AnfNodePtr CreateSquareNode(const FuncGraphPtr &func_graph, const AnfNodePtr &inp, const ShapeVector &shape_vec,
                            const TypeId &type_id) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto op_obj = python_adapter::GetPyFn(kOpsFunctionName, kSquareOpName)();
  auto op_prim = GetPrimitiveFromPyAdapter(op_obj);
  auto square = func_graph->NewCNode(op_prim, {inp});
  MS_EXCEPTION_IF_NULL(square);
  auto abs = std::make_shared<abstract::AbstractTensor>(TypeIdToType(type_id), shape_vec);
  square->set_abstract(abs);
  return square;
}

AnfNodePtr CreateReduceSumNode(const FuncGraphPtr &func_graph, const AnfNodePtr &square, const AnfNodePtr &clip_by_norm,
                               const ShapeVector &shape_vec, const TypeId &type_id) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto op_obj = python_adapter::GetPyFn(kOpsFunctionName, kReduceSumOpName)();
  auto op_prim = GetPrimitiveFromPyAdapter(op_obj);
  // Sync the attribute of `ClipByNorm` to `ReduceSum`

  const auto dim = shape_vec.size();
  std::vector<int64_t> axis;
  for (size_t i = 0; i < dim; ++i) {
    (void)axis.emplace_back(i);
  }

  // Set abstract to `reduce_sum` op
  int64_t ddim = SizeToLong(dim);
  ShapeVector reduce_sum_output_shape = shape_vec;
  for (const auto &idx : axis) {
    if (idx < -ddim || idx >= ddim) {
      MS_EXCEPTION(ValueError) << "The range of axis value should in [" << -ddim << ", " << ddim
                               << "), but got: " << idx;
    }
    auto positive_idx = idx < 0 ? idx + ddim : idx;
    reduce_sum_output_shape[positive_idx] = 1;
  }
  auto abs = std::make_shared<abstract::AbstractTensor>(TypeIdToType(type_id), reduce_sum_output_shape);
  op_prim->set_attr(kAttrKeepDims, MakeValue<bool>(false));
  auto axis_node = CreateValueNode(MakeValue(axis), kNumberTypeInt64);
  auto reduce_sum = func_graph->NewCNode(op_prim, {square, axis_node});
  MS_EXCEPTION_IF_NULL(reduce_sum);
  reduce_sum->set_abstract(abs);
  return reduce_sum;
}

AnfNodePtr CreateLayerNormNode(const FuncGraphPtr &graph, const AnfNodePtr &input_node) {
  MS_EXCEPTION_IF_NULL(graph);
  // Calc the sum of square
  auto types = {common::AnfAlgo::GetOutputInferDataType(input_node, 0)};
  ShapeVector shape = {1};
  auto input = input_node->cast<CNodePtr>();
  auto shape_vec = common::AnfAlgo::GetOutputInferShape(input, 0);
  auto x_type_id = common::AnfAlgo::GetPrevNodeOutputInferDataType(input_node, 0);
  auto square = CreateSquareNode(graph, input_node, shape_vec, x_type_id);
  auto reduce_sum = CreateReduceSumNode(graph, square, input_node, shape_vec, x_type_id);
  auto sqrt_obj = python_adapter::GetPyFn(kOpsFunctionName, prim::kPrimSqrt->name())();
  auto sqrt_prim = GetPrimitiveFromPyAdapter(sqrt_obj);
  auto sqrt_node = graph->NewCNode(sqrt_prim, {reduce_sum});
  MS_EXCEPTION_IF_NULL(sqrt_node);
  sqrt_node->set_abstract(reduce_sum->abstract());
  common::AnfAlgo::SetOutputInferTypeAndShape(types, {shape}, sqrt_node.get());
  return sqrt_node;
}

AnfNodePtr CreateLambApplyWeightAssignNode(const FuncGraphPtr &graph, const AnfNodePtr &w_norm,
                                           const AnfNodePtr &g_norm, const AnfNodePtr &lr, const AnfNodePtr &update,
                                           const AnfNodePtr &param, const bool is_exist_umonad_node,
                                           const AnfNodePtr &update_state_node) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> new_node_inputs = {w_norm, g_norm, lr, update, param};
  if (is_exist_umonad_node) {
    (void)new_node_inputs.emplace_back(update_state_node);
  }
  auto op_obj = python_adapter::GetPyFn(kOpsInnerFunctionName, prim::kLambApplyWeightAssign->name())();
  auto op_prim = GetPrimitiveFromPyAdapter(op_obj);
  auto new_node = graph->NewCNode(op_prim, new_node_inputs);
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_abstract(param->abstract());
  return new_node;
}

AnfNodePtr CreateLoadOp(const FuncGraphPtr &graph, const string &op_name, const AnfNodePtr &node1,
                        const AnfNodePtr &node2, const AnfNodePtr &node3) {
  MS_EXCEPTION_IF_NULL(graph);
  auto op_prim = NewValueNode(prim::kPrimLoad);
  auto new_node = graph->NewCNode({op_prim, node1, node2});
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_abstract(node3->abstract());
  return new_node;
}

void SetScopeForNewNodes(const std::vector<AnfNodePtr> &nodes, const AnfNodePtr &lamb_node) {
  auto lamb_scope = lamb_node->scope();
  for (auto node : nodes) {
    node->set_scope(lamb_scope);
  }
}

const AnfNodePtr ProcessLambSplit(const FuncGraphPtr &graph, const AnfNodePtr &node) {
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
    param_node = CreateLoadOp(graph, prim::kPrimLoad->name(), ori_inputs[kParamIndex], ori_inputs[kUMonadIndex],
                              ori_inputs[kParamIndex]);
    global_step_node = CreateLoadOp(graph, prim::kPrimLoad->name(), ori_inputs[kGlobalStepIndex],
                                    ori_inputs[kUMonadIndex], ori_inputs[kGlobalStepIndex]);

    // For multiple load scenarios, MakeTuple needs to be executed as the input parameter of UpdateState
    std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple), param_node, global_step_node};
    auto make_tuple_node = NewCNode(make_tuple_inputs, graph);
    MS_EXCEPTION_IF_NULL(make_tuple_node);

    // graph mode need umonad and update-state function to keep order
    update_state_load_node =
      CreateUpdateStateNode(graph, is_exist_umonad_node, ori_inputs[kUMonadIndex], make_tuple_node);
    std::vector<AnfNodePtr> new_nodes{param_node, global_step_node, make_tuple_node, update_state_load_node};
    SetScopeForNewNodes(new_nodes, node);
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

  auto value_one = std::make_shared<tensor::Tensor>(1.0, kFloat32);
  // cast delay flag to float32
  auto weight_decay_flag = CreateValueNode(value_one, kNumberTypeFloat32);

  auto num_one = CreateValueNode(value_one, kNumberTypeFloat32);
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
  auto lamb_node =
    CreateLambApplyWeightAssignNode(graph, w_norm, g_norm, ori_inputs[kLearningRateIndex], update,
                                    ori_inputs[kParamIndex], is_exist_umonad_node, update_state_opt_assign_node);
  std::vector<AnfNodePtr> new_nodes{param_fp32,
                                    gradient_fp32,
                                    new_global_step,
                                    weight_decay_flag,
                                    num_one,
                                    sub_beta1,
                                    sub_beta2,
                                    update,
                                    update_state_opt_assign_node,
                                    w_norm,
                                    g_norm,
                                    lamb_node};
  SetScopeForNewNodes(new_nodes, node);
  auto mgr = graph->manager();
  MS_EXCEPTION_IF_NULL(mgr);
  const auto users = mgr->node_users()[node];
  for (auto &iter : users) {
    auto user_node = iter.first;
    auto name = GetCNodeFuncName(user_node->cast<CNodePtr>());
    if (name == prim::kPrimUpdateState->name()) {
      mgr->SetEdge(user_node, kIndex1, update_state_opt_assign_node);
      mgr->SetEdge(user_node, kIndex2, lamb_node);
    }
  }
  return lamb_node;
}
}  // namespace

AnfNodePtr LambForGE::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  Reset();
  FuncGraphPtr fg = node->func_graph();
  if (fg != nullptr && IsPrimitiveCNode(node, prim::kPrimLamb)) {
    return ProcessLambSplit(fg, node);
  }
  return nullptr;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
