/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/matmul_allreduce_add_rmsnorm_fusion.h"

#include <vector>
#include <string>

#include "ir/tensor_new.h"
#include "plugin/ascend/res_manager/collective/multi_ascend_collective_comm_lib.h"
#include "include/runtime/hardware_abstract/collective/collective_manager.h"
#include "primitive/other_ops.h"
#include "backend/common/pass/common/gllo_utils.h"
#include "plugin/ascend/res_manager/hal_manager/ascend_hal_manager.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "include/backend/common/pass_manager/helper.h"
#include "utils/ms_context.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace opt {
static const std::vector<uint32_t> supported_rank_size{1, 2, 4, 8};
const BaseRef MatMulAllReduceAddRmsNormFusion::DefinePattern() const {
  auto transpose_a = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(transpose_a != nullptr, {});
  auto transpose_b = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(transpose_b != nullptr, {});
  VectorRef matmul = VectorRef({prim::kPrimMatMul, x1_, x2_, transpose_a, transpose_b});
  VectorRef allreduce = VectorRef({prim::kPrimAllReduce, matmul});
  VarPtr shape_tuple = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(shape_tuple != nullptr, {});
  VectorRef make_tuple = VectorRef({prim::kPrimMakeTuple, shape_tuple});
  VectorRef reshape = VectorRef({prim::kPrimReshape, allreduce, make_tuple});
  VectorRef add = VectorRef({prim::kPrimAdd, residual_, reshape});
  VectorRef rmsnorm = VectorRef({prim::kPrimRmsNorm, add, gamma_, eps_});
  return rmsnorm;
}

bool MatMulAllReduceAddRmsNormFusion::IsSupport(const AnfNodePtr &node, const FuncGraphPtr &graph) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);

  if (common::IsExecuteSimulation()) {
    MS_LOG(INFO) << "Not support MatMulAllReduceAddRmsNormFusion when MS_SIMULATION_LEVEL=3.";
    return false;
  }

  if (!ms_context->IsEnableInferBoost()) {
    MS_LOG(INFO) << "for MatMulAllReduceAddRmsNormFusion ops, infer_boost must be enabled.";
    return false;
  }

  bool is_enable_lccl = device::ascend::AscendHalManager::GetInstance().EnableLccl();
  if (is_enable_lccl) {
    MS_LOG(INFO) << "disable MatMulAllReduceAddRmsNormFusion when lccl is enabled.";
    return false;
  }

  // only support ascend910b
  auto soc_version = ms_context->ascend_soc_version();
  if (soc_version != kAscendVersion910b) {
    MS_LOG(INFO) << "MatMulAllReduceAddRmsNorm does not support soc version " << soc_version;
    return false;
  }

  auto add_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), kIndex0);
  auto reshape_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(add_node), kIndex1);
  auto allreduce_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(reshape_node), kIndex0);
  auto matmul_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(allreduce_node), kIndex0);

  auto matmul_cnode = matmul_node->cast<CNodePtr>();
  auto transpose_a = matmul_cnode->input(kIndex3)->cast<ValueNodePtr>();
  auto is_transpose_a = GetValue<bool>(transpose_a->value());
  if (is_transpose_a) {
    MS_LOG(INFO) << "only support transpose_a=False, but got transpose_a=True.";
    return false;
  }

  auto x1_type = common::AnfAlgo::GetPrevNodeOutputInferDataType(matmul_node, kIndex0);
  auto x2_type = common::AnfAlgo::GetPrevNodeOutputInferDataType(matmul_node, kIndex1);
  auto residual_type = common::AnfAlgo::GetPrevNodeOutputInferDataType(add_node, kIndex0);
  auto gamma_type = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, kIndex1);
  if ((x1_type != kNumberTypeFloat16 && x1_type != kNumberTypeBFloat16) ||
      (x2_type != kNumberTypeFloat16 && x2_type != kNumberTypeBFloat16) ||
      (residual_type != kNumberTypeFloat16 && residual_type != kNumberTypeBFloat16) ||
      (gamma_type != kNumberTypeFloat16 && gamma_type != kNumberTypeBFloat16)) {
    MS_LOG(INFO) << "input dtype does not support, x1_dtype = " << x1_type << ", x2_dtype = " << x2_type
                 << ", residual_dtype = " << residual_type << ", gamma_dtype = " << gamma_type;
    return false;
  }

  auto x1_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(matmul_node, kIndex0);
  auto x2_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(matmul_node, kIndex1);
  auto residual_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(add_node, kIndex0);
  if (residual_shape.size() != kSizeThree) {
    MS_LOG(INFO) << "only support residual with three dimensions.";
    return false;
  }
  if ((x1_shape.size() != kSizeTwo && x1_shape.size() != kSizeThree) || x2_shape.size() != kSizeTwo) {
    MS_LOG(INFO) << "only support x1 two or three dimensions and x2 two dimensions, but got x1 " << x1_shape.size()
                 << " dimensions and x2 " << x2_shape.size() << " dimensions.";
    return false;
  }

  return true;
}

std::vector<std::string> MatMulAllReduceAddRmsNormFusion::MustExistPrimitiveName() const {
  std::vector<std::string> ret;
  ret.emplace_back(prim::kPrimAllReduce->name());
  ret.emplace_back(prim::kPrimRmsNorm->name());
  ret.emplace_back(prim::kPrimMatMul->name());
  return ret;
}

AnfNodePtr MatMulAllReduceAddRmsNormFusion::NewTransposeNode(const FuncGraphPtr &func_graph, const AnfNodePtr &x2,
                                                             const AnfNodePtr &node,
                                                             const TypeId &add_result_type) const {
  MS_LOG(INFO) << "start to create Transpose node.";
  auto prim = std::make_shared<Primitive>(ops::kNameTranspose);
  ShapeVector perm_vec{1, 0};
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), x2, CreateShapeValueNode(func_graph, perm_vec, false)};
  auto transpose_cnode = NewCNode(inputs, func_graph);
  MS_EXCEPTION_IF_NULL(transpose_cnode);

  std::vector<TypeId> transpose_types;
  std::vector<BaseShapePtr> transpose_shapes;
  ShapeVector x2_shape_vector = x2->Shape()->GetShapeVector();
  std::reverse(x2_shape_vector.begin(), x2_shape_vector.end());
  auto transpose_shape = x2->Shape()->Clone();
  MS_EXCEPTION_IF_NULL(transpose_shape);
  transpose_shape->SetShapeVector(x2_shape_vector);
  transpose_types.push_back(add_result_type);
  transpose_shapes.push_back(transpose_shape);

  common::AnfAlgo::SetOutputTypeAndDetailShape(transpose_types, transpose_shapes, transpose_cnode.get());
  transpose_cnode->set_scope(node->scope());
  auto build_info = GenerateKernelBuildInfo(transpose_cnode);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, transpose_cnode.get());
  MS_LOG(INFO) << "create Transpose node success.";
  return transpose_cnode;
}

CNodePtr MatMulAllReduceAddRmsNormFusion::CreateMatMulAllReduceAddRmsNormNode(const FuncGraphPtr &func_graph,
                                                                              const AnfNodePtr &node,
                                                                              const EquivPtr &equiv,
                                                                              const TypeId &add_result_type) const {
  MS_LOG(INFO) << "start to create MatMulAllReduceAddRmsNorm node.";
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);

  auto x1 = utils::cast<AnfNodePtr>((*equiv)[x1_]);
  auto x2 = utils::cast<AnfNodePtr>((*equiv)[x2_]);

  // create empty bias node
  TypeId bias_tensor_type = kNumberTypeFloat16;
  std::vector<int64_t> bias_tensor_shape = {0};
  auto empty_bias_tensor = tensor::from_spec(bias_tensor_type, bias_tensor_shape, device::DeviceType::kCPU);
  auto bias = CreateValueNodeWithKernelInfo(func_graph, empty_bias_tensor);

  auto residual = utils::cast<AnfNodePtr>((*equiv)[residual_]);
  auto gamma = utils::cast<AnfNodePtr>((*equiv)[gamma_]);
  auto eps = utils::cast<AnfNodePtr>((*equiv)[eps_]);

  auto add_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), kIndex0);
  auto reshape_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(add_node), kIndex1);
  auto allreduce_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(reshape_node), kIndex0);
  auto matmul_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(allreduce_node), kIndex0);

  auto allreduce_cnode = allreduce_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(allreduce_cnode);
  auto allreduce_prim = GetCNodePrimitive(allreduce_cnode);
  MS_EXCEPTION_IF_NULL(allreduce_prim);
  auto group_ptr = allreduce_prim->GetAttr(kAttrNameGroup);
  auto group_name = GetValue<std::string>(group_ptr);
  auto collective_mgr = distributed::collective::CollectiveManager::instance();
  auto rank_size = collective_mgr->GetGroupSize(group_name);
  auto rank_list = collective_mgr->GetGroupRanks(group_name);
  auto is_single_node =
    device::ascend::MultiAscendCollectiveCommLib::GetInstance().isGroupWithinLocalMachine(rank_list);
  auto it = std::find(supported_rank_size.begin(), supported_rank_size.end(), rank_size);
  if (it == supported_rank_size.end() || !is_single_node) {
    MS_LOG(INFO) << "only support rank size 1, 2, 4, 8 in single node.";
    return nullptr;
  }
  auto reduce_op_ptr = allreduce_prim->GetAttr(kAttrNameOp);
  auto reduction = GetValue<std::string>(reduce_op_ptr);
  auto iter = std::find(support_reduce_op_list_.begin(), support_reduce_op_list_.end(), reduction);
  if (iter == support_reduce_op_list_.end()) {
    MS_LOG(INFO) << "reduce operation is not supported.";
    return nullptr;
  }

  auto group = CreateValueNodeWithKernelInfo(func_graph, group_ptr);
  auto reduce_op = CreateValueNodeWithKernelInfo(func_graph, MakeValue<int64_t>(0));
  auto comm_turn = CreateValueNodeWithKernelInfo(func_graph, MakeValue<int64_t>(0));
  auto stream_mode = CreateValueNodeWithKernelInfo(func_graph, MakeValue<int64_t>(1));
  MS_EXCEPTION_IF_NULL(group);
  MS_EXCEPTION_IF_NULL(reduce_op);
  MS_EXCEPTION_IF_NULL(comm_turn);
  MS_EXCEPTION_IF_NULL(stream_mode);

  auto matmul_cnode = matmul_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(matmul_cnode);
  auto transpose_b = matmul_cnode->input(kIndex4)->cast<ValueNodePtr>();
  auto is_transpose_b = GetValue<bool>(transpose_b->value());

  AnfNodePtr transposed_x2 = x2;
  if (is_transpose_b) {
    transposed_x2 = NewTransposeNode(func_graph, x2, node, add_result_type);
  }

  auto prim = std::make_shared<Primitive>(ops::kNameMatmulAllReduceAddRmsNorm);
  std::vector<AnfNodePtr> inputs;
  inputs = {NewValueNode(prim), x1,        transposed_x2, bias, residual, gamma, eps, group,
            reduce_op,          comm_turn, stream_mode};
  auto matmul_allreduce_addrmsnorm_cnode = NewCNode(inputs, func_graph);
  MS_EXCEPTION_IF_NULL(matmul_allreduce_addrmsnorm_cnode);
  MS_LOG(INFO) << "create MatMulAllReduceAddRmsNorm node success.";

  return matmul_allreduce_addrmsnorm_cnode;
}

const AnfNodePtr MatMulAllReduceAddRmsNormFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                          const EquivPtr &equiv) const {
  MS_LOG(INFO) << "start to process MatMulAllReduceAddRmsNormFusion.";
  if (!IsSupport(node, graph)) {
    return nullptr;
  }

  auto add = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), kIndex0);
  MS_EXCEPTION_IF_NULL(add);
  auto rms_norm = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(rms_norm);

  std::vector<TypeId> add_result_types;
  std::vector<BaseShapePtr> add_result_shapes;
  add_result_types.push_back(common::AnfAlgo::GetOutputInferDataType(add, kIndex0));
  add_result_shapes.push_back(AnfAlgo::GetOutputDetailShape(add, kIndex0));

  auto matmul_allreduce_addrmsnorm_cnode = CreateMatMulAllReduceAddRmsNormNode(graph, node, equiv, add_result_types[0]);
  MS_CHECK_TRUE_RET(matmul_allreduce_addrmsnorm_cnode != nullptr, nullptr);

  std::vector<TypeId> types;
  std::vector<BaseShapePtr> shapes;
  types.push_back(add_result_types[0]);
  shapes.push_back(add_result_shapes[0]);
  size_t output_num = AnfAlgo::GetOutputElementNum(node);
  for (size_t i = 0; i < output_num - 1; i++) {
    types.push_back(add_result_types[0]);
    shapes.push_back(add_result_shapes[0]);
  }
  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, matmul_allreduce_addrmsnorm_cnode.get());
  matmul_allreduce_addrmsnorm_cnode->set_scope(node->scope());
  auto build_info = GenerateKernelBuildInfo(matmul_allreduce_addrmsnorm_cnode);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, matmul_allreduce_addrmsnorm_cnode.get());

  auto prim_getitem1 = std::make_shared<Primitive>(kTupleGetItemOpName);
  std::vector<AnfNodePtr> add_result_inputs = {NewValueNode(prim_getitem1), matmul_allreduce_addrmsnorm_cnode,
                                               NewValueNode(static_cast<int64_t>(0))};
  auto add_result = NewCNode(add_result_inputs, graph);
  common::AnfAlgo::SetOutputTypeAndDetailShape(add_result_types, add_result_shapes, add_result.get());
  add_result->set_scope(add->scope());
  build_info = GenerateKernelBuildInfo(add_result);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, add_result.get());

  auto prim_getitem2 = std::make_shared<Primitive>(kTupleGetItemOpName);
  std::vector<AnfNodePtr> rms_norm_result_inputs = {NewValueNode(prim_getitem2), matmul_allreduce_addrmsnorm_cnode,
                                                    NewValueNode(static_cast<int64_t>(1))};
  auto rms_norm_result = NewCNode(rms_norm_result_inputs, graph);
  common::AnfAlgo::SetOutputTypeAndDetailShape(add_result_types, add_result_shapes, rms_norm_result.get());
  rms_norm_result->set_scope(rms_norm->scope());
  build_info = GenerateKernelBuildInfo(rms_norm_result);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, rms_norm_result.get());

  FuncGraphManagerPtr manager = graph->manager();
  MS_EXCEPTION_IF_NULL(matmul_allreduce_addrmsnorm_cnode);

  // replace
  (void)manager->Replace(add, add_result);
  (void)manager->Replace(rms_norm, rms_norm_result);
  MS_LOG(INFO) << "process MatMulAllReduceAddRmsNormFusion success.";
  return matmul_allreduce_addrmsnorm_cnode;
}
}  // namespace opt
}  // namespace mindspore
