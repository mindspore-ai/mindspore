/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/transpose_ext_matmul_ext_transpose_fusion.h"

#include <algorithm>
#include <iterator>
#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "backend/common/pass/common/gllo_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_info.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "include/utils/utils.h"
#include "primitive/math_ops.h"
#include "primitive/nn_ops.h"
#include "utils/ms_context.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace opt {

namespace {
constexpr auto kTransposeBatchMatmulTransposeOpName = "TransposeBatchMatmulTranspose";

ValueNodePtr CreateTupleConstNode(const FuncGraphPtr &func_graph, const std::vector<int64_t> &vals) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto vnode = kernel_graph->NewValueNode(MakeValue<std::vector<int64_t>>(vals));
  kernel_graph->AddValueNodeToGraph(vnode);
  return vnode;
}

ValueNodePtr CreateBoolConstNode(const FuncGraphPtr &func_graph, bool value) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto vnode = kernel_graph->NewValueNode(MakeValue<bool>(value));
  kernel_graph->AddValueNodeToGraph(vnode);
  return vnode;
}

ShapeVector GetPermFromDims(const AnfNodePtr &transpose_node, const ShapeVector &input_shape) {
  MS_EXCEPTION_IF_NULL(transpose_node);
  auto cnode = transpose_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->size() < kIndex4) {
    MS_LOG(INFO) << "TransposeExtMatmulExtTranspose failed because cnode->size() < kIndex4.";
    return {};
  }
  auto dim0_value_node = cnode->input(kIndex2)->cast<ValueNodePtr>();
  auto dim1_value_node = cnode->input(kIndex3)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(dim0_value_node);
  MS_EXCEPTION_IF_NULL(dim1_value_node);
  int64_t dim0 = GetValue<int64_t>(dim0_value_node->value());
  int64_t dim1 = GetValue<int64_t>(dim1_value_node->value());
  auto rank = SizeToLong(input_shape.size());
  if (dim0 < 0) dim0 += rank;
  if (dim1 < 0) dim1 += rank;
  if (dim0 < 0 || dim1 < 0 || dim0 >= rank || dim1 >= rank) {
    MS_LOG(INFO) << "TransposeExtMatmulExtTranspose failed because dim0(" << dim0 << ") or dim1(" << dim1
                 << ") is invalid.";
    return {};
  }
  ShapeVector perm(rank);
  std::iota(perm.begin(), perm.end(), 0);
  std::swap(perm[LongToSize(dim0)], perm[LongToSize(dim1)]);
  return perm;
}

// Return true if perm swaps the last two dimensions (e.g., [..., N, M] -> [..., M, N]).
bool IsSwapLastTwoPerm(const ShapeVector &perm) {
  constexpr size_t kLastTwo = 2;
  if (perm.size() < kLastTwo) {
    return false;
  }
  ShapeVector expect(perm.size());
  std::iota(expect.begin(), expect.end(), 0);
  auto last_idx = expect.size() - 1;
  auto second_last_idx = expect.size() - kLastTwo;
  std::swap(expect[second_last_idx], expect[last_idx]);
  return perm == expect;
}

// If node is a TransposeExtView or Load(TransposeExtView, U), return the inner TransposeExtView.
CNodePtr GetTransposeFromNode(const AnfNodePtr &node) {
  if (node == nullptr || !node->isa<CNode>()) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  if (IsPrimitiveCNode(cnode, prim::kPrimTransposeExtView)) {
    return cnode;
  }
  if (IsPrimitiveCNode(cnode, prim::kPrimLoad)) {
    auto inner = cnode->input(kIndex1);
    if (inner != nullptr && inner->isa<CNode>() &&
        IsPrimitiveCNode(inner->cast<CNodePtr>(), prim::kPrimTransposeExtView)) {
      return inner->cast<CNodePtr>();
    }
  }
  return nullptr;
}

// Normalize weight input: if it is a transpose on last two dims, return the original contiguous weight
// and set transpose_b_flag to true. Keep Load monad if needed.
std::pair<AnfNodePtr, bool> BuildWeightInputAndTransB(const FuncGraphPtr &func_graph, const AnfNodePtr &weight_node) {
  MS_EXCEPTION_IF_NULL(func_graph);
  if (weight_node == nullptr) {
    return {weight_node, false};
  }
  auto w_trans = GetTransposeFromNode(weight_node);
  if (w_trans == nullptr) {
    return {weight_node, false};
  }
  auto w_input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(w_trans, kIndex0);
  auto w_perm = GetPermFromDims(w_trans, w_input_shape);
  if (!IsSwapLastTwoPerm(w_perm)) {
    return {weight_node, false};
  }
  auto orig_w = w_trans->input(kIndex1);
  if (IsPrimitiveCNode(weight_node, prim::kPrimLoad)) {
    auto w_load = weight_node->cast<CNodePtr>();
    auto new_load = NewCNode({NewValueNode(prim::kPrimLoad), orig_w, w_load->input(kIndex2)}, func_graph);
    new_load->set_scope(w_load->scope());
    if (weight_node->abstract() != nullptr) {
      new_load->set_abstract(weight_node->abstract()->Clone());
    }
    return {new_load, true};
  }
  return {orig_w, true};
}
}  // namespace

std::vector<std::string> TransposeExtMatmulExtTranspose::MustExistPrimitiveName() const {
  // Match TransposeExtView + Load + MatMulExt + TransposeExtView
  std::vector<std::string> ret{prim::kPrimTransposeExtView->name(), prim::kPrimMatMulExt->name()};
  return ret;
}

const BaseRef TransposeExtMatmulExtTranspose::DefinePattern() const {
  // IR pattern:
  // %0 = TransposeExtView(x, dim0_in, dim1_in, U)
  // %1 = UpdateState(u1_input0, u1_input1)
  // %2 = Load(%0, %1)
  // %3 = MatMulExt(%2, y)
  // %4 = UpdateState(u2_input0, u2_input1)
  // %5 = TransposeExtView(%3, dim0_out, dim1_out, %4)
  auto x = std::make_shared<Var>();
  auto y = std::make_shared<Var>();
  auto u0 = std::make_shared<Var>();
  auto dim0_in = std::make_shared<Var>();
  auto dim1_in = std::make_shared<Var>();
  auto dim0_out = std::make_shared<Var>();
  auto dim1_out = std::make_shared<Var>();

  auto u1_input0 = std::make_shared<Var>();
  auto u1_input1 = std::make_shared<Var>();
  auto u2_input0 = std::make_shared<Var>();
  auto u2_input1 = std::make_shared<Var>();

  auto transpose_in =
    VectorRef({std::make_shared<Primitive>(prim::kPrimTransposeExtView->name()), x, dim0_in, dim1_in, u0});
  auto u1 = VectorRef({prim::kPrimUpdateState, u1_input0, u1_input1});
  auto load = VectorRef({prim::kPrimLoad, transpose_in, u1});
  auto matmul = VectorRef({prim::kPrimMatMulExt, load, y});
  auto u2 = VectorRef({prim::kPrimUpdateState, u2_input0, u2_input1});
  VectorRef pattern({std::make_shared<Primitive>(prim::kPrimTransposeExtView->name()), matmul, dim0_out, dim1_out, u2});
  return pattern;
}

const AnfNodePtr TransposeExtMatmulExtTranspose::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                         const EquivPtr &equiv) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->IsEnableInferBoost()) {
    MS_LOG(INFO) << "TransposeExtMatmulExtTranspose failed because infer boost is off.";
    return nullptr;
  }
  auto const &soc_version = ms_context->ascend_soc_version();
  const std::vector<std::string> valid_soc_version{"ascend910b", "ascend910_93", "ascend310p"};
  if (!soc_version.empty() &&
      (std::find(valid_soc_version.begin(), valid_soc_version.end(), soc_version) == valid_soc_version.end())) {
    MS_LOG(INFO) << "TransposeExtMatmulExtTranspose failed because soc is not support: " << soc_version;
    return nullptr;
  }

  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(equiv);
  // ... -> TransposeExtViewIn -> UpdateState -> Load -> MatMulExt -> UpdateState -> TransposeExtViewOut
  auto transpose_out = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(transpose_out != nullptr, {});
  auto mm_cnode = transpose_out->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(mm_cnode != nullptr, {});
  MS_CHECK_TRUE_RET(mm_cnode->func_graph() == transpose_out->func_graph(), {});

  // First input of MatMulExt is Load
  auto load_cnode = mm_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(load_cnode != nullptr, {});
  if (!IsPrimitiveCNode(load_cnode, prim::kPrimLoad)) {
    return nullptr;
  }
  // Load's first input is the inner TransposeExtView
  auto transpose_in = load_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(transpose_in != nullptr, {});

  // Get dims for input and output transpose ext
  auto input_x_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(transpose_in, kIndex0);
  auto perm_in_value = GetPermFromDims(transpose_in, input_x_shape);
  if (perm_in_value.empty()) {
    MS_LOG(INFO) << "TransposeExtMatmulExtTranspose failed because perm_in is invalid.";
    return nullptr;
  }
  auto mm_shape = common::AnfAlgo::GetOutputInferShape(mm_cnode, 0);
  auto perm_out_value = GetPermFromDims(transpose_out, mm_shape);
  if (perm_out_value.empty()) {
    MS_LOG(INFO) << "TransposeExtMatmulExtTranspose failed because perm_out is invalid.";
    return nullptr;
  }
  if (perm_in_value != perm_out_value) {
    MS_LOG(INFO) << "TransposeExtMatmulExtTranspose failed because perm_in(" << perm_in_value << ") != perm_out("
                 << perm_out_value << ").";
    return nullptr;
  }

  // Check supported perm and ranks: support (1,0,2) for 3D or (0,2,1,3) for 4D like original pass
  static const ShapeVector perm_3d = {1, 0, 2};
  static const ShapeVector perm_4d = {0, 2, 1, 3};
  if (!((SizeToLong(input_x_shape.size()) == SizeToLong(perm_3d.size()) && perm_in_value == perm_3d) ||
        (SizeToLong(input_x_shape.size()) == SizeToLong(perm_4d.size()) && perm_in_value == perm_4d))) {
    MS_LOG(INFO) << "TransposeExtMatmulExtTranspose failed because unsupported perm. input_shape: " << input_x_shape
                 << ", perm_in_value: " << perm_in_value;
    return nullptr;
  }

  // Build weight input and transpose_b flag (handle mint.transpose on weight to ensure contiguity)
  auto weight_and_flag = BuildWeightInputAndTransB(func_graph, mm_cnode->input(kIndex2));
  auto fused_weight_input = weight_and_flag.first;
  bool transpose_b_flag = weight_and_flag.second;

  MS_LOG(INFO) << "TransposeExtMatmulExtTranspose start create fused node.";
  // Create fused node and its inputs
  PrimitivePtr fused_prim = std::make_shared<Primitive>(kTransposeBatchMatmulTransposeOpName);
  MS_CHECK_TRUE_RET(fused_prim, {});
  auto perm_in_node = CreateTupleConstNode(func_graph, perm_in_value);
  auto perm_out_node = CreateTupleConstNode(func_graph, perm_out_value);
  auto trans_a_node = CreateBoolConstNode(func_graph, false);
  auto trans_b_node = CreateBoolConstNode(func_graph, transpose_b_flag);
  auto fused_inputs = std::vector<AnfNodePtr>{NewValueNode(fused_prim),
                                              transpose_in->input(kIndex1),
                                              fused_weight_input,
                                              perm_in_node,
                                              perm_out_node,
                                              trans_a_node,
                                              trans_b_node};
  CNodePtr fusion_cnode = NewCNode(fused_inputs, func_graph);

  fusion_cnode->set_scope(transpose_out->scope());
  // Ensure kernel_info object exists
  fusion_cnode->set_kernel_info(std::make_shared<device::KernelInfo>());
  if (node->abstract() != nullptr) {
    fusion_cnode->set_abstract(transpose_out->abstract()->Clone());
  }

  // Clean state chain to avoid leftover TransposeExtView/Load nodes.
  // Replace u1 (second input of Load) and u2 (fourth input of output transpose) with original U (u0).
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  auto u0 = transpose_in->input(kIndex4);
  auto u1 = load_cnode->input(kIndex2);
  auto u2 = transpose_out->input(kIndex4);
  (void)mng->Replace(u1, u0);
  (void)mng->Replace(u2, u0);

  // Ensure no remaining reference to the input TransposeExtView by replacing it with its original tensor.
  (void)mng->Replace(transpose_in, transpose_in->input(kIndex1));

  // Attach kernel build info for fused node
  auto build_info = GenerateKernelBuildInfo(fusion_cnode);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, fusion_cnode.get());
  MS_LOG(INFO) << "TransposeExtMatmulExtTranspose end create fused node.";

  return fusion_cnode;
}
}  // namespace opt
}  // namespace mindspore
