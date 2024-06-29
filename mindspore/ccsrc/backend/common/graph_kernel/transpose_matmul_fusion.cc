/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "backend/common/graph_kernel/transpose_matmul_fusion.h"

#include <algorithm>
#include <vector>

#include "ir/graph_utils.h"
#include "mindspore/core/ops/math_ops.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "backend/common/graph_kernel/graph_kernel_helper.h"
#include "backend/common/graph_kernel/core/graph_kernel_callback.h"
namespace mindspore::graphkernel {
bool IsMatMul(const AnfNodePtr &node) {
  return IsPrimitiveCNode(node, prim::kPrimMatMul) || IsPrimitiveCNode(node, prim::kPrimBatchMatMul);
}

bool IsTargetTranspose(const AnfNodePtr &node, const AnfNodePtr &input) {
  auto transpose = input->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(transpose);
  auto perm_node = transpose->input(kIndex2);
  MS_EXCEPTION_IF_NULL(perm_node);
  if (!perm_node->isa<ValueNode>()) {
    return false;
  }
  auto perm_valuenode = perm_node->cast<ValueNodePtr>();
  auto perm = GetValue<std::vector<int64_t>>(perm_valuenode->value());
  (void)std::transform(perm.begin(), perm.end(), perm.begin(),
                       [&perm](int64_t axis) -> int64_t { return axis < 0 ? axis + SizeToLong(perm.size()) : axis; });
  // the target transpose only changes the last two axes.
  for (size_t i = 0; i < perm.size() - kSizeTwo; i++) {
    if (perm[i] != SizeToLong(i)) {
      return false;
    }
  }
  return perm[perm.size() - 2] == SizeToLong(perm.size() - 1) && perm[perm.size() - 1] == SizeToLong(perm.size() - 2);
}

bool TransposeMatmulFusion::Run(const FuncGraphPtr &func_graph) {
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  auto cb = Callback::Instance();
  MS_EXCEPTION_IF_NULL(cb);
  auto nodes = TopoSort(func_graph->get_return());
  for (const auto &node : nodes) {
    if (!IsMatMul(node)) {
      continue;
    }
    if (cb->IsUseDeviceInfo() && cb->GetOutputFormat(node, 0) != kOpFormat_DEFAULT) {
      continue;
    }
    auto matmul = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(matmul);
    auto lhs = matmul->input(kIndex1);
    auto rhs = matmul->input(kIndex2);
    bool trans_a = IsPrimitiveCNode(lhs, prim::kPrimTranspose) && IsTargetTranspose(node, lhs);
    bool trans_b = IsPrimitiveCNode(rhs, prim::kPrimTranspose) && IsTargetTranspose(node, rhs);
    auto prim = GetCNodePrimitive(matmul);
    AnfNodePtrList inputs{NewValueNode(prim)};
    if (trans_a) {
      auto lhs_cnode = lhs->cast<CNodePtr>();
      inputs.emplace_back(lhs_cnode->input(kIndex1));
    } else {
      inputs.emplace_back(lhs);
    }
    if (trans_b) {
      auto rhs_cnode = rhs->cast<CNodePtr>();
      inputs.emplace_back(rhs_cnode->input(kIndex1));
    } else {
      inputs.emplace_back(rhs);
    }
    if (!trans_a && !trans_b) {
      continue;
    }
    auto input_trans_a_node = matmul->input(kIndex3);
    auto input_trans_b_node = matmul->input(kIndex4);

    if (!input_trans_a_node->isa<ValueNode>() || !input_trans_b_node->isa<ValueNode>()) {
      continue;
    }
    // update transpose inputs of matmul
    auto input_trans_a = GetValue<bool>(input_trans_a_node->cast<ValueNodePtr>()->value());
    auto input_trans_b = GetValue<bool>(input_trans_b_node->cast<ValueNodePtr>()->value());
    auto new_trans_a = MakeValue<bool>(trans_a ^ input_trans_a);
    auto new_trans_b = MakeValue<bool>(trans_b ^ input_trans_b);
    auto new_trans_a_node = NewValueNode(new_trans_a);
    auto new_trans_b_node = NewValueNode(new_trans_b);
    new_trans_a_node->set_abstract(new_trans_a->ToAbstract());
    new_trans_b_node->set_abstract(new_trans_b->ToAbstract());
    inputs.emplace_back(new_trans_a_node);
    inputs.emplace_back(new_trans_b_node);
    auto new_matmul = func_graph->NewCNode(inputs);
    func_graph->AddValueNode(new_trans_a_node);
    func_graph->AddValueNode(new_trans_b_node);
    // clone cnode attrs
    new_matmul->set_attrs(matmul->attrs());
    new_matmul->set_abstract(matmul->abstract());
    if (cb->IsUseDeviceInfo()) {
      auto build_info = AnfAlgo::GetSelectKernelBuildInfo(node);
      AnfAlgo::SetSelectKernelBuildInfo(build_info, new_matmul.get());
    }
    (void)mng->Replace(node, new_matmul);
  }
  return true;
}
}  // namespace mindspore::graphkernel
