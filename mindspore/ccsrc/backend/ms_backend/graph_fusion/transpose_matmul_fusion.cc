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
#include "backend/ms_backend/graph_fusion/transpose_matmul_fusion.h"

#include <algorithm>
#include <vector>
#include <utility>

#include "ir/graph_utils.h"
#include "primitive/math_ops.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "backend/ms_backend/graph_fusion/graph_kernel_helper.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/graph_kernel_callback.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_cluster_cloud.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"

namespace mindspore::graphkernel {
bool IsMatMul(const AnfNodePtr &node) {
  return IsPrimitiveCNode(node, prim::kPrimMatMul) || IsPrimitiveCNode(node, prim::kPrimBatchMatMul) ||
         (IsPrimitiveCNode(node, prim::kPrimGroupedMatmul) &&
          StaticShapeCluster::CanClusterableOp(node, StaticShapeCluster::GetClusterOps()));
}

bool IsTargetTranspose(const AnfNodePtr &input) {
  if (IsPrimitiveCNode(input, prim::kPrimTranspose)) {
    auto transpose = input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(transpose);
    auto perm_node = transpose->input(kIndex2);
    MS_EXCEPTION_IF_NULL(perm_node);
    if (!perm_node->isa<ValueNode>()) {
      return false;
    }
    auto rank = SizeToLong(GetShape(input).size());
    auto perm = GetValue<std::vector<int64_t>>(perm_node->cast<ValueNodePtr>()->value());
    (void)std::transform(perm.begin(), perm.end(), perm.begin(),
                         [rank](int64_t axis) -> int64_t { return axis < 0 ? axis + rank : axis; });
    // the target transpose only changes the last two axes.
    std::swap(perm[perm.size() - kSizeOne], perm[perm.size() - kSizeTwo]);
    for (size_t i = 0; i < perm.size(); i++) {
      if (perm[i] != SizeToLong(i)) {
        return false;
      }
    }
    return true;
  }
  if (IsPrimitiveCNode(input, prim::kPrimTransposeExtView)) {
    const int64_t kDimM = -2;
    const int64_t kDimN = -1;
    auto transpose = input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(transpose);
    auto perm0_node = transpose->input(kIndex2);
    auto perm1_node = transpose->input(kIndex3);
    MS_EXCEPTION_IF_NULL(perm0_node);
    MS_EXCEPTION_IF_NULL(perm1_node);
    if (!perm0_node->isa<ValueNode>() || !perm1_node->isa<ValueNode>()) {
      return false;
    }
    auto rank = SizeToLong(GetShape(input).size());
    auto perm0 = GetValue<int64_t>(perm0_node->cast<ValueNodePtr>()->value());
    auto perm1 = GetValue<int64_t>(perm1_node->cast<ValueNodePtr>()->value());
    if (perm0 < 0) {
      perm0 += rank;
    }
    if (perm1 < 0) {
      perm1 += rank;
    }
    return perm0 == rank + kDimN && perm1 == rank + kDimM;
  }
  return false;
}

bool TransposeMatmulFusion::Run(const FuncGraphPtr &func_graph) {
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  auto cb = Callback::Instance();
  MS_EXCEPTION_IF_NULL(cb);
  bool changed = false;
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

    for (size_t i = kIndex1; i < kIndex3; i++) {
      auto trans_node = matmul->input(i);
      bool trans = IsTargetTranspose(trans_node);
      if (trans) {
        auto prim = GetCNodePrimitive(matmul);
        MS_EXCEPTION_IF_NULL(prim);
        changed = true;
        auto attr_name = i == kIndex1 ? kTransposeA : kTransposeB;
        bool ori_trans = GetValue<bool>(prim->GetAttr(attr_name));
        auto new_prim = prim->Clone();
        new_prim->set_attr(attr_name, MakeValue<bool>(trans ^ ori_trans));
        matmul->set_input(0, NewValueNode(new_prim));
        mng->SetEdge(matmul, i, trans_node->cast<CNodePtr>()->input(kIndex1));
      }
    }
  }
  if (changed) {
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }
  return changed;
}
}  // namespace mindspore::graphkernel
