/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "backend/optimizer/graph_kernel/cast_matmul_fusion.h"
#include <tuple>
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/kernel_compiler/common_utils.h"
#include "backend/optimizer/graph_kernel/graph_kernel_helper.h"

namespace mindspore {
namespace opt {
namespace {
// Check if leaf is used by root
bool HasPath(const AnfNodePtr &leaf, const AnfNodePtr &root, const FuncGraphManagerPtr &mng) {
  MS_EXCEPTION_IF_NULL(mng);
  bool result = false;
  auto IncludeUser = [&result, &root](const AnfNodePtr &node) {
    if (node == root) {
      result = true;
    }
    return result ? EXCLUDE : FOLLOW;
  };
  static_cast<void>(DeepLinkedGraphSearch(leaf, IncludeUser));
  return result;
}

// Update matmul's BuildInfo as last input changed
void UpdateBuildInfo(const AnfNodePtr &matmul_node, const AnfNodePtr &cast_node) {
  std::vector<std::string> input_formats = AnfAlgo::GetAllInputFormats(matmul_node);
  std::vector<TypeId> input_types = AnfAlgo::GetAllInputDeviceTypes(matmul_node);
  input_types.pop_back();
  auto cast_types = AnfAlgo::GetAllInputDeviceTypes(cast_node);
  input_types.push_back(cast_types.front());
  std::vector<std::string> output_formats = AnfAlgo::GetAllOutputFormats(matmul_node);
  std::vector<TypeId> output_types = AnfAlgo::GetAllOutputDeviceTypes(matmul_node);
  auto graph_sel_info =
    BuildSelectKernelBuildInfo(input_formats, input_types, output_formats, output_types, matmul_node);
  AnfAlgo::SetSelectKernelBuildInfo(graph_sel_info, matmul_node.get());
}
}  // namespace

/* MatMul supports fp32 bias, so remove the redundant cast if cast cannot fuse forword
 *  case1, cast only used by MatMul
 *
 *   bias_fp32 = depend(bias_fp32, u)
 *   %0 = cast(bias_fp32, fp16)
 *   %1 = MatMul(A_fp16, B_fp16, %0)
 *   ------>
 *   bias_fp32 = depend(bias_fp32, u)
 *   %1 = MatMul(A_fp16, B_fp16, bias_fp32)
 *
 *  case2, cast used by MatMul and UpdateStatus
 *
 *   bias_fp32 = load(p, status)
 *   %0 = cast(bias_fp32, fp16)
 *   %1 = MatMul(A_fp16, B_fp16, %0)
 *   %2 = UpstateStatus(status, %0)
 *   ------>
 *   bias_fp32 = load(p, status)
 *   %1 = MatMul(A_fp16, B_fp16, bias_fp32)
 *   %2 = UpstateStatus(status, %1)
 */
bool CastMatmulFusion::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }
  auto changed = false;
  auto nodes = TopoSort(func_graph->get_return());
  for (auto node : nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimMatMul)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (cnode->size() != 4) {
      continue;
    }
    auto cast_node = cnode->input(3);
    if (!IsPrimitiveCNode(cast_node, prim::kPrimCast)) {
      continue;
    }
    auto cast_input_type = AnfAlgo::GetInputDeviceDataType(cast_node, 0);
    auto cast_output_type = AnfAlgo::GetOutputDeviceDataType(cast_node, 0);
    if (cast_input_type != kNumberTypeFloat32 || cast_output_type != kNumberTypeFloat16) {
      continue;
    }
    // Cast cannot fuse with its input
    if (IsFusibleOp((cast_node->cast<CNodePtr>())->input(1))) {
      continue;
    }

    auto user_index_set = mng->node_users()[cast_node];
    // Case1 : Cast is only used by matmul
    if (user_index_set.size() == 1) {
      mng->Replace(cast_node, (cast_node->cast<CNodePtr>())->input(1));
      UpdateBuildInfo(cnode, cast_node);
      changed = true;
      continue;
    }

    // Case2 : Cast is used by matmul and Upstatus
    if (user_index_set.size() > 2) {
      continue;
    }
    for (auto user_index : user_index_set) {
      // Exclude when UpdateStatus-> ... ->matmul path is found
      if (IsPrimitiveCNode(user_index.first, prim::kPrimUpdateState) && !HasPath(user_index.first, node, mng)) {
        auto update_state = (user_index.first)->cast<CNodePtr>();
        update_state->set_input(2, node);
        cnode->set_input(4, (cast_node->cast<CNodePtr>())->input(1));
        mng->RemoveRoots();
        mng->KeepRoots({func_graph});
        UpdateBuildInfo(cnode, cast_node);
        changed = true;
      }
    }
  }

  return changed;
}
}  // namespace opt
}  // namespace mindspore
