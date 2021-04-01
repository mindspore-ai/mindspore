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
#include "backend/optimizer/graph_kernel/optimize_matmul.h"
#include <tuple>
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/kernel_compiler/common_utils.h"
#include "backend/optimizer/graph_kernel/graph_kernel_helper.h"

namespace mindspore {
namespace opt {
/* MatMul supports fp32 bias, so remove the redundant cast when cast only used by MatMul
 *
 *   %0 = cast(bias_fp32, fp16)
 *   %1 = MatMul(A_fp16, B_fp16, %0)
 *   ------>
 *   %1 = MatMul(A_fp16, B_fp16, bias_fp32)
 */
bool OptimizeMatmul::Run(const FuncGraphPtr &func_graph) {
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
    if (cast_input_type == kNumberTypeFloat32 && cast_output_type == kNumberTypeFloat16 &&
        mng->node_users()[cast_node].size() == 1) {
      mng->Replace(cast_node, (cast_node->cast<CNodePtr>())->input(1));
      changed = true;
    }
  }

  return changed;
}
}  // namespace opt
}  // namespace mindspore
