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
#include "backend/common/graph_kernel/cast_matmul_fusion.h"

#include <vector>
#include <string>
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/common_utils.h"
#include "backend/common/graph_kernel/graph_kernel_helper.h"

namespace mindspore::graphkernel {
namespace {
// Update matmul's BuildInfo as last input changed
void UpdateBuildInfo(const AnfNodePtr &matmul_node, const AnfNodePtr &cast_node) {
  std::vector<std::string> input_formats = AnfAlgo::GetAllInputFormats(matmul_node);
  std::vector<TypeId> input_types = AnfAlgo::GetAllInputDeviceTypes(matmul_node);
  input_types.pop_back();
  auto cast_types = AnfAlgo::GetAllInputDeviceTypes(cast_node);
  input_types.push_back(cast_types.front());
  std::vector<std::string> output_formats = AnfAlgo::GetAllOutputFormats(matmul_node);
  std::vector<TypeId> output_types = AnfAlgo::GetAllOutputDeviceTypes(matmul_node);
  auto graph_sel_info = BuildSelectKernelBuildInfo(input_formats, input_types, output_formats, output_types,
                                                   AnfAlgo::GetProcessor(matmul_node));
  AnfAlgo::SetSelectKernelBuildInfo(graph_sel_info, matmul_node.get());
}

/* MatMul supports fp32 bias, so remove the redundant cast if cast cannot fuse forword
 * and cast only used by MatMul
 *
 *   %0 = cast(bias_fp32, fp16)
 *   %1 = MatMul(A_fp16, B_fp16, %0)
 *   ------>
 *   %1 = MatMul(A_fp16, B_fp16, bias_fp32)
 */
bool DoFuse(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }
  MS_EXCEPTION_IF_NULL(mng);
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
    auto cast_node = cnode->inputs().back();  // bias node
    if (!IsPrimitiveCNode(cast_node, prim::kPrimCast)) {
      continue;
    }
    auto cast_input_type = AnfAlgo::GetInputDeviceDataType(cast_node, 0);
    auto cast_output_type = AnfAlgo::GetOutputDeviceDataType(cast_node, 0);
    if (cast_input_type != kNumberTypeFloat32 || cast_output_type != kNumberTypeFloat16) {
      continue;
    }
    // Cast cannot fuse with its input
    auto params = func_graph->parameters();
    auto iter = std::find(params.begin(), params.end(), (cast_node->cast<CNodePtr>())->input(1));
    if (iter == params.end()) {
      continue;
    }
    // Cast is only used by matmul
    auto user_index_set = mng->node_users()[cast_node];
    if (user_index_set.size() == 1) {
      (void)mng->Replace(cast_node, (cast_node->cast<CNodePtr>())->input(1));
      UpdateBuildInfo(cnode, cast_node);
      changed = true;
      continue;
    }
  }

  return changed;
}
}  // namespace

bool CastMatmulFusion::Run(const FuncGraphPtr &func_graph) {
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  auto changed = false;
  auto nodes = TopoSort(func_graph->get_return());
  for (auto node : nodes) {
    if (!common::AnfAlgo::IsGraphKernel(node)) {
      continue;
    }
    auto graph_kernel_fg = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
    MS_EXCEPTION_IF_NULL(graph_kernel_fg);
    changed = DoFuse(graph_kernel_fg) || changed;
  }
  return changed;
}
}  // namespace mindspore::graphkernel
