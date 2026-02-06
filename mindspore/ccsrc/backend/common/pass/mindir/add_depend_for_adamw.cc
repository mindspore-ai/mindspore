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

#include "backend/common/pass/mindir/add_depend_for_adamw.h"
#include "ir/anf.h"
#include "ir/graph_utils.h"
#include "primitive/nn_optimizer_ops.h"
#include "primitive/math_ops.h"
#include "primitive/framework_ops.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "utils/log_adapter.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"

namespace mindspore {
namespace opt {
namespace {
const char kAdamWStr[] = "AdamW";
}
bool AddDependForAdamW::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto nodes = TopoSort(graph->get_return(), SuccDeeperSimple, AlwaysInclude);
  AnfNodePtr last_addcmul = nullptr;
  bool changed = false;
  for (const auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (IsPrimitiveCNode(node, prim::kPrimAddcmul)) {
      auto full_name = node->fullname_with_scope();
      if (full_name.find(kAdamWStr) == std::string::npos) {
        continue;
      }
      last_addcmul = node;
    }
    if (IsPrimitiveCNode(node, prim::kPrimCast)) {
      auto full_name = node->fullname_with_scope();
      if (last_addcmul != nullptr && full_name.find(kAdamWStr) != std::string::npos) {
        MS_LOG(INFO) << "Add Depend for AdamW Cast op: " << full_name
                     << ", AdamW addcmul op: " << last_addcmul->fullname_with_scope();
        changed = true;
        auto cast_node = node->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(cast_node);
        auto cast_input = cast_node->input(1);
        MS_EXCEPTION_IF_NULL(cast_input);
        MS_EXCEPTION_IF_NULL(cast_node->func_graph());
        auto depend = cast_node->func_graph()->NewCNode({NewValueNode(prim::kPrimDepend), cast_input, last_addcmul});
        MS_EXCEPTION_IF_NULL(depend);
        depend->set_abstract(cast_input->abstract());
        cast_node->set_input(1, depend);
        manager->SetEdge(cast_node, 1, depend);
      }
    }
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
