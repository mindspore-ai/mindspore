/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "backend/common/pass/erase_invalid_micro_depend.h"

#include <map>
#include <memory>
#include <vector>
#include <algorithm>
#include "include/common/utils/parallel_context.h"
#include "ops/sequence_ops.h"
#include "ops/other_ops.h"
#include "ops/framework_ops.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace opt {
namespace {
constexpr char MICRO[] = "micro";

bool NeedProcess() {
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto stages = parallel_context->pipeline_stage_split_num();
  if (stages <= 1) {
    return false;
  }
  return true;
}

bool EraseMicroDepend(const FuncGraphPtr &graph) {
  if (!NeedProcess()) {
    return false;
  }
  AnfNodePtr return_node = graph->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  std::vector<AnfNodePtr> all_nodes = TopoSort(return_node);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!IsPrimitiveCNode(node, prim::kPrimDepend)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    MS_EXCEPTION_IF_NULL(cnode->input(kIndex1));
    MS_EXCEPTION_IF_NULL(cnode->input(kIndex2));
    if (!cnode->input(kIndex1)->isa<ValueNode>()) {
      continue;
    }
    if (!IsPrimitiveCNode(cnode->input(kIndex2), prim::kPrimMakeTuple)) {
      continue;
    }
    auto make_tuple = cnode->input(kIndex2)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(make_tuple);
    bool not_has_micro = false;
    std::vector<CNodePtr> tuple_inputs;
    for (size_t i = 1; i < make_tuple->inputs().size(); i++) {
      auto input = make_tuple->input(i);
      MS_EXCEPTION_IF_NULL(input);
      if (!input->isa<CNode>()) {
        not_has_micro = true;
        break;
      }
      auto cnode_input = input->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode_input);
      if (!cnode_input->HasPrimalAttr(MICRO)) {
        not_has_micro = true;
        break;
      }
      tuple_inputs.emplace_back(cnode_input);
    }
    if (not_has_micro) {
      continue;
    }

    sort(tuple_inputs.begin(), tuple_inputs.end(),
         [](const CNodePtr &a, const CNodePtr &b) { return a->GetPrimalAttr(MICRO) > b->GetPrimalAttr(MICRO); });

    manager->SetEdge(cnode, kIndex2, tuple_inputs[0]);
  }
  return true;
}
}  // namespace

bool EraseInvalidMicroDepend::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  return EraseMicroDepend(func_graph);
}
}  // namespace opt
}  // namespace mindspore
