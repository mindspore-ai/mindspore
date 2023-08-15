/**
 * Copyright 2023-2023 Huawei Technologies Co., Ltd
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
#include "backend/common/pass/merge_transdata.h"

#include <map>
#include <memory>
#include <tuple>
#include <utility>

#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/parallel_context.h"
#include "include/common/utils/utils.h"
#include "ir/graph_utils.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/framework_ops.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kIndexOne = 1;
}

bool MergeTransData::Run(const FuncGraphPtr &func_graph) {
  // Only pipeline parallel need run this pass
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto stages = parallel_context->pipeline_stage_split_num();
  auto grad_accu_num = parallel_context->grad_accumulation_step();
  if (stages <= 1 && grad_accu_num <= 1) {
    return false;
  }

  std::map<std::tuple<AnfNodePtr, std::string, ShapeVector>, std::vector<CNodePtr>> transdata_map;
  MS_EXCEPTION_IF_NULL(func_graph);
  const std::vector<AnfNodePtr> &node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (IsOneOfPrimitiveCNode(node, {prim::kPrimTransData})) {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      const auto &out_format = AnfAlgo::GetOutputFormat(cnode, 0);
      const auto &out_shape = AnfAlgo::GetOutputDeviceShape(cnode, 0);
      auto prenode = common::AnfAlgo::VisitKernelWithReturnType(cnode->input(kIndexOne), 0, true);
      transdata_map[{prenode.first, out_format, out_shape}].push_back(cnode);
    }
  }

  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  for (auto &kv : transdata_map) {
    if (kv.second.size() <= kIndexOne) {
      continue;
    }
    for (size_t i = kIndexOne; i < kv.second.size(); i++) {
      if (IsPrimitiveCNode(kv.second[i]->input(kIndexOne), prim::kPrimDepend)) {
        auto depend_node = kv.second[i]->input(kIndexOne)->cast<CNodePtr>();
        auto new_depend_node =
          func_graph->NewCNode({NewValueNode(prim::kPrimDepend), kv.second[kIndexZero], depend_node->input(kIndexTwo)});
        (void)manager->Replace(kv.second[i], new_depend_node);
      } else {
        (void)manager->Replace(kv.second[i], kv.second[kIndexZero]);
      }
    }
  }

  return true;
}
}  // namespace opt
}  // namespace mindspore
