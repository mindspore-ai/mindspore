/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "include/backend/optimizer/graph_optimizer.h"
#include "backend/common/optimizer/cache_manager.h"

namespace mindspore {
namespace opt {
void GraphOptimizer::AddPassManager(const PassManagerPtr &pass_manager) {
  if (pass_manager != nullptr) {
    pass_managers_.push_back(pass_manager);
  }
}

FuncGraphPtr GraphOptimizer::Optimize(const FuncGraphPtr &func_graph, bool run_only_once) {
  run_only_once_ = (pass_managers_.size() == 1) ? true : run_only_once;
  // cppcheck-suppress *
  auto manager = Manage(func_graph, true);

  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i < pass_managers_.size(); ++i) {
      const PassManagerPtr &pm = pass_managers_[i];
      if (pm != nullptr && pm->Run(func_graph)) {
        changed = true;
      }
    }
    if (run_only_once_) {
      break;
    }
  }

  std::vector<FuncGraphPtr> func_graphs;
  func_graphs.push_back(func_graph);
  (void)TopoSort(func_graph->get_return());
  auto func_graph_index = manager->func_graph_index(func_graph);
  MS_EXCEPTION_IF_NULL(func_graph_index);
  func_graph_index->set_has_gen_index(false);

  return func_graph;
}
}  // namespace opt
}  // namespace mindspore
