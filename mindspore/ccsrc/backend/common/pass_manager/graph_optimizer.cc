/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#include "include/backend/common/pass_manager/graph_optimizer.h"
#include "ir/graph_utils.h"

namespace mindspore {
namespace opt {
void GraphOptimizer::AddPassManager(const PassManagerPtr &pass_manager) {
  if (pass_manager != nullptr) {
    pass_managers_.push_back(pass_manager);
  }
}

FuncGraphPtr GraphOptimizer::Optimize(const FuncGraphPtr &func_graph, bool run_only_once) {
  run_only_once_ = (pass_managers_.size() == 1) ? true : run_only_once;
  MS_EXCEPTION_IF_NULL(func_graph);
  // cppcheck-suppress *
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    manager = Manage(func_graph, true);
    MS_EXCEPTION_IF_NULL(manager);
    manager->AddFuncGraph(func_graph);
    func_graph->set_manager(manager);
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i < pass_managers_.size(); ++i) {
      const PassManagerPtr &pm = pass_managers_[i];
      MS_EXCEPTION_IF_NULL(pm);
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
  if (!func_graph->has_user_data<FuncGraphPassIndex>()) {
    func_graph->set_user_data<FuncGraphPassIndex>(std::make_shared<FuncGraphPassIndex>());
  }
  auto func_graph_index = func_graph->user_data<FuncGraphPassIndex>();
  MS_EXCEPTION_IF_NULL(func_graph_index);
  func_graph_index->set_has_gen_index(false);

  return func_graph;
}
}  // namespace opt
}  // namespace mindspore
