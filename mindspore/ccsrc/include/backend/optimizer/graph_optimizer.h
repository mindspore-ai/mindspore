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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_OPTIMIZER_GRAPH_OPTIMIZER_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_OPTIMIZER_GRAPH_OPTIMIZER_H_

#include <string>
#include <vector>
#include "include/backend/optimizer/pass_manager.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace opt {
class BACKEND_EXPORT GraphOptimizer {
 public:
  explicit GraphOptimizer(const std::string &name = "graph_optimizer") : name_(name) {}
  virtual ~GraphOptimizer() = default;

  void AddPassManager(const PassManagerPtr &pass_manager);
  FuncGraphPtr Optimize(const FuncGraphPtr &func_graph, bool run_only_once = true);

 private:
  const std::string name_ = "graph_optimizer";
  std::vector<PassManagerPtr> pass_managers_{};
  bool run_only_once_ = true;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_OPTIMIZER_GRAPH_OPTIMIZER_H_
