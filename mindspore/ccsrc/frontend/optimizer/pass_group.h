/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_PASS_GROUP_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_PASS_GROUP_H_

#include <utility>
#include <vector>
#include <string>
#include <memory>

#include "frontend/optimizer/py_pass.h"

namespace mindspore {
namespace opt {
namespace python_pass {
class PassGroup {
 public:
  explicit PassGroup(const std::string &name = "pass_group", bool run_only_once = false)
      : name_(name), passes_{}, run_only_once_(run_only_once) {}
  virtual ~PassGroup() = default;
  // Add graph pass, the pass object will be freed when pass manager freed.
  void AddPass(const PythonPassPtr &pass);
  // Delete graph pass before the pass manager is freed.
  bool DeletePass(const std::string &pass_name);
  // Run passes added in pass manager on the input graph
  // @param [inout] graph The graph to be optimized
  // @return true, graph changed
  // @return false, graph not changed
  bool Run(const FuncGraphPtr &func_graph) const;
  // Run the given graph passes on the input graph
  // @param [inout] func_graph The graph to be optimized
  // @param [in] passes The given graph passes
  // @param [inout] res MatchResult used to collect all matched patterns and nodes
  // @return true, graph changed
  // @return false, graph not changed
  bool Run(const FuncGraphPtr &func_graph, const std::vector<PythonPassPtr> &passes, const MatchResultPtr &res) const;
  std::string name() const { return name_; }
  void SetRunOnlyOnce(bool run_only_once) { run_only_once_ = run_only_once; }
  size_t size() { return passes_.size(); }

 private:
  const std::string name_;
  std::vector<PythonPassPtr> passes_;
  bool run_only_once_;
};
using PassGroupPtr = std::shared_ptr<PassGroup>;
}  // namespace python_pass
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_PASS_GROUP_H_
