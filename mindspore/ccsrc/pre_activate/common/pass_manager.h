/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PRE_ACTIVATE_COMMON_PASS_MANAGER_H_
#define MINDSPORE_CCSRC_PRE_ACTIVATE_COMMON_PASS_MANAGER_H_

#include <utility>
#include <vector>
#include <string>
#include <memory>

#include "pre_activate/common/pass.h"
#include "pre_activate/common/node_pass.h"

namespace mindspore {
namespace opt {
// @brief For optimization passes management
class PassManager {
 public:
  explicit PassManager(const std::string &name = "pm", bool run_only_once = true)
      : name_(name), passes_{}, run_only_once_(run_only_once) {}
  virtual ~PassManager() = default;
  // Get all the passes added by AddPass
  const std::vector<PassPtr> &Passes() const;
  // Add graph pass, the pass object will be freed when pass manager freed.
  void AddPass(const PassPtr &pass);
  // Run passes added in pass manager on the input graph
  // @param [inout] graph The graph to be optimized
  // @return true, graph changed
  // @return false, graph not changed
  bool Run(const FuncGraphPtr &func_graph) const;
  // Run the given graph passes on the input graph
  // @param [inout] graph The graph to be optimized
  // @param [in] passes The given graph passes
  // @return true, graph changed
  // @return false, graph not changed
  bool Run(const FuncGraphPtr &func_graph, const std::vector<PassPtr> &passes) const;
  std::string name() const { return name_; }

 private:
  const std::string name_;
  std::vector<PassPtr> passes_;
  bool run_only_once_;
};
using PassManagerPtr = std::shared_ptr<PassManager>;
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PRE_ACTIVATE_COMMON_PASS_MANAGER_H_
