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
#ifndef MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_PASS_MANAGER_PASS_MANAGER_H_
#define MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_PASS_MANAGER_PASS_MANAGER_H_

#include <utility>
#include <vector>
#include <string>
#include <memory>
#include <map>

#include "include/backend/common/pass_manager/pass.h"
#include "include/backend/common/pass_manager/node_pass.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace opt {
// @brief For optimization passes management
class BACKEND_COMMON_EXPORT PassManager {
 public:
  explicit PassManager(const std::string &name = "pm", bool run_only_once = true);
  virtual ~PassManager() = default;
  // Get all the passes added by AddPass
  const std::vector<PassPtr> &Passes() const { return passes_; }
  // Add graph pass, the pass object will be freed when pass manager freed.
  // Please use AddFusionPass instead if graph pass is doing fusion.
  virtual void AddPass(const PassPtr &pass);
  // Add graph fusion pass, which can turn on/off by graphkernelflags.
  void AddFusionPass(const PassPtr &pass, bool condition = true);
  // Run passes added in pass manager on the input graph
  // @param [in out] graph The graph to be optimized
  // @return true, graph changed
  // @return false, graph not changed
  virtual bool Run(const FuncGraphPtr &func_graph) const;
  // Run the given graph passes on the input graph
  // @param [in out] graph The graph to be optimized
  // @param [in] passes The given graph passes
  // @return true, graph changed
  // @return false, graph not changed
  virtual bool Run(const FuncGraphPtr &func_graph, const std::vector<PassPtr> &passes) const;
  std::string name() const { return name_; }

 protected:
  virtual bool RunPass(const FuncGraphPtr &func_graph, size_t pass_id, const PassPtr &pass) const;
  virtual std::string GetPassFullname(size_t pass_id, const PassPtr &pass) const;
  virtual void DumpPassIR(const FuncGraphPtr &func_graph, const std::string &pass_fullname) const;

  const std::string name_;
  std::vector<PassPtr> passes_;
  bool run_only_once_;
  std::map<PassPtr, bool> fusion_passes_switch_;
};
using PassManagerPtr = std::shared_ptr<PassManager>;
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_PASS_MANAGER_PASS_MANAGER_H_
