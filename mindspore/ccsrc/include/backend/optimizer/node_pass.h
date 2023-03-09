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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_NODE_PASS_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_NODE_PASS_H_
#include <string>
#include <memory>
#include <vector>
#include <set>

#include "include/backend/optimizer/pass.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace opt {
// @brief ANF Node level optimization base pass
class BACKEND_EXPORT NodePass : public Pass {
 public:
  explicit NodePass(const std::string &name) : Pass(name) {}
  ~NodePass() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;
  virtual bool IsFastPass() { return false; }
  virtual void AfterProcess(const AnfNodePtr &, const AnfNodePtr &, const FuncGraphPtr &, const FuncGraphIndexPtr &) {}
  virtual std::string GetPatternRootPrimitiveName() { return ""; }
  virtual AnfNodePtr Run(const FuncGraphPtr &func_graph, const AnfNodePtr &node) = 0;
  virtual std::vector<std::string> MustExistPrimitiveName() const { return {}; }

 private:
  bool ProcessFastPassNode(const AnfNodePtr &node, const FuncGraphPtr &func_graph,
                           const FuncGraphIndexPtr &func_graph_index, const FuncGraphManagerPtr &manager);
  bool ProcessFastPass(const FuncGraphPtr &func_graph, const FuncGraphIndexPtr &func_graph_index);
  bool ProcessPass(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager);
};
void GenIndex(const FuncGraphPtr &func_graph, const FuncGraphIndexPtr &func_graph_index);
void ModifyOutputAndCallerToMap(const CNodePtr &cnode, const FuncGraphPtr &fg,
                                mindspore::HashMap<AnfNodePtr, std::set<AnfNodePtr>> *out_caller_map,
                                bool is_add = true);
std::string GetCNodeKey(const AnfNodePtr &node);
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_NODE_PASS_H_
