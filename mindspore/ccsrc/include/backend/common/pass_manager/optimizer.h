/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_PASS_MANAGER_OPTIMIZER_H_
#define MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_PASS_MANAGER_OPTIMIZER_H_

#include <memory>
#include <string>
#include <vector>
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "include/backend/common/pass_manager/pattern_engine.h"
#include "include/backend/common/pass_manager/helper.h"
#include "include/backend/visible.h"
#include "include/backend/common/pass_manager/node_pass.h"
#include "include/backend/common/pass_manager/graph_optimizer.h"
#include "include/backend/common/kernel_graph/kernel_graph.h"

namespace mindspore {
class Visitor;

namespace opt {
using PatternListType = std::initializer_list<BaseRef>;

class BACKEND_COMMON_EXPORT PatternPass : public NodePass {
 public:
  explicit PatternPass(const std::string &name = "", bool multigraph = true);
  ~PatternPass() override = default;
  CNodePtr NewCNode(const std::vector<AnfNodePtr> &inputs, const FuncGraphPtr &fg) const;
  CNodePtr NewCNode(const CNodePtr &cnode, const KernelGraphPtr &fg) const;

 protected:
  void GetOrigNodes();
  bool CheckNodeStreamAndCoreAttrs(const FuncGraphPtr &func_graph) const;
  bool multigraph_ = true;
  PatternEngine pattern_engine_;
  PrimitiveVarMapPtr primitive_vars_;
  EquivPtr equiv_;
  std::vector<AnfNodePtr> orig_nodes_;
};

class BACKEND_COMMON_EXPORT PatternProcessPass : public PatternPass {
 public:
  explicit PatternProcessPass(const std::string &name = "", bool multigraph = true) : PatternPass(name, multigraph) {}
  ~PatternProcessPass() override = default;
  virtual const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const = 0;
  virtual const BaseRef DefinePattern() const;
  AnfNodePtr Run(const FuncGraphPtr &func_graph, const AnfNodePtr &node) override;

 private:
  void Build();
  AnfNodePtr pattern_ = nullptr;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_PASS_MANAGER_OPTIMIZER_H_
