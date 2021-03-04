/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_FUNCTIONALIZE_COND_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_FUNCTIONALIZE_COND_H_

#include <string>
#include <set>
#include <vector>
#include <map>
#include "backend/optimizer/common/pass.h"
#include "tools/converter/converter_flags.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/optimizer/graph/functionalize_control_op_pass.h"

using mindspore::lite::converter::FmkType;
namespace mindspore::opt {

typedef enum { kThenBranch = 0, kElseBranch = 1 } BranchType;

// Functionalize all the switch-merge nodes of a loop-free graph into single switch node.
// Precondition: While loops must have been functionalized.
class FunctionalizeCond {
 public:
  FunctionalizeCond(FuncGraphPtr fg, CNodePtr merge_node) : fg_(fg), merge_node_(merge_node) {}

  STATUS Process();

 private:
  STATUS GetSwitchBranchType(const CNodePtr &switch_cnode, BranchType *branch_type);
  STATUS BranchSubGraphAddNodes(const FuncGraphPtr &graph, const AnfNodePtr &root_node, BranchType branch_type);
  FuncGraphPtr CreateBranchGraph(const AnfNodePtr &node, std::string name, BranchType branch_type);
  int PosInInputNodes(const CNodePtr &node);
  STATUS IdentifySubgraphInput(const FuncGraphPtr &graph, std::string graph_name);
  CNodePtr CreateNewIf(const FuncGraphPtr &else_branch, const FuncGraphPtr &then_branch);
  STATUS VerifyPredictNode();

  FuncGraphPtr fg_ = nullptr;
  CNodePtr merge_node_ = nullptr;
  CNodePtr pred_node_ = nullptr;
  std::vector<CNodePtr> input_nodes_{};
  std::vector<AnfNodePtr> pred_nodes_{};
};
}  // namespace mindspore::opt

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_FUNCTIONALIZE_COND_H_
