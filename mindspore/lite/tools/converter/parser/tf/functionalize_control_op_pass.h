/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TF_FUNCTIONALIZE_CONTROL_OP_PASS_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TF_FUNCTIONALIZE_CONTROL_OP_PASS_H_
#include <string>
#include <set>
#include <utility>
#include <vector>
#include <memory>
#include "include/backend/optimizer/pass.h"
#include "tools/converter/ops/ops_def.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "include/registry/converter_context.h"

using mindspore::converter::FmkType;
namespace mindspore::opt {
using AimFunc = std::function<bool(const AnfNodePtr &)>;
class FunctionalizeControlOpPass : public Pass {
 public:
  FunctionalizeControlOpPass() : Pass("functionalize_control_op_pass") {}
  ~FunctionalizeControlOpPass() override = default;
  bool Run(const FuncGraphPtr &graph) override;
  static FuncGraphPtr NewFuncGraph(const std::string &subgraph_name, const FmkType &fmk_type);
  static bool IsMerge(const AnfNodePtr &node) { return CheckPrimitiveType(node, prim::kPrimMerge); }
  static bool IsLoopCond(const AnfNodePtr &node) {
    return CheckPrimitiveType(node, std::make_shared<Primitive>(lite::kNameLoopCond));
  }
  static bool IsEnter(const AnfNodePtr &node) {
    return CheckPrimitiveType(node, std::make_shared<Primitive>(lite::kNameEnter));
  }
  static bool IsExit(const AnfNodePtr &node) {
    return CheckPrimitiveType(node, std::make_shared<Primitive>(lite::kNameExit));
  }
  static bool IsSwitch(const AnfNodePtr &node) { return CheckPrimitiveType(node, prim::kPrimSwitch); }
  static bool IsNextIteration(const AnfNodePtr &node) {
    return CheckPrimitiveType(node, std::make_shared<Primitive>(lite::kNameNextIteration));
  }
  static bool IsControlFlowOp(const AnfNodePtr &node) {
    return IsLoopCond(node) || IsEnter(node) || IsMerge(node) || IsSwitch(node) || IsExit(node) ||
           IsNextIteration(node);
  }
  static CNodePtr BelongToWhichNode(const CNodePtr &node, const AimFunc &aim_func,
                                    const FilterFunc &filter_func = nullptr);
  static int GetSubgraphIndex() {
    static int subgraph_index = 1;
    return subgraph_index++;
  }
  // The names of nodes with the same prefix are a cluster.
  static std::string NodeClusterName(const AnfNodePtr &node);
  void InitNodeClusters(const FuncGraphPtr &func_graph);
  // return the position in node_clusters_
  size_t WhichCluster(const std::string &cluster_name);

 protected:
  STATUS BuildWhileSubgraph(const FuncGraphPtr &func_graph);
  static STATUS BuildIfSubgraph(const FuncGraphPtr &func_graph);
  std::vector<std::pair<std::string, std::vector<AnfNodePtr>>> node_clusters_{};
  std::vector<CNodePtr> loop_cond_nodes_{};
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TF_FUNCTIONALIZE_CONTROL_OP_PASS_H_
