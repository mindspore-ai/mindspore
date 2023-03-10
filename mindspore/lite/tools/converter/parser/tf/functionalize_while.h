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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TF_FUNCTIONALIZE_WHILE_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TF_FUNCTIONALIZE_WHILE_H_
#include <string>
#include <set>
#include <vector>
#include <map>
#include "include/backend/optimizer/pass.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/converter/parser/tf/functionalize_control_op_pass.h"

namespace mindspore::opt {

constexpr const int POS_INVALID = -1;

class FunctionalizeWhile {
 public:
  FunctionalizeWhile(std::vector<AnfNodePtr> node_cluster, const CNodePtr &loop_cond_node, FuncGraphPtr fg)
      : node_cluster_(node_cluster), loop_cond_node_(loop_cond_node), fg_(fg) {}

  virtual ~FunctionalizeWhile() = default;

  // while
  STATUS BuildWhileNode();
  STATUS IdentifyWhileNodeInput();
  STATUS IdentifyWhileNodeExternalInput();
  STATUS IdentifyWhileNodeOutput();
  STATUS UpdateExitNodeUser();
  STATUS NewWhileNode();
  STATUS InsertFuncGraphToWhileInput();
  bool WhileNodeExternalInputIsContain(const AnfNodePtr &node);

  // cond subgraph
  STATUS BuildCondGraph();
  STATUS CondSubgraphAddNodes();
  STATUS IdentifyCondSubgraphInput();
  STATUS IdentifyCondSubgraphOutput();

  // body subgraph
  STATUS BuildBodyGraph();
  STATUS BodySubgraphAddNodes();
  STATUS IdentifyBodySubgraphInput();
  STATUS IdentifyBodySubgraphOutput();

  CNodePtr BlongToWhichSwitch(const CNodePtr &node);
  CNodePtr BlongToWhichMerge(const CNodePtr &node);
  CNodePtr BlongToWhichEnter(const CNodePtr &node);
  CNodePtr BlongToWhichExternalEnter(const CNodePtr &node);
  int PosInInputEnterNodes(const CNodePtr &node);
  STATUS DropUselessNodesInMainGraph();

  STATUS Process();

 private:
  std::vector<AnfNodePtr> node_cluster_{};
  const CNodePtr loop_cond_node_;
  FuncGraphPtr fg_;

  FuncGraphPtr cond_sub_func_graph_ = nullptr;
  FuncGraphPtr body_sub_func_graph_ = nullptr;
  CNodePtr while_node_ = nullptr;

  std::string cond_subgraph_name_{};
  std::string body_subgraph_name_{};

  // while
  std::vector<CNodePtr> input_enter_nodes_{};
  std::vector<CNodePtr> external_input_enter_nodes_{};
  std::vector<CNodePtr> output_exit_nodes_{};

  // pair (next iteration node, next iteration node input)
  std::map<AnfNodePtr, AnfNodePtr> body_subgraph_output_map_{};
  // pair (switch node, switch output in body graph)
  std::map<AnfNodePtr, AnfNodePtr> body_subgraph_input_map_{};
  // pair (switch node, switch output in body graph)
  std::map<AnfNodePtr, AnfNodePtr> cond_subgraph_input_map_{};
};

}  // namespace mindspore::opt
#endif  // MINDSPORE_LITE_SRC_PASS_FUNCTIONALIZE_WHILE_PASS_H_
