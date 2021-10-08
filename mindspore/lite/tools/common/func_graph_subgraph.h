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

#ifndef MINDSPORE_LITE_TOOLS_FUNC_GRAPH_SUBGRAPH_H
#define MINDSPORE_LITE_TOOLS_FUNC_GRAPH_SUBGRAPH_H

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <utility>
#include "src/common/log_adapter.h"
#include "include/errorcode.h"
#include "ir/anf.h"
#include "ir/func_graph.h"

namespace mindspore::lite {
class SubGraph;
using SubGraphPtr = std::shared_ptr<SubGraph>;
class SubGraph {
 public:
  explicit SubGraph(FuncGraphPtr belong_anf, std::string graph_name = "")
      : belong_anf_(std::move(belong_anf)), name_(std::move(graph_name)) {}

  virtual ~SubGraph() = default;
  int Init(const std::set<CNodePtr> &head_nodes = {});
  int Reset(const std::set<CNodePtr> &nodes, const std::set<CNodePtr> &head_nodes = {});

  bool MergeSubGraph(const SubGraphPtr &subgraph);

  std::set<CNodePtr> GetNodes() const;
  std::set<CNodePtr> GetInCNodes() const;
  std::set<CNodePtr> GetOutCNodes() const;

  int ApplySubGraph();

 private:
  std::set<CNodePtr> GetInputCNodes() const;
  std::set<CNodePtr> GetOutputCNodes() const;
  // init subgraph methods
  int InitSubGraphNode(const std::set<CNodePtr> &head_nodes);
  int InitSubGraphInNode();
  int InitSubGraphOutNode();
  // merge subgraph methods
  std::set<CNodePtr> FindCommonOutputs(const SubGraphPtr &subgraph) const;
  bool IfDependOnSameNode(const SubGraphPtr &subgraph) const;
  // apply subgraph methods
  SubGraphPtr FindBeforeSubGraphInBelongAnf() const;
  SubGraphPtr FindAfterSubGraphInBelongAnf() const;
  int CreateParameterForPartialSubGraph(const FuncGraphPtr &sub_graph, std::vector<AnfNodePtr> *partial_inputs,
                                        std::map<AnfNodePtr, AnfNodePtr> *partial_inputs_and_subgraph_input_map);
  int CreateCNodeForPartialSubGraph(const FuncGraphPtr &sub_graph,
                                    const std::map<AnfNodePtr, AnfNodePtr> &partial_inputs_and_subgraph_input_map);
  int CreatePartialInBelongAnf();
  static int SetFuncGraphOutput(const FuncGraphPtr &graph, const std::set<CNodePtr> &outputs);

 private:
  std::set<CNodePtr> nodes_;
  std::set<CNodePtr> in_nodes_;
  std::set<CNodePtr> out_nodes_;
  const FuncGraphPtr belong_anf_ = nullptr;
  const std::string name_;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_TOOLS_FUNC_GRAPH_SUBGRAPH_H
