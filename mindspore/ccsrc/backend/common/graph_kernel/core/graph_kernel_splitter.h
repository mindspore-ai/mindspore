/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_GRAPH_KERNEL_SPLITTER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_GRAPH_KERNEL_SPLITTER_H_
#include <memory>
#include <string>
#include <vector>
#include "ir/func_graph.h"
#include "include/backend/optimizer/pass.h"
#include "backend/common/graph_kernel/core/split_schemer.h"

namespace mindspore::graphkernel {
class GraphKernelSplitter : public opt::Pass {
 public:
  explicit GraphKernelSplitter(const std::string name = "graph_kernel_splitter") : Pass(name) {}
  ~GraphKernelSplitter() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;
  virtual bool TrySplit(const CNodePtr &sub_root_cnode);
  virtual SplitSchemerPtr GetSplitSchema(const std::string &processor);
  virtual bool CanSplit(const CNodePtr &node) const;
};
using GraphKernelSplitterPtr = std::shared_ptr<GraphKernelSplitter>;

class Area {
 public:
  explicit Area(const AnfNodePtrList &anf_arr);
  ~Area() = default;
  // Set the external inputs of spy as a Parameter.
  void CreateParameters(const FuncGraphPtr &func_graph, mindspore::HashMap<ParameterPtr, AnfNodePtr> *param_node_map);
  // Make a return node for traitor nodes.
  void CreateReturnNode(const FuncGraphPtr &func_graph, mindspore::HashMap<AnfNodePtr, size_t> *tuple_node_index);
  void AddTraitor(const AnfNodePtr &node);
  const mindspore::HashSet<AnfNodePtr> &nodes() const;
  const std::vector<AnfNodePtr> &spy_cnodes() const;

 private:
  // This is a CNode that does not belong to this area.
  bool IsExternalCNode(const AnfNodePtr &node) const;
  // nodes in this area
  mindspore::HashSet<AnfNodePtr> nodes_;
  // if a node's output is used by other Area, it's a traitor
  std::vector<AnfNodePtr> traitor_nodes_;
  // if a node use other Area's output, it's a spy
  std::vector<AnfNodePtr> spy_cnodes_;
};

class Splitter {
 public:
  using SplitterPtr = std::shared_ptr<Splitter>;
  Splitter(const CNodePtr &main_cnode, const SplitSchemerPtr &split_schemer)
      : main_func_graph_(main_cnode->func_graph()), old_subgraph_cnode_(main_cnode), split_schemer_(split_schemer) {}
  virtual ~Splitter() = default;
  bool Split();

 protected:
  virtual void PostProcess() {}
  FuncGraphPtr main_func_graph_;
  CNodePtr old_subgraph_cnode_;                // The cnode that holds the original sub_func_graph
  std::vector<CNodePtr> new_subgraph_cnodes_;  // The cnode list that hold the new sub_func_graph
  std::vector<AnfNodePtr> maingraph_nodes_;    // The nodes in main graph finally, include "call" and inlined node
  SplitSchemerPtr split_schemer_;
  mindspore::HashMap<ParameterPtr, AnfNodePtr> param_to_main_graph_node_map_;

 private:
  // Maintain new subgraphs in main graph.
  void RebuildGraph(const std::vector<size_t> &cnodes_group_id);
  void BindFuncGraph() const;
  // Recover the original subgraph's parameter if the new graph needs it
  void RecoverParameter();
  CNodePtr InlineSubFuncGraph(const CNodePtr &main_node);
  void SetSplitNodeName(const std::vector<size_t> &cnodes_group_id) const;
  // Set the new sub_func_graph node as input of nodes original main graph.
  void ConnectToMainGraph(const std::vector<size_t> &cnodes_group_id);
  void UpdateMainNodesKernelInfo() const;
  // Copy all Parameter and ValueNode that the area used.
  void AreaExpand(const Area &area);
  void GenParamMap();
  ParameterPtr ParameterClone(const ParameterPtr &param, const FuncGraphPtr &func);
};

class SplitterCCE : public Splitter {
 public:
  SplitterCCE(const CNodePtr &main_cnode, const SplitSchemerPtr &split_schemer) : Splitter(main_cnode, split_schemer) {}
  virtual ~SplitterCCE() = default;

 protected:
  virtual void PostProcess();

 private:
  void SetCNodeUseCCE();
};

}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_GRAPH_KERNEL_SPLITTER_H_
