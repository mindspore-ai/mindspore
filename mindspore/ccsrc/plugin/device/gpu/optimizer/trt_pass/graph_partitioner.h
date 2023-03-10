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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_TRT_PASS_GRAPH_PARTITIONER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_TRT_PASS_GRAPH_PARTITIONER_H_

#include <memory>
#include <set>
#include <map>
#include <tuple>
#include <string>
#include "utils/hash_map.h"
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
enum class NodeType : char {
  kSupport = 0,  // Node could be convert to trt.
  kUnsupported,  // Node could not be convert to trt.
  kInvalid
};

// Class keep node information about type, topo index, and sub graph id.
struct NodeInfo {
  NodeInfo() : topo_index_(0), type_(NodeType::kInvalid), graph_id_(""), final_(false) {}
  explicit NodeInfo(const NodeType &t, const size_t &i) : topo_index_(i), type_(t), graph_id_(""), final_(false) {}

  const size_t topo_index() const { return topo_index_; }
  const NodeType &type() const { return type_; }
  const std::string &graph_id() const { return graph_id_; }
  const bool final() { return final_; }

  size_t topo_index_;
  NodeType type_;
  std::string graph_id_;
  bool final_;
};

// Represents dependencies between subgraphs for cyclic check when graph partition.
// It store the dependencies with key-value structure between subgraphs.
// The graph like:
//      graph-0   graph-1
//         |        |
//         v        v
//           graph-2
//             |
//             v
//           graph-3
//
// The data store in the class instance:
//    Key      Values
//    graph-2: {graph-0, graph-1}
//    graph-3: {grapph-0, graph-1, graph-2}
class GraphDependency {
 public:
  // Add dependency from rhs to lhs.
  void AddDependency(const string &lhs, const string &rhs) { dependencies_[lhs].insert(rhs); }

  // Inherit dependency from rhs to lhs.
  void InheritDependency(const string &lhs, const string &rhs);

  // Check whether lhs depend rhs.
  bool ExistDependency(const string &lhs, const string &rhs) const;

  // Display all dependencies.
  std::string ToString() const;

 private:
  mindspore::HashMap<std::string, std::set<std::string>> dependencies_;
};

using Subgraph = std::tuple<FuncGraphPtr, AnfNodePtrList, AnfNodePtrList>;

// Class for graph partition. The general graph partition contains three steps:
// 1. We first traverse all nodes in the graph and collect node information including node type, topological index.
// 2. Then we use node information to guide the node grouping. All consecutive nodes with the same type will
//    be divided into the same group, unless the grouping leads to cycle.
// 3. At last, we filter the interest groups and sort the node with topological order which is necessary for subgraph
// creation.
//
// The result contain inputs, output and function graph:
// Inputs: actual arguments of function.
// Outputs: user nodes in root graph.
// Function graph: including formal parameter, cnodes, and return node. It should be noted that the parameters will
// copy the name and default value of the actual arguments for building Trt graph latter.
class GraphPartitioner {
 public:
  GraphPartitioner() = default;
  ~GraphPartitioner() = default;

  // Graph segments with graph id and cnode list.
  std::map<std::string, AnfNodePtrList> Partition(const FuncGraphPtr &root_graph);

  // Create subgraph with segments. The result contain inputs, output and function graph
  Subgraph CreateNewGraph(const AnfNodePtrList &segments);

 private:
  void NewSubGraph(NodeInfo *node_info);
  bool ExistCycleAfterMerge(const AnfNodePtr &node, const std::string &target_graph_id);
  void MergeParentBranchRecursively(const AnfNodePtr &node, const std::string &old_graph_id,
                                    const std::string &new_graph_id);
  bool NodeGrouping(const FuncGraphPtr &func_graph);
  std::map<std::string, AnfNodePtrList> CollectSegments();

  mindspore::HashMap<AnfNodePtr, NodeInfo> node_info_;
  GraphDependency dependency_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_TRT_PASS_GRAPH_PARTITIONER_H_
