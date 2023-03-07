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
#include "backend/common/graph_kernel/core/graph_kernel_splitter.h"
#include <algorithm>
#include <vector>
#include <string>
#include <utility>
#include <queue>
#include "ir/anf.h"
#include "utils/anf_utils.h"
#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "backend/common/graph_kernel/core/graph_kernel_callback.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "backend/common/graph_kernel/core/convert_op_input_attr.h"
#include "backend/common/graph_kernel/split_model/split_model_factory.h"

namespace mindspore::graphkernel {
namespace {
void TraverseFuncGraphFromCNode(const CNodePtr &cnode, const std::function<void(AnfNodePtr &)> &callback) {
  mindspore::HashSet<AnfNodePtr> visited;
  std::queue<AnfNodePtr> que;
  que.push(cnode);
  (void)visited.insert(cnode);
  while (!que.empty()) {
    auto ft_node = que.front();
    que.pop();
    callback(ft_node);
    auto ft_cnode = ft_node->cast<CNodePtr>();
    if (ft_cnode == nullptr) {
      continue;
    }
    for (const auto &in_node : ft_cnode->inputs()) {
      if (visited.count(in_node) == 0) {
        que.push(in_node);
        (void)visited.insert(in_node);
      }
    }
  }
}

// Visited each AnfNode once, use callback to do the job on AnfNode
inline void TraverseFuncGraph(const FuncGraphPtr &root, const std::function<void(AnfNodePtr &)> &callback) {
  TraverseFuncGraphFromCNode(root->get_return(), callback);
}

class Area {
 public:
  explicit Area(const AnfNodePtrList &anf_arr) {
    nodes_.insert(anf_arr.cbegin(), anf_arr.cend());
    for (auto &node : anf_arr) {
      auto cnode = node->cast<CNodePtr>();
      if (cnode == nullptr) {
        continue;
      }
      const auto &inputs = cnode->inputs();
      if (std::any_of(inputs.begin(), inputs.end(), [this](const AnfNodePtr &node) { return IsExternalCNode(node); })) {
        (void)spy_cnodes_.emplace_back(node);
      }
    }
  }

  ~Area() = default;

  // Set the external inputs of spy as a Parameter.
  void CreateParameters(const FuncGraphPtr &func_graph, mindspore::HashMap<ParameterPtr, AnfNodePtr> *param_node_map) {
    mindspore::HashMap<AnfNodePtr, ParameterPtr> node_param_map;
    for (auto node : this->spy_cnodes_) {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      for (size_t i = 1; i < cnode->inputs().size(); ++i) {
        AnfNodePtr in_node = cnode->input(i);
        if (!IsExternalCNode(in_node)) {
          continue;
        }
        auto it = node_param_map.find(in_node);
        if (it == node_param_map.end()) {
          auto new_param = std::make_shared<Parameter>(func_graph);
          new_param->set_abstract(in_node->abstract());
          func_graph->add_parameter(new_param);
          (void)node_param_map.emplace(in_node, new_param);
          cnode->set_input(i, new_param);
        } else {
          cnode->set_input(i, it->second);
        }
      }
    }
    this->spy_cnodes_.clear();  // spy list is not useful anymore
    for (auto &&elem : node_param_map) {
      (void)param_node_map->emplace(elem.second, elem.first);
    }
    return;
  }

  // Make a return node for traitor nodes.
  void CreateReturnNode(const FuncGraphPtr &func_graph, mindspore::HashMap<AnfNodePtr, size_t> *tuple_node_index) {
    // If there's no traitor in the area, it means that this area is the last part
    // of the original FuncGraph, it already contains the original Return node.
    if (traitor_nodes_.empty()) {
      for (auto &node : nodes_) {
        if (IsPrimitiveCNode(node, prim::kPrimReturn)) {
          func_graph->set_return(node->cast<CNodePtr>());
          node->set_func_graph(func_graph);
          return;
        }
      }
      MS_LOG(ERROR) << "Cannot find the return node in " << func_graph->ToString();
      return;
    }
    AnfNodePtrList return_inputs = {NewValueNode(prim::kPrimReturn)};
    if (traitor_nodes_.size() > 1) {
      // The area has multiple output, it's necessary to make a tuple for them.
      AnfNodePtrList maketuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
      AbstractBasePtrList abstracts;
      size_t i = 0;
      for (auto &traitor : traitor_nodes_) {
        (void)tuple_node_index->emplace(traitor, i++);
        (void)maketuple_inputs.emplace_back(traitor);
        (void)abstracts.emplace_back(traitor->abstract());
      }
      auto maketuple_node = func_graph->NewCNode(maketuple_inputs);
      maketuple_node->set_abstract(std::make_shared<abstract::AbstractTuple>(abstracts));
      (void)nodes_.insert(maketuple_node);
      (void)return_inputs.emplace_back(maketuple_node);
    } else {
      (void)return_inputs.emplace_back(traitor_nodes_[0]);
    }
    auto return_node = func_graph->NewCNode(return_inputs);
    return_node->set_abstract(return_inputs.back()->abstract());
    func_graph->set_return(return_node);
    (void)nodes_.insert(return_node);
    traitor_nodes_.clear();  // traitor list is not useful anymore
    return;
  }

  void AddTraitor(const AnfNodePtr &node) {
    if (std::find(traitor_nodes_.begin(), traitor_nodes_.end(), node) == traitor_nodes_.end()) {
      (void)traitor_nodes_.emplace_back(node);
    }
  }

  const mindspore::HashSet<AnfNodePtr> &nodes() const { return nodes_; }
  const std::vector<AnfNodePtr> &spy_cnodes() const { return spy_cnodes_; }

 private:
  // This is a CNode that does not belong to this area.
  bool IsExternalCNode(const AnfNodePtr &node) const { return node->isa<CNode>() && this->nodes_.count(node) == 0; }

  // nodes in this area
  mindspore::HashSet<AnfNodePtr> nodes_;
  // if a node's output is used by other Area, it's a traitor
  std::vector<AnfNodePtr> traitor_nodes_;
  // if a node use other Area's output, it's a spy
  std::vector<AnfNodePtr> spy_cnodes_;
};

class AreaGraph {
 public:
  using AreaGraphPtr = std::shared_ptr<AreaGraph>;

  // Build an area graph to maintain the relation between areas.
  // Input node_groups: A group list, each element is a AnfNode list representing the node set in this group.
  static AreaGraphPtr BuildAreaGraph(const std::vector<AnfNodePtrList> &node_groups) {
    auto area_graph = std::make_shared<AreaGraph>(node_groups);
    if (area_graph == nullptr) {
      return nullptr;
    }
    if (!area_graph->TopoSort()) {
      MS_LOG(WARNING) << "The groups have a cycle. The first node is " << node_groups[0][0]->fullname_with_scope();
      return nullptr;
    }
    return area_graph;
  }

  // Split the graph to multiple areas, and reconnect the edges between the areas.
  // The output `main_cnodes` is a topo-sorted cnode list in main graph, holding the new sub_func_graphs.
  // The output `cnode_group_id` represents the indices of main_cnodes before topo-sorting.
  void SplitGraph(const FuncGraphPtr &main_func_graph, std::vector<CNodePtr> *main_cnodes,
                  std::vector<size_t> *cnode_group_id, const std::function<void(const Area &)> &expand_callback) {
    main_cnodes->clear();
    main_cnodes->resize(areas_.size(), nullptr);

    for (auto &area : this->areas_) {
      expand_callback(area);
    }

    for (auto index : topo_order_) {
      auto &current_area = areas_[index];
      auto sub_func_graph = std::make_shared<FuncGraph>();
      mindspore::HashMap<ParameterPtr, AnfNodePtr> param_node_map;

      current_area.CreateParameters(sub_func_graph, &param_node_map);
      current_area.CreateReturnNode(sub_func_graph, &node_index_in_returned_tuple_);
      auto new_main_cnode = this->CreateMainCNode(main_func_graph, sub_func_graph, *main_cnodes, param_node_map);
      (*main_cnodes)[index] = new_main_cnode;
    }

    SortCNodes(main_cnodes);
    *cnode_group_id = std::move(topo_order_);  // The topo_order is not used anymore.
    return;
  }

  explicit AreaGraph(const std::vector<AnfNodePtrList> &node_groups) : edge_prev_(node_groups.size()) {
    for (size_t i = 0; i < node_groups.size(); ++i) {
      (void)areas_.emplace_back(node_groups[i]);
      for (const auto &node : node_groups[i]) {
        node_area_map_[node] = i;
      }
    }
    for (auto &area : areas_) {
      for (auto &spy : area.spy_cnodes()) {
        auto cnode = spy->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(cnode);
        size_t v = node_area_map_[spy];
        for (auto &in_node : cnode->inputs()) {
          if (!in_node->isa<CNode>()) {
            continue;
          }
          // area edge u -> v
          size_t u = node_area_map_[in_node];
          if (u == v) {
            continue;
          }
          areas_[u].AddTraitor(in_node);
          if (std::find(edge_prev_[v].begin(), edge_prev_[v].end(), u) == edge_prev_[v].end()) {
            (void)edge_prev_[v].emplace_back(u);
          }
        }
      }
    }
  }
  ~AreaGraph() = default;

 private:
  // Topological sort the areas.
  bool TopoSort() {
    std::vector<int> out_degree(edge_prev_.size(), 0);
    std::queue<size_t> que;
    for (auto &prev : edge_prev_) {
      for (size_t i : prev) {
        out_degree[i]++;
      }
    }
    for (size_t i = 0; i < out_degree.size(); ++i) {
      if (out_degree[i] == 0) {
        que.push(i);
      }
    }
    while (!que.empty()) {
      size_t u = que.front();
      que.pop();
      (void)topo_order_.emplace_back(u);
      for (size_t i : edge_prev_[u]) {
        if (--out_degree[i] == 0) {
          que.push(i);
        }
      }
    }
    std::reverse(topo_order_.begin(), topo_order_.end());
    return topo_order_.size() == areas_.size();
  }

  // Make a CNode in main graph to hold the sub_func_graph.
  CNodePtr CreateMainCNode(const FuncGraphPtr &main_func_graph, const FuncGraphPtr &sub_func_graph,
                           const std::vector<CNodePtr> &main_cnodes,
                           const mindspore::HashMap<ParameterPtr, AnfNodePtr> &param_node_map) {
    TraceGuard guard(std::make_shared<TraceOpt>(sub_func_graph->debug_info()));
    AnfNodePtrList main_cnode_inputs = {NewValueNode(sub_func_graph)};
    for (const auto &param : sub_func_graph->parameters()) {
      // assert the param exists.
      const auto &input_node = param_node_map.find(param->cast<ParameterPtr>())->second;
      size_t input_area = node_area_map_[input_node];
      // if the input node is in a tuple, then we need to create a GetItem fot it.
      if (node_index_in_returned_tuple_.count(input_node) != 0) {
        auto idx_val = SizeToLong(node_index_in_returned_tuple_[input_node]);
        auto idx = NewValueNode(idx_val);
        idx->set_abstract(std::make_shared<abstract::AbstractScalar>(idx_val));
        AnfNodePtrList getitem_inputs = {NewValueNode(prim::kPrimTupleGetItem), main_cnodes[input_area], idx};
        TraceGuard g_sub(std::make_shared<TraceOpt>(main_cnodes[input_area]->debug_info()));
        auto getitem_node = main_func_graph->NewCNode(getitem_inputs);
        auto abs_tuple = dyn_cast<abstract::AbstractTuple>(main_cnodes[input_area]->abstract());
        if (idx_val < SizeToLong(abs_tuple->size())) {
          getitem_node->set_abstract(abs_tuple->elements()[LongToSize(idx_val)]);
        } else {
          getitem_node->set_abstract(main_cnodes[input_area]->abstract());
        }
        (void)main_cnode_inputs.emplace_back(getitem_node);
      } else {
        (void)main_cnode_inputs.emplace_back(main_cnodes[input_area]);
      }
    }
    auto new_main_cnode = main_func_graph->NewCNode(main_cnode_inputs);
    new_main_cnode->set_abstract(sub_func_graph->output()->abstract());
    return new_main_cnode;
  }

  void SortCNodes(std::vector<CNodePtr> *main_cnodes) const {
    std::vector<CNodePtr> main_cnodes_sorted;
    (void)std::transform(topo_order_.begin(), topo_order_.end(), std::back_inserter(main_cnodes_sorted),
                         [main_cnodes](size_t index) { return main_cnodes->at(index); });
    *main_cnodes = std::move(main_cnodes_sorted);
  }

  // Areas in this subgraph
  std::vector<Area> areas_;
  // Adjacency table of areas
  std::vector<std::vector<size_t>> edge_prev_;
  // Topological order of areas
  std::vector<size_t> topo_order_;
  // Map AnfNode to Area id
  mindspore::HashMap<AnfNodePtr, size_t> node_area_map_;
  // Map the nodes to their index if there are multiple value in an area
  mindspore::HashMap<AnfNodePtr, size_t> node_index_in_returned_tuple_;
};

class Splitter {
 public:
  using SplitterPtr = std::shared_ptr<Splitter>;

  bool Split() {
    GenParamMap();
    auto ori_sub_func_graph = GetCNodeFuncGraph(old_subgraph_cnode_);
    if (!split_schemer_->Split(ori_sub_func_graph)) {
      return false;
    }

    auto area_graph = AreaGraph::BuildAreaGraph(split_schemer_->split_plan());
    if (area_graph == nullptr) {
      return false;
    }

    // The output new_subgraph_cnodes are topo sorted, use a list to store its order in split_plan.
    std::vector<size_t> cnodes_group_id;
    area_graph->SplitGraph(main_func_graph_, &new_subgraph_cnodes_, &cnodes_group_id,
                           [this](const Area &area) { this->AreaExpand(area); });

    RebuildGraph(cnodes_group_id);

    return true;
  }

  static SplitterPtr MakeSplitter(const CNodePtr &main_cnode, const SplitSchemerPtr &split_schemer) {
    MS_EXCEPTION_IF_NULL(main_cnode);
    MS_EXCEPTION_IF_NULL(main_cnode->func_graph());
    MS_EXCEPTION_IF_NULL(split_schemer);
    return std::make_shared<Splitter>(main_cnode, split_schemer);
  }

  Splitter(const CNodePtr &main_cnode, const SplitSchemerPtr &split_schemer)
      : main_func_graph_(main_cnode->func_graph()), old_subgraph_cnode_(main_cnode), split_schemer_(split_schemer) {}
  ~Splitter() = default;

 private:
  void ResetInlinedNodesKernelInfo() const {
    for (const auto &node : inlined_nodes_) {
      ConvertOpUtils::ConvertAttrToInput(node);
      Callback::Instance()->ResetKernelInfo(node);
    }
  }

  // Maintain new subgraphs in main graph.
  void RebuildGraph(const std::vector<size_t> &cnodes_group_id) {
    BindFuncGraph();
    RecoverParameter();
    SetSplitNodeName(cnodes_group_id);
    ConnectToMainGraph(cnodes_group_id);
    UpdateSubGraphInfo();
    ResetInlinedNodesKernelInfo();
  }

  // Rebind nodes to its new sub_func_graph
  void BindFuncGraph() const {
    for (const auto &cnode : new_subgraph_cnodes_) {
      auto sub_func_graph = GetCNodeFuncGraph(cnode);
      auto callback = [&sub_func_graph](const AnfNodePtr &node) {
        if (!node->isa<ValueNode>()) {
          node->set_func_graph(sub_func_graph);
        }
      };
      TraverseFuncGraph(sub_func_graph, callback);
    }
  }

  // Recover the original subgraph's parameter if the new graph needs it
  void RecoverParameter() {
    for (const auto &cnode : new_subgraph_cnodes_) {
      auto sub_func_graph = GetCNodeFuncGraph(cnode);
      auto callback = [&cnode, &sub_func_graph, this](const AnfNodePtr &node) {
        auto param = node->cast<ParameterPtr>();
        if (param == nullptr) {
          return;
        }
        auto it = this->param_to_main_graph_node_map_.find(param);
        if (it != this->param_to_main_graph_node_map_.end()) {
          auto input = it->second;
          cnode->add_input(input);
          sub_func_graph->add_parameter(param);
          // Avoid repeating parameters.
          (void)this->param_to_main_graph_node_map_.erase(it);
        }
      };
      TraverseFuncGraph(sub_func_graph, callback);
    }
  }

  CNodePtr InlineSubFuncGraph(const CNodePtr &main_node) {
    auto func_graph = GetCNodeFuncGraph(main_node);
    const auto &inputs = main_node->inputs();
    auto output = func_graph->output()->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(output);
    const auto &parameters = func_graph->parameters();
    mindspore::HashMap<AnfNodePtr, AnfNodePtr> param_input;
    for (size_t i = 0; i < parameters.size(); ++i) {
      param_input[parameters[i]] = inputs[i + 1];
    }
    auto sub_nodes = TopoSort(func_graph->get_return());
    for (auto node : sub_nodes) {
      if (auto cnode = node->cast<CNodePtr>(); cnode != nullptr) {
        cnode->set_func_graph(main_func_graph_);
        for (size_t i = 1; i < cnode->inputs().size(); ++i) {
          auto iter = param_input.find(cnode->input(i));
          if (iter != param_input.end()) {
            cnode->set_input(i, iter->second);
          }
        }
        if (AnfUtils::IsRealKernel(node)) {
          (void)inlined_nodes_.emplace_back(node);
        }
      }
    }
    return output;
  }

  void SetSplitNodeName(const std::vector<size_t> &cnodes_group_id) const {
    auto old_func_graph = GetCNodeFuncGraph(old_subgraph_cnode_);
    std::string ori_node_name;
    if (old_func_graph->has_attr(kAttrNodeName)) {
      ori_node_name = GetValue<std::string>(old_func_graph->get_attr(kAttrNodeName));
    } else {
      ori_node_name = GetValue<std::string>(old_func_graph->get_attr("graph_kernel"));
    }
    for (size_t i = 0; i < new_subgraph_cnodes_.size(); ++i) {
      auto group_id = cnodes_group_id[i];
      if (!split_schemer_->NeedInline(group_id)) {
        std::string node_name = ori_node_name + "_" + std::to_string(group_id);
        AnfUtils::SetNodeAttr(kAttrNodeName, MakeValue(node_name), new_subgraph_cnodes_[i]);
      }
    }
  }

  // Set the new sub_func_graph node as input of nodes original main graph.
  void ConnectToMainGraph(const std::vector<size_t> &cnodes_group_id) {
    // For single output kernel, the last area contains the original output node (return node),
    //  to replace old subgraph with new subgraphs, just replace the old CNode with new last CNode.
    // For multiple output kernel, to avoid returning Parameter, the last MakeTuple was distribute to
    //  a new FuncGraph, just inline the last MakeTuple node.
    std::vector<CNodePtr> tmp_subgraph_cnodes;
    mindspore::HashMap<AnfNodePtr, AnfNodePtr> replace_map;

    for (size_t i = 0; i < new_subgraph_cnodes_.size(); ++i) {
      if (split_schemer_->NeedInline(cnodes_group_id[i])) {
        // Connect the sub_graph's inner node to main_graph
        auto output = InlineSubFuncGraph(new_subgraph_cnodes_[i]);
        if (i + 1 == new_subgraph_cnodes_.size()) {
          replace_map[this->old_subgraph_cnode_] = output;
        } else {
          replace_map[new_subgraph_cnodes_[i]] = output;
        }
      } else {
        if (i + 1 == new_subgraph_cnodes_.size()) {
          replace_map[this->old_subgraph_cnode_] = new_subgraph_cnodes_.back();
        }
        (void)tmp_subgraph_cnodes.emplace_back(new_subgraph_cnodes_[i]);
      }
    }
    new_subgraph_cnodes_ = std::move(tmp_subgraph_cnodes);

    TraverseFuncGraph(main_func_graph_, [&replace_map](const AnfNodePtr &node) {
      auto cnode = node->cast<CNodePtr>();
      if (cnode == nullptr) {
        return;
      }
      for (size_t i = 1; i < cnode->inputs().size(); ++i) {
        auto input_node = cnode->input(i);
        auto iter = replace_map.find(input_node);
        if (iter != replace_map.end()) {
          cnode->set_input(i, iter->second);
        }
      }
    });
  }

  void UpdateSubGraphInfo() const {
    auto graph_manager = main_func_graph_->manager();
    MS_EXCEPTION_IF_NULL(graph_manager);

    for (auto cnode : new_subgraph_cnodes_) {
      auto sub_func_graph = GetCNodeFuncGraph(cnode);
      // add new sub_func_graph to manager
      graph_manager->AddFuncGraph(sub_func_graph);

      // set GraphKernel attr
      auto attr = GkUtils::ExtractGraphKernelName(TopoSort(sub_func_graph->get_return()), "", "split");
      sub_func_graph->set_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, MakeValue(attr));

      // set kernel info
      Callback::Instance()->SetGraphKernelNodeKernelInfo(cnode);
    }
  }

  // Copy all Parameter and ValueNode that the area used.
  void AreaExpand(const Area &area) {
    mindspore::HashMap<AnfNodePtr, AnfNodePtr> old_valuenode_and_param_map;
    for (auto sub_node : area.nodes()) {
      auto sub_cnode = sub_node->cast<CNodePtr>();
      if (sub_cnode == nullptr) {
        continue;
      }
      for (size_t i = 1; i < sub_cnode->inputs().size(); ++i) {
        auto in_node = sub_cnode->input(i);
        if (in_node->isa<CNode>()) {
          continue;
        }
        auto it = old_valuenode_and_param_map.find(in_node);
        if (it != old_valuenode_and_param_map.end()) {
          sub_cnode->set_input(i, it->second);
        } else {
          if (in_node->isa<Parameter>()) {
            auto param = in_node->cast<ParameterPtr>();
            auto cp_param = this->ParameterClone(param, in_node->func_graph());
            old_valuenode_and_param_map[in_node] = cp_param->cast<AnfNodePtr>();
            sub_cnode->set_input(i, cp_param);
          }
        }
      }
    }
  }

  void GenParamMap() {
    auto sub_func_graph = GetCNodeFuncGraph(old_subgraph_cnode_);
    auto &param_arr = sub_func_graph->parameters();
    for (size_t i = 0; i < param_arr.size(); ++i) {
      auto param = param_arr[i]->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(param);
      param_to_main_graph_node_map_[param] = old_subgraph_cnode_->input(i + 1);
    }
  }

  ParameterPtr ParameterClone(const ParameterPtr &param, const FuncGraphPtr &func) {
    ParameterPtr param_c = std::make_shared<Parameter>(func);
    param_c->set_name(param->name());
    param_c->set_abstract(param->abstract());
    auto node = param_to_main_graph_node_map_[param];
    param_to_main_graph_node_map_[param_c] = node;
    return param_c;
  }

  FuncGraphPtr main_func_graph_;
  CNodePtr old_subgraph_cnode_;                // The cnode that holds the original sub_func_graph
  std::vector<CNodePtr> new_subgraph_cnodes_;  // The cnode list that hold the new sub_func_graph
  std::vector<AnfNodePtr> inlined_nodes_;
  SplitSchemerPtr split_schemer_;
  mindspore::HashMap<ParameterPtr, AnfNodePtr> param_to_main_graph_node_map_;
};

class CppCostModelSplitSchemer : public CommonSplitSchemer {
 public:
  explicit CppCostModelSplitSchemer(const std::string &processor) : processor_(processor) {}
  ~CppCostModelSplitSchemer() = default;
  bool Split(const FuncGraphPtr &func_graph) override {
    if (!SplitByCostModel(func_graph)) {
      return false;
    }
    GroupReturnNode(func_graph);
    return true;
  }

 protected:
  bool SplitByCostModel(const FuncGraphPtr &func_graph) {
    mindspore::HashMap<inner::NodePtr, AnfNodePtr> op_node_map;
    auto lg = GkUtils::AnfGraph2LiteGraph(func_graph, &op_node_map);
    MS_LOG(DEBUG) << "Litegraph: " << lg->ToString();
    // use the original node index to sort the split_plan's nodes.
    mindspore::HashMap<AnfNodePtr, size_t> node_idx_map;
    for (size_t i = 0; i < lg->ops().size(); ++i) {
      node_idx_map[op_node_map[lg->ops()[i]]] = i;
    }
    auto model = inner::SplitModelFactory::Instance().CreateSplitModel(processor_);
    MS_EXCEPTION_IF_NULL(model);
    model->Run(lg);
    auto &areas = model->areas();
    for (auto &area : areas) {
      AnfNodePtrList nodes;
      for (auto &op : area->ops()) {
        (void)nodes.emplace_back(op_node_map[op]);
        node_group_[nodes.back()] = split_plan_.size();
      }
      std::sort(nodes.begin(), nodes.end(), [&node_idx_map](const AnfNodePtr &a, const AnfNodePtr &b) {
        return node_idx_map[a] < node_idx_map[b];
      });
      (void)split_plan_.emplace_back(std::move(nodes));
      need_inline_.push_back((area->mode() == inner::AreaMode::BASIC ? 1 : 0));
    }
    return split_plan_.size() > 1 || (split_plan_.size() == 1 && NeedInline(0));
  }

  std::string processor_;
};
}  // namespace

std::shared_ptr<SplitSchemer> GraphKernelSplitter::GetSplitSchema(const std::string &processor) {
  return std::make_shared<CppCostModelSplitSchemer>(processor);
}

bool GraphKernelSplitter::TrySplit(const CNodePtr &sub_root_cnode) {
  MS_LOG(DEBUG) << "Split process node: " << sub_root_cnode->fullname_with_scope();
  auto processor = Callback::Instance()->GetTargetFromContext();
  auto schm = GetSplitSchema(processor);
  MS_EXCEPTION_IF_NULL(schm);
  auto splitter = Splitter::MakeSplitter(sub_root_cnode, schm);
  MS_EXCEPTION_IF_NULL(splitter);
  bool result = splitter->Split();
  MS_LOG(DEBUG) << "Split node completed, result: " << result;
  return result;
}

bool GraphKernelSplitter::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }
  auto todos = TopoSort(func_graph->get_return());

  // Split subgraphs in reversed topo order,
  // since the nodes behind the processing node may be modified when splitting.
  bool changed = false;
  for (auto iter = todos.crbegin(); iter != todos.crend(); ++iter) {
    auto node = (*iter)->cast<CNodePtr>();
    if (node != nullptr && AnfUtils::IsGraphKernel(node)) {
      changed = TrySplit(node) || changed;
    }
  }
  mng->RemoveRoots();
  mng->KeepRoots({func_graph});
  return changed;
}
}  // namespace mindspore::graphkernel
