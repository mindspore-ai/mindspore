/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "backend/optimizer/graph_kernel/graph_kernel_splitter.h"
#include <algorithm>
#include <vector>
#include <string>
#include <unordered_set>
#include <utility>
#include <queue>
#include <map>
#include <unordered_map>
#include "frontend/optimizer/irpass.h"
#include "pipeline/jit/parse/python_adapter.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/kernel_compiler/common_utils.h"
#include "backend/kernel_compiler/akg/akg_kernel_json_decoder.h"
#include "backend/optimizer/graph_kernel/graph_kernel_helper.h"
#include "debug/anf_ir_dump.h"

namespace mindspore {
namespace opt {
namespace {
void TraverseFuncGraphFromCNode(const CNodePtr &cnode, std::function<void(AnfNodePtr &)> callback) {
  std::unordered_set<AnfNodePtr> visited;
  std::queue<AnfNodePtr> que;
  que.push(cnode);
  visited.insert(cnode);
  while (!que.empty()) {
    auto ft_node = que.front();
    que.pop();
    callback(ft_node);
    auto ft_cnode = ft_node->cast<CNodePtr>();
    if (ft_cnode == nullptr) continue;
    for (const auto &in_node : ft_cnode->inputs()) {
      if (visited.count(in_node) == 0) {
        que.push(in_node);
        visited.insert(in_node);
      }
    }
  }
}

// Visited each AnfNode once, use callback to do the job on AnfNode
inline void TraverseFuncGraph(const FuncGraphPtr &root, std::function<void(AnfNodePtr &)> callback) {
  TraverseFuncGraphFromCNode(root->get_return(), callback);
}

class Area {
 public:
  explicit Area(const AnfNodePtrList &anf_arr) {
    nodes_.insert(anf_arr.begin(), anf_arr.end());
    for (auto &node : anf_arr) {
      auto cnode = node->cast<CNodePtr>();
      if (cnode == nullptr) continue;
      const auto &inputs = cnode->inputs();
      if (std::any_of(inputs.begin(), inputs.end(), [this](const AnfNodePtr &node) { return IsExternalCNode(node); })) {
        spy_cnodes_.push_back(node);
      }
    }
  }

  ~Area() = default;

  // Set the external inputs of spy as a Parameter.
  void CreateParameters(const FuncGraphPtr &func_graph, std::unordered_map<ParameterPtr, AnfNodePtr> *param_node_map) {
    std::unordered_map<AnfNodePtr, ParameterPtr> node_param_map;
    for (auto node : this->spy_cnodes_) {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      for (size_t i = 1; i < cnode->inputs().size(); ++i) {
        AnfNodePtr in_node = cnode->input(i);
        if (!IsExternalCNode(in_node)) continue;
        auto it = node_param_map.find(in_node);
        if (it == node_param_map.end()) {
          auto new_param = std::make_shared<Parameter>(func_graph);
          new_param->set_abstract(in_node->abstract());
          func_graph->add_parameter(new_param);
          node_param_map.insert(std::make_pair(in_node, new_param));
          cnode->set_input(i, new_param);
        } else {
          cnode->set_input(i, it->second);
        }
      }
    }
    this->spy_cnodes_.clear();  // spy list is not useful anymore
    for (auto &&elem : node_param_map) {
      param_node_map->insert(std::make_pair(elem.second, elem.first));
    }
    return;
  }

  // Make a return node for traitor nodes.
  void CreateReturnNode(const FuncGraphPtr &func_graph, std::unordered_map<AnfNodePtr, size_t> *tuple_node_index) {
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
        tuple_node_index->insert(std::make_pair(traitor, i++));
        maketuple_inputs.push_back(traitor);
        abstracts.push_back(traitor->abstract());
      }
      auto maketuple_node = func_graph->NewCNode(maketuple_inputs);
      maketuple_node->set_abstract(std::make_shared<abstract::AbstractTuple>(abstracts));
      nodes_.insert(maketuple_node);
      return_inputs.push_back(maketuple_node);
    } else {
      return_inputs.push_back(traitor_nodes_[0]);
    }
    auto return_node = func_graph->NewCNode(return_inputs);
    return_node->set_abstract(return_inputs.back()->abstract());
    func_graph->set_return(return_node);
    nodes_.insert(return_node);
    traitor_nodes_.clear();  // traitor list is not useful anymore
    return;
  }

  void AddTraitor(const AnfNodePtr &node) {
    if (std::find(traitor_nodes_.begin(), traitor_nodes_.end(), node) == traitor_nodes_.end()) {
      traitor_nodes_.push_back(node);
    }
  }

  const std::unordered_set<AnfNodePtr> &nodes() const { return nodes_; }
  const std::vector<AnfNodePtr> &spy_cnodes() const { return spy_cnodes_; }

 private:
  // This is a CNode that does not belong to this area.
  bool IsExternalCNode(const AnfNodePtr &node) { return node->isa<CNode>() && this->nodes_.count(node) == 0; }

  // nodes in this area
  std::unordered_set<AnfNodePtr> nodes_;
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
    if (area_graph == nullptr) return nullptr;
    if (!area_graph->TopoSort()) {
      MS_LOG(WARNING) << "The groups have a cycle.";
      return nullptr;
    }
    return area_graph;
  }

  // Split the graph to multiple areas, and reconnect the edges between the areas.
  // The output `main_cnodes` is a topo-sorted cnode list in main graph, holding the new sub_func_graphs.
  // The output `cnode_group_id` represents the indices of main_cnodes before topo-sorting.
  void SplitGraph(const FuncGraphPtr &main_func_graph, std::vector<CNodePtr> *main_cnodes,
                  std::vector<size_t> *cnode_group_id, std::function<void(const Area &)> expand_callback) {
    main_cnodes->clear();
    main_cnodes->resize(areas_.size(), nullptr);

    for (auto &area : this->areas_) {
      expand_callback(area);
    }

    for (auto index : topo_order_) {
      auto &current_area = areas_[index];
      auto sub_func_graph = std::make_shared<FuncGraph>();
      std::unordered_map<ParameterPtr, AnfNodePtr> param_node_map;

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
      areas_.emplace_back(node_groups[i]);
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
          if (!in_node->isa<CNode>()) continue;
          // area edge u -> v
          size_t u = node_area_map_[in_node];
          if (u == v) continue;
          areas_[u].AddTraitor(in_node);
          if (std::find(edge_prev_[v].begin(), edge_prev_[v].end(), u) == edge_prev_[v].end()) {
            edge_prev_[v].push_back(u);
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
      if (out_degree[i] == 0) que.push(i);
    }
    while (!que.empty()) {
      size_t u = que.front();
      que.pop();
      topo_order_.push_back(u);
      for (size_t i : edge_prev_[u]) {
        if (--out_degree[i] == 0) que.push(i);
      }
    }
    std::reverse(topo_order_.begin(), topo_order_.end());
    return topo_order_.size() == areas_.size();
  }

  // Make a CNode in main graph to hold the sub_func_graph.
  CNodePtr CreateMainCNode(const FuncGraphPtr &main_func_graph, const FuncGraphPtr &sub_func_graph,
                           const std::vector<CNodePtr> &main_cnodes,
                           const std::unordered_map<ParameterPtr, AnfNodePtr> &param_node_map) {
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
        getitem_node->set_abstract(main_cnodes[input_area]->abstract());
        main_cnode_inputs.push_back(getitem_node);
      } else {
        main_cnode_inputs.push_back(main_cnodes[input_area]);
      }
    }
    auto new_main_cnode = main_func_graph->NewCNode(main_cnode_inputs);
    new_main_cnode->set_abstract(sub_func_graph->get_return()->abstract());
    return new_main_cnode;
  }

  void SortCNodes(std::vector<CNodePtr> *main_cnodes) {
    std::vector<CNodePtr> main_cnodes_sorted;
    std::transform(topo_order_.begin(), topo_order_.end(), std::back_inserter(main_cnodes_sorted),
                   [main_cnodes](int index) { return main_cnodes->at(index); });
    *main_cnodes = std::move(main_cnodes_sorted);
  }

  // Areas in this subgraph
  std::vector<Area> areas_;
  // Adjacency table of areas
  std::vector<std::vector<size_t>> edge_prev_;
  // Topological order of areas
  std::vector<size_t> topo_order_;
  // Map AnfNode to Area id
  std::unordered_map<AnfNodePtr, size_t> node_area_map_;
  // Map the nodes to their index if there are multiple value in an area
  std::unordered_map<AnfNodePtr, size_t> node_index_in_returned_tuple_;
};

class Splitter {
 public:
  class SplitSchemer {
   public:
    virtual bool Split(const FuncGraphPtr &func_graph) = 0;
    virtual bool NeedInline(size_t group_id) const { return false; }
    const std::vector<AnfNodePtrList> &split_plan() const { return split_plan_; }

   protected:
    std::vector<AnfNodePtrList> split_plan_;
  };
  using SplitSchemerPtr = std::shared_ptr<SplitSchemer>;
  using SplitterPtr = std::shared_ptr<Splitter>;

  bool Split() {
    GenParamMap();
    auto ori_sub_func_graph = AnfAlgo::GetCNodeFuncGraphPtr(old_subgraph_cnode_);
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

  static SplitterPtr MakeSplitter(const CNodePtr &main_cnode, SplitSchemerPtr split_schemer) {
    MS_EXCEPTION_IF_NULL(main_cnode);
    MS_EXCEPTION_IF_NULL(main_cnode->func_graph());
    MS_EXCEPTION_IF_NULL(split_schemer);
    return std::make_shared<Splitter>(main_cnode, split_schemer);
  }

  Splitter(const CNodePtr &main_cnode, SplitSchemerPtr split_schemer)
      : main_func_graph_(main_cnode->func_graph()), old_subgraph_cnode_(main_cnode), split_schemer_(split_schemer) {}
  ~Splitter() = default;

 private:
  void ResetInlinedNodesKernelInfo() {
    for (const auto &node : inlined_nodes_) {
      ResetKernelInfo(node);
    }
  }

  // Maintain new subgraphs in main graph.
  void RebuildGraph(const std::vector<size_t> &cnodes_group_id) {
    BindFuncGraph();
    RecoverParameter();
    ConnectToMainGraph(cnodes_group_id);
    UpdateSubGraphInfo();
    ResetInlinedNodesKernelInfo();
  }

  // Rebind nodes to its new sub_func_graph
  void BindFuncGraph() {
    for (const auto &cnode : new_subgraph_cnodes_) {
      auto sub_func_graph = AnfAlgo::GetCNodeFuncGraphPtr(cnode);
      auto callback = [&sub_func_graph, this](const AnfNodePtr &node) {
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
      auto sub_func_graph = AnfAlgo::GetCNodeFuncGraphPtr(cnode);
      auto callback = [&cnode, &sub_func_graph, this](const AnfNodePtr &node) {
        auto param = node->cast<ParameterPtr>();
        if (param == nullptr) return;
        auto it = this->param_to_main_graph_node_map_.find(param);
        if (it != this->param_to_main_graph_node_map_.end()) {
          cnode->add_input(it->second);
          sub_func_graph->add_parameter(param);
          // Avoid repeating parameters.
          this->param_to_main_graph_node_map_.erase(it);
        }
      };
      TraverseFuncGraph(sub_func_graph, callback);
    }
  }

  CNodePtr InlineSubFuncGraph(const CNodePtr &main_node) {
    auto func_graph = AnfAlgo::GetCNodeFuncGraphPtr(main_node);
    const auto &inputs = main_node->inputs();
    auto output = func_graph->output()->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(output);
    const auto &parameters = func_graph->parameters();
    std::unordered_map<AnfNodePtr, AnfNodePtr> param_input;
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
        if (AnfAlgo::IsRealKernel(node)) {
          inlined_nodes_.push_back(node);
        }
      }
    }
    return output;
  }

  // Set the new sub_func_graph node as input of nodes original main graph.
  void ConnectToMainGraph(const std::vector<size_t> &cnodes_group_id) {
    // For single output kernel, the last area contains the original output node (return node),
    //  to replace old subgraph with new subgraphs, just replace the old CNode with new last CNode.
    // For multiple output kernel, to avoid returning Parameter, the last MakeTuple was distribute to
    //  a new FuncGraph, just inline the last MakeTuple node.
    std::vector<CNodePtr> tmp_subgraph_cnodes;
    std::unordered_map<AnfNodePtr, AnfNodePtr> replace_map;

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
        tmp_subgraph_cnodes.push_back(new_subgraph_cnodes_[i]);
      }
    }
    new_subgraph_cnodes_ = std::move(tmp_subgraph_cnodes);

    TraverseFuncGraph(main_func_graph_, [&replace_map](const AnfNodePtr &node) {
      auto cnode = node->cast<CNodePtr>();
      if (cnode == nullptr) return;
      for (size_t i = 1; i < cnode->inputs().size(); ++i) {
        auto input_node = cnode->input(i);
        auto iter = replace_map.find(input_node);
        if (iter != replace_map.end()) {
          cnode->set_input(i, iter->second);
        }
      }
    });
  }

  void UpdateSubGraphInfo() {
    auto graph_manager = main_func_graph_->manager();
    MS_EXCEPTION_IF_NULL(graph_manager);

    for (auto cnode : new_subgraph_cnodes_) {
      auto sub_func_graph = AnfAlgo::GetCNodeFuncGraphPtr(cnode);
      // add new sub_func_graph to manager
      graph_manager->AddFuncGraph(sub_func_graph);

      // set GraphKernel attr
      auto attr = ExtractGraphKernelName(TopoSort(sub_func_graph->get_return()), "", "split");
      sub_func_graph->set_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, MakeValue(attr));

      // set kernel info
      AnfNodePtrList inputs(cnode->inputs().begin() + 1, cnode->inputs().end());
      AnfNodePtrList outputs;
      kernel::GetFuncGraphOutputNodes(sub_func_graph, &outputs);
      SetNewKernelInfo(cnode, sub_func_graph, inputs, outputs, AnfAlgo::GetProcessor(old_subgraph_cnode_));
    }
  }

  // Copy all Parameter and ValueNode that the area used.
  void AreaExpand(const Area &area) {
    std::unordered_map<AnfNodePtr, AnfNodePtr> old_valuenode_and_param_map;
    for (auto sub_node : area.nodes()) {
      auto sub_cnode = sub_node->cast<CNodePtr>();
      if (sub_cnode == nullptr) continue;
      for (size_t i = 1; i < sub_cnode->inputs().size(); ++i) {
        auto in_node = sub_cnode->input(i);
        if (in_node->isa<CNode>()) continue;
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
    auto sub_func_graph = AnfAlgo::GetCNodeFuncGraphPtr(old_subgraph_cnode_);
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
    param_to_main_graph_node_map_[param_c] = param_to_main_graph_node_map_[param];
    return param_c;
  }

  FuncGraphPtr main_func_graph_;
  CNodePtr old_subgraph_cnode_;                // The cnode that holds the original sub_func_graph
  std::vector<CNodePtr> new_subgraph_cnodes_;  // The cnode list that hold the new sub_func_graph
  std::vector<AnfNodePtr> inlined_nodes_;
  SplitSchemerPtr split_schemer_;
  std::unordered_map<ParameterPtr, AnfNodePtr> param_to_main_graph_node_map_;
};

class CostModelSplitSchemer : public Splitter::SplitSchemer {
 public:
  bool Split(const FuncGraphPtr &func_graph) override {
    if (!func_graph->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL)) {
      MS_EXCEPTION(NotSupportError) << "func_graph must be a GraphKernel node.";
    }
    func_graph_ = func_graph;
    this->Run();
    return !split_plan_.empty();
  }

  bool NeedInline(size_t group_id) const override {
    if (group_id >= need_inline_.size()) {
      MS_LOG(EXCEPTION) << "The group_id " << group_id << " should be less than the group num " << need_inline_.size();
    }
    return need_inline_[group_id] != 0;
  }

 protected:
  virtual bool SplitByCostModel() {
    // Use an address map to record the anf node address when converting to json,
    // it will recover the original node after split.
    std::map<std::string, AnfNodePtr> address_node_map;

    // convert anf-ir to json
    nlohmann::json json_desc;
    DumpOption dump_option;
    dump_option.is_before_select_kernel = false;
    dump_option.save_ptr_address = true;
    if (!AnfToJsonDesc(topo_valid_nodes_, dump_option, &json_desc, &address_node_map)) {
      MS_LOG(ERROR) << "Collect json desc failed.";
      return false;
    }

    // call costmodel split function.
    auto json_desc_str = json_desc.dump();
    MS_LOG(DEBUG) << "CallPyFn: [" << kGraphKernelSplitFunc << "] with input json:\n" << json_desc_str;
    auto ret = parse::python_adapter::CallPyFn(kGraphKernelModule, kGraphKernelSplitFunc, json_desc_str);
    if (py::isinstance<py::none>(ret)) {
      MS_LOG(ERROR) << "CallPyFn: [" << kGraphKernelSplitFunc << "] return invalid result. input json:\n"
                    << json_desc_str;
      return false;
    }
    std::string split_graphs_str = py::cast<std::string>(ret);
    if (split_graphs_str.empty()) {
      MS_LOG(ERROR) << "CallPyFn: [" << kGraphKernelSplitFunc << "] return invalid result. input json:\n"
                    << json_desc_str;
      return false;
    }

    if (!DecodeJson(split_graphs_str, address_node_map)) {
      MS_LOG(ERROR) << "Failed to decode split graphs. input json:\n" << split_graphs_str;
      return false;
    }
    return true;
  }

  virtual bool DecodeJson(const std::string &json_desc, const std::map<std::string, AnfNodePtr> &address_node_map) {
    auto kernel_json = nlohmann::json::parse(json_desc);
    kernel::AkgKernelJsonDecoder akg_kernel_json_decoder;
    std::vector<nlohmann::json> graph_descs = kernel_json[kJsonKeyGraphDesc];
    std::vector<std::string> graph_modes = kernel_json[kJsonKeyGraphMode];
    if (graph_modes.size() != graph_descs.size()) {
      MS_LOG(ERROR) << "Size of graph_mode " << graph_modes.size() << " mismatch graph_desc " << graph_descs.size();
      return false;
    }

    // recover json to anfnode.
    split_plan_.clear();
    for (const auto &graph_desc : graph_descs) {
      AnfNodePtrList res_graph;
      if (!akg_kernel_json_decoder.DecodeSplitNodes(graph_desc, address_node_map, &res_graph)) {
        MS_LOG(ERROR) << "Failed decode sub graph, " << graph_desc;
        return false;
      }
      split_plan_.push_back(std::move(res_graph));
    }

    // ops to be inlined.
    need_inline_.clear();
    std::transform(graph_modes.begin(), graph_modes.end(), std::back_inserter(need_inline_),
                   [](const std::string &mode) { return mode == "basic" ? 1 : 0; });
    return true;
  }

  virtual void Run() {
    auto mng = func_graph_->manager();
    if (mng == nullptr) {
      mng = Manage(func_graph_, true);
      func_graph_->set_manager(mng);
    }
    GetValidKernelNodes();
    // call CostModel to get a split plan.
    if (!SplitByCostModel() || split_plan_.size() != need_inline_.size() || split_plan_.empty()) {
      split_plan_.clear();
      need_inline_.clear();
      return;
    } else if (split_plan_.size() == 1 && !NeedInline(0)) {
      /*In this case, the CostModel decided to keep the whole graph unchanged.*/
      split_plan_.clear();
      need_inline_.clear();
      return;
    } else {
      MS_LOG(INFO) << "CostModel split success. The kernel is split to " << split_plan_.size() << " parts.";
    }
    MapNodeGroup();
    GroupReturnNode();
    GroupVirtualNodes();
  }

  virtual bool IsValidKernelNode(const AnfNodePtr &node) const {
    if (!node->isa<CNode>()) return false;
    if (AnfAlgo::IsRealKernel(node)) return true;
    return false;
  }

  virtual void GetValidKernelNodes() {
    topo_all_nodes_ = TopoSort(func_graph_->get_return());
    topo_valid_nodes_.clear();
    std::copy_if(topo_all_nodes_.begin(), topo_all_nodes_.end(), std::back_inserter(topo_valid_nodes_),
                 [this](const AnfNodePtr &node) { return IsValidKernelNode(node); });
  }

  void MapNodeGroup() {
    node_group_.clear();
    for (size_t i = 0; i < split_plan_.size(); ++i) {
      for (const auto &node : split_plan_[i]) {
        node_group_[node] = i;
      }
    }
  }

  // group the return node and last MakeTuple node (if exists).
  virtual void GroupReturnNode() {
    AnfNodePtrList outputs;
    kernel::GetFuncGraphOutputNodes(func_graph_, &outputs);
    auto ret_node = func_graph_->get_return();
    auto output = func_graph_->output();
    MS_EXCEPTION_IF_NULL(output);

    if (IsValidKernelNode(output)) {
      auto group_id = node_group_[ret_node] = node_group_[output];
      split_plan_[group_id].push_back(ret_node);
      return;
    }
    // assign the make_tuple node to a new group.
    if (AnfAlgo::CheckPrimitiveType(output, prim::kPrimMakeTuple)) {
      auto group_id = split_plan_.size();
      split_plan_.push_back({output, ret_node});
      need_inline_.push_back(1);
      node_group_[ret_node] = node_group_[output] = group_id;
      return;
    }
  }

  // assign virtual node to the same group of its input.
  virtual void GroupVirtualNodes() {
    for (const auto &node : topo_all_nodes_) {
      if (node_group_.count(node)) continue;
      auto cnode = node->cast<CNodePtr>();
      if (cnode == nullptr) continue;
      bool found = false;
      for (const auto &input : cnode->inputs()) {
        auto iter = node_group_.find(input);
        if (iter != node_group_.end()) {
          node_group_[node] = iter->second;
          split_plan_[iter->second].push_back(node);
          found = true;
          break;
        }
      }
      if (!found) {
        MS_LOG(WARNING) << cnode->fullname_with_scope() << " is ungrouped.";
      }
    }
  }

  std::shared_ptr<FuncGraph> func_graph_;
  AnfNodePtrList topo_all_nodes_;
  AnfNodePtrList topo_valid_nodes_;
  std::unordered_map<AnfNodePtr, size_t> node_group_;
  std::vector<int> need_inline_;
};

bool TrySplit(const CNodePtr &sub_root_cnode) {
  MS_LOG(INFO) << "Split process node: " << sub_root_cnode->fullname_with_scope();
  auto splitter = Splitter::MakeSplitter(sub_root_cnode, std::make_shared<CostModelSplitSchemer>());
  MS_EXCEPTION_IF_NULL(splitter);
  bool result = splitter->Split();
  MS_LOG(INFO) << "Split node completed, result: " << result;
  return result;
}
}  // namespace

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
    if (node != nullptr && AnfAlgo::IsGraphKernel(node)) {
      changed = TrySplit(node) || changed;
    }
  }
  mng->RemoveRoots();
  mng->KeepRoots({func_graph});
  return changed;
}
}  // namespace opt
}  // namespace mindspore
