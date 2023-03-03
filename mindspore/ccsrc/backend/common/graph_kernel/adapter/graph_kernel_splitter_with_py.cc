/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "backend/common/graph_kernel/adapter/graph_kernel_splitter_with_py.h"

#include <algorithm>
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <map>
#include <set>
#include <nlohmann/json.hpp>
#include "utils/hash_map.h"
#include "utils/ms_context.h"
#include "include/common/utils/python_adapter.h"
#include "kernel/akg/akg_kernel_json_generator.h"
#include "kernel/common_utils.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "backend/common/graph_kernel/graph_kernel_helper.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"

namespace mindspore::graphkernel {
struct StitchInfo {
  std::vector<std::string> stitch_ops;
  std::vector<std::string> stitch_atomic_ops;
};

class SplitNodesDecoder {
  StitchInfo GetStitchInfo(const nlohmann::json &kernel_json) const {
    StitchInfo info;
    if (kernel_json.find(kJsonKeyBufferStitch) != kernel_json.end()) {
      nlohmann::json buffer_stitch = kernel_json[kJsonKeyBufferStitch];
      if (buffer_stitch.find(kJsonKeyStitchOp) != buffer_stitch.end()) {
        std::vector<std::string> stitch_ops = buffer_stitch[kJsonKeyStitchOp];
        info.stitch_ops = stitch_ops;
      }
      if (buffer_stitch.find(kJsonKeyStitchAtomicOp) != buffer_stitch.end()) {
        std::vector<std::string> stitch_atomic_ops = buffer_stitch[kJsonKeyStitchAtomicOp];
        info.stitch_atomic_ops = stitch_atomic_ops;
      }
    }
    return info;
  }

  std::set<std::string> GetRecomputeOps(const nlohmann::json &kernel_json) const {
    if (kernel_json.find(kJsonKeyRecomputeOps) != kernel_json.end()) {
      std::vector<std::string> recompute_ops = kernel_json[kJsonKeyRecomputeOps];
      return std::set<std::string>(recompute_ops.begin(), recompute_ops.end());
    }
    return std::set<std::string>();
  }

  bool IsRecomputeOp(const nlohmann::json &op_desc, const std::set<std::string> &recompute_ops) const {
    std::vector<nlohmann::json> output_descs = op_desc[kJsonKeyOutputDesc];
    if (output_descs.empty() || output_descs[0].find(kJsonKeyTensorName) == output_descs[0].end()) {
      return false;
    }
    std::string tensor_name = output_descs[0][kJsonKeyTensorName];
    return recompute_ops.count(tensor_name) > 0;
  }

  CNodePtr NewRecomputeNode(const AnfNodePtr &orig_node, std::map<AnfNodePtr, AnfNodePtr> *node_map) const {
    auto func_graph = orig_node->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    auto cnode = orig_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    TraceGuard guard(std::make_shared<TraceOpt>(cnode->debug_info()));
    auto orig_inputs = cnode->inputs();
    std::vector<AnfNodePtr> inputs;
    for (auto inp : orig_inputs) {
      if (node_map->find(inp) == node_map->end()) {
        inputs.push_back(inp);
        continue;
      }
      inputs.push_back((*node_map)[inp]);
    }
    CNodePtr cp_node = func_graph->NewCNode(inputs);
    func_graph->AddNode(cp_node);
    ScopePtr scope = (orig_node->scope() != kDefaultScope) ? orig_node->scope() : kDefaultScope;
    cp_node->set_scope(scope);
    cp_node->CloneCNodeInfo(cnode);
    (*node_map)[orig_node] = cp_node;
    return cp_node->cast<CNodePtr>();
  }

  void SetStitchAttr(const nlohmann::json &op_desc, const StitchInfo &info, const CNodePtr &node) const {
    std::vector<nlohmann::json> output_descs = op_desc[kJsonKeyOutputDesc];
    if (output_descs.empty() || output_descs[0].find(kJsonKeyTensorName) == output_descs[0].end()) {
      return;
    }
    std::string tensor_name = output_descs[0][kJsonKeyTensorName];
    if (std::find(info.stitch_ops.begin(), info.stitch_ops.end(), tensor_name) != info.stitch_ops.end()) {
      AnfUtils::SetNodeAttr(kAttrStitch, MakeValue("common"), node);
      MS_LOG(INFO) << "Enable common stitch fusion by " << node->fullname_with_scope();
    }
    if (std::find(info.stitch_atomic_ops.begin(), info.stitch_atomic_ops.end(), tensor_name) !=
        info.stitch_atomic_ops.end()) {
      AnfUtils::SetNodeAttr(kAttrStitch, MakeValue("atomic"), node);
      MS_LOG(INFO) << "Enable atomic add stitch fusion by " << node->fullname_with_scope();
    }
  }

  // replace original region root op by its copy in this res_graphs
  void ConnectRecomputeOps(AnfNodePtrList *res_graphs, const AnfNodePtr &orig_region_root,
                           const AnfNodePtr &cp_region_root) const {
    for (auto &node : *res_graphs) {
      auto cnode = node->cast<CNodePtr>();
      auto inputs = cnode->inputs();
      for (size_t i = 1; i < inputs.size(); ++i) {
        if (inputs[i] != orig_region_root) {
          continue;
        }
        cnode->set_input(i, cp_region_root);
      }
    }
  }

 public:
  bool DecodeSplitNodes(const nlohmann::json &kernel_json, const std::map<std::string, AnfNodePtr> &address_node_map,
                        AnfNodePtrList *res_graphs) const {
    MS_EXCEPTION_IF_NULL(res_graphs);
    MS_LOG(DEBUG) << "start decode, " << kernel_json;
    // decode cnodes in graph.
    std::vector<nlohmann::json> op_node_descs = kernel_json[kJsonKeyOpDesc];
    if (op_node_descs.empty()) {
      MS_LOG(ERROR) << "Error decode, no cnodes for graph: " << kernel_json;
      return false;
    }
    StitchInfo info = GetStitchInfo(kernel_json);
    auto recompute_ops = GetRecomputeOps(kernel_json);
    // key_value: original_copied
    std::map<AnfNodePtr, AnfNodePtr> node_map;
    // nodes would be copied
    AnfNodePtrList orig_region_nodes;
    // nodes would not be copied
    AnfNodePtrList no_cp_nodes;
    for (const auto &op_desc : op_node_descs) {
      if (op_desc.find(kJsonKeyPtrAddress) == op_desc.end() || op_desc[kJsonKeyPtrAddress].is_null()) {
        MS_LOG(ERROR) << "Decode failed, key: " << kJsonKeyPtrAddress << " not found in: " << op_desc;
        return false;
      }

      std::string ptr_address = op_desc[kJsonKeyPtrAddress];
      if (address_node_map.count(ptr_address) == 0) {
        MS_LOG(ERROR) << "Decode failed, ptr_address not found in map.";
        return false;
      }
      auto node = address_node_map.at(ptr_address)->cast<CNodePtr>();
      if (IsRecomputeOp(op_desc, recompute_ops)) {
        auto cp_node = NewRecomputeNode(node, &node_map);
        orig_region_nodes.push_back(node);
        SetStitchAttr(op_desc, info, cp_node);
        res_graphs->push_back(cp_node);
        continue;
      }
      SetStitchAttr(op_desc, info, node);
      res_graphs->push_back(node);
      no_cp_nodes.push_back(node);
    }
    for (auto orig_node : orig_region_nodes) {
      ConnectRecomputeOps(&no_cp_nodes, orig_node, node_map[orig_node]);
    }
    MS_LOG(DEBUG) << "decode cnodes success, size: " << res_graphs->size();
    return true;
  }
};

class CostModelSplitSchemer : public SplitSchemer {
 public:
  virtual ~CostModelSplitSchemer() = default;
  bool Split(const FuncGraphPtr &func_graph) override {
    if (!func_graph->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL)) {
      MS_EXCEPTION(NotSupportError) << "func_graph must be a GraphKernel node.";
    }
    func_graph_ = func_graph;
    this->Run();
    return !split_plan_.empty();
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
    // set the "node_name" for tracing split result.
    std::string node_name = json_desc["op"];
    func_graph_->set_attr(kAttrNodeName, MakeValue(node_name));

    // call costmodel split function.
    auto json_desc_str = json_desc.dump();
    auto flags_str = GraphKernelFlags::GetInstance().DumpAllFlags();
    MS_LOG(DEBUG) << "CallPyFn: [" << kGraphKernelSplitFunc << "] with input json: " << json_desc_str
                  << ". flag: " << flags_str;
    auto ret = python_adapter::CallPyFn(kGraphKernelModule, kGraphKernelSplitFunc, json_desc_str, flags_str);
    if (py::isinstance<py::none>(ret)) {
      MS_LOG(ERROR) << "CallPyFn: [" << kGraphKernelSplitFunc << "] return invalid result. input json:\n"
                    << json_desc_str << ". flag: " << flags_str;
      return false;
    }
    std::string split_graphs_str = py::cast<std::string>(ret);
    if (split_graphs_str.empty()) {
      MS_LOG(ERROR) << "CallPyFn: [" << kGraphKernelSplitFunc << "] return invalid result. input json:\n"
                    << json_desc_str << ". flag: " << flags_str;
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
      if (!SplitNodesDecoder().DecodeSplitNodes(graph_desc, address_node_map, &res_graph)) {
        MS_LOG(ERROR) << "Failed decode sub graph, " << graph_desc;
        return false;
      }
      (void)split_plan_.emplace_back(std::move(res_graph));
    }

    // ops to be inlined.
    need_inline_.clear();
    (void)std::transform(graph_modes.begin(), graph_modes.end(), std::back_inserter(need_inline_),
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
      // In this case, the CostModel decided to keep the whole graph unchanged.
      split_plan_.clear();
      need_inline_.clear();
      return;
    } else {
      MS_LOG(DEBUG) << "CostModel split succeeded. The kernel is split to " << split_plan_.size() << " parts.";
    }
    MapNodeGroup();
    GroupReturnNode();
    GroupVirtualNodes();
  }

  virtual bool IsValidKernelNode(const AnfNodePtr &node) const {
    if (!node->isa<CNode>()) {
      return false;
    }
    if (AnfUtils::IsRealKernel(node)) {
      return true;
    }
    return false;
  }

  virtual void GetValidKernelNodes() {
    topo_all_nodes_ = TopoSort(func_graph_->get_return());
    topo_valid_nodes_.clear();
    (void)std::copy_if(topo_all_nodes_.begin(), topo_all_nodes_.end(), std::back_inserter(topo_valid_nodes_),
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
      auto group_id = node_group_[output];
      node_group_[ret_node] = group_id;
      (void)split_plan_[group_id].emplace_back(ret_node);
      return;
    }
    // assign the make_tuple node to a new group.
    if (common::AnfAlgo::CheckPrimitiveType(output, prim::kPrimMakeTuple)) {
      auto group_id = split_plan_.size();
      (void)split_plan_.emplace_back(AnfNodePtrList{output, ret_node});
      (void)need_inline_.emplace_back(1);
      node_group_[output] = group_id;
      node_group_[ret_node] = group_id;
      return;
    }
  }

  // assign virtual node to the same group of its input.
  virtual void GroupVirtualNodes() {
    for (const auto &node : topo_all_nodes_) {
      if (node_group_.count(node) != 0) {
        continue;
      }
      auto cnode = node->cast<CNodePtr>();
      if (cnode == nullptr) {
        continue;
      }
      bool found = false;
      for (const auto &input : cnode->inputs()) {
        auto iter = node_group_.find(input);
        if (iter != node_group_.end()) {
          auto group_id = iter->second;
          node_group_[node] = group_id;
          (void)split_plan_[group_id].emplace_back(node);
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
  mindspore::HashMap<AnfNodePtr, size_t> node_group_;
};

std::shared_ptr<SplitSchemer> GraphKernelSplitterWithPy::GetSplitSchema(const std::string &processor) {
  if (processor != kCPUDevice && processor != kAscendDevice) {
    MS_LOG(DEBUG) << "use py split model";
    return std::make_shared<CostModelSplitSchemer>();
  } else {
    MS_LOG(DEBUG) << "use c++ split model";
    return GraphKernelSplitter::GetSplitSchema(processor);
  }
}
}  // namespace mindspore::graphkernel
