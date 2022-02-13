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
#include "common/graph_kernel/adapter/graph_kernel_splitter_with_py.h"

#include <algorithm>
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <map>
#include "utils/hash_map.h"
#include "utils/ms_context.h"
#include "pipeline/jit/parse/python_adapter.h"
#include "kernel/akg/akg_kernel_json_generator.h"
#include "kernel/common_utils.h"
#include "common/graph_kernel/graph_kernel_helper.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "utils/context/graph_kernel_flags.h"

namespace mindspore::graphkernel {
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
    auto flags_str = CollectSplitFlags();
    MS_LOG(DEBUG) << "CallPyFn: [" << kGraphKernelSplitFunc << "] with input json: " << json_desc_str
                  << ". flag: " << flags_str;
    auto ret = parse::python_adapter::CallPyFn(kGraphKernelModule, kGraphKernelSplitFunc, json_desc_str, flags_str);
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
      if (!SplitNodesDecoder::DecodeSplitNodes(graph_desc, address_node_map, &res_graph)) {
        MS_LOG(ERROR) << "Failed decode sub graph, " << graph_desc;
        return false;
      }
      split_plan_.emplace_back(std::move(res_graph));
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
    if (!node->isa<CNode>()) return false;
    if (AnfUtils::IsRealKernel(node)) return true;
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
      auto group_id = node_group_[output];
      node_group_[ret_node] = group_id;
      split_plan_[group_id].emplace_back(ret_node);
      return;
    }
    // assign the make_tuple node to a new group.
    if (AnfAlgo::CheckPrimitiveType(output, prim::kPrimMakeTuple)) {
      auto group_id = split_plan_.size();
      split_plan_.emplace_back(AnfNodePtrList{output, ret_node});
      need_inline_.emplace_back(1);
      node_group_[output] = group_id;
      node_group_[ret_node] = group_id;
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

  virtual std::string CollectSplitFlags() {
    const auto &flags = GraphKernelFlags::GetInstance();
    nlohmann::json flag_json;
    flag_json["dump_as_text"] = flags.dump_as_text;
    flag_json["enable_stitch_fusion"] = flags.enable_stitch_fusion;
    flag_json["enable_recompute_fusion"] = flags.enable_recompute_fusion;
    return flag_json.dump();
  }

  std::shared_ptr<FuncGraph> func_graph_;
  AnfNodePtrList topo_all_nodes_;
  AnfNodePtrList topo_valid_nodes_;
  mindspore::HashMap<AnfNodePtr, size_t> node_group_;
  std::vector<int> need_inline_;
};

std::shared_ptr<SplitSchemer> GraphKernelSplitterWithPy::GetSplitSchema(const std::string &processor) {
  // default use c++ split model for CPU target.
  if (processor != kCPUDevice || common::GetEnv("MS_DEV_GRAPH_KERNEL_PY_SPLIT_MODEL") == "on") {
    MS_LOG(DEBUG) << "use py split model";
    return std::make_shared<CostModelSplitSchemer>();
  } else {
    MS_LOG(DEBUG) << "use c++ split model";
    return GraphKernelSplitter::GetSplitSchema(processor);
  }
}
}  // namespace mindspore::graphkernel
