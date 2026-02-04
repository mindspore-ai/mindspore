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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_ADAPTER_GRAPH_KERNEL_SPLITTER_WITH_PY_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_ADAPTER_GRAPH_KERNEL_SPLITTER_WITH_PY_H_
#include <memory>
#include <string>
#include <map>
#include <nlohmann/json.hpp>
#include "backend/ms_backend/graph_fusion/core/graph_kernel_splitter.h"
#include "ir/func_graph_flag.h"

namespace mindspore::graphkernel {
class SplitByJsonSchemer : public SplitSchemer {
 public:
  SplitByJsonSchemer() = default;
  SplitByJsonSchemer(const std::map<std::string, AnfNodePtr> &address_node_map, const std::string &json_desc_str)
      : address_node_map_(address_node_map), json_desc_str_(json_desc_str) {}
  virtual ~SplitByJsonSchemer() = default;
  bool Split(const FuncGraphPtr &func_graph) override {
    if (!func_graph->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL)) {
      MS_EXCEPTION(NotSupportError) << "func_graph must be a GraphKernel node.";
    }
    func_graph_ = func_graph;
    this->Run();
    return !split_plan_.empty();
  }

 protected:
  virtual bool SplitByCostModel() { return SplitByJsonStr(address_node_map_, json_desc_str_); }
  virtual bool SplitByJsonStr(const std::map<std::string, AnfNodePtr> &address_node_map, std::string split_graphs_str);
  void RemoveHangingNodes();
  virtual bool DecodeJson(const std::string &json_desc, const std::map<std::string, AnfNodePtr> &address_node_map);
  virtual void Run();
  virtual bool IsValidKernelNode(const AnfNodePtr &node) const;
  virtual void GetValidKernelNodes();
  void MapNodeGroup();

  // group the return node and last MakeTuple node (if exists).
  virtual void GroupReturnNode();

  // assign virtual node to the same group of its input.
  virtual void GroupVirtualNodes();

  std::shared_ptr<FuncGraph> func_graph_;
  AnfNodePtrList topo_all_nodes_;
  AnfNodePtrList topo_valid_nodes_;
  mindspore::HashMap<AnfNodePtr, size_t> node_group_;
  std::map<std::string, AnfNodePtr> address_node_map_;
  std::string json_desc_str_;
};

class CostModelSplitSchemer : public SplitByJsonSchemer {
 public:
  CostModelSplitSchemer() = default;
  virtual ~CostModelSplitSchemer() = default;

 protected:
  bool SplitByCostModel() override;
};

class GraphKernelSplitterWithPy : public GraphKernelSplitter {
 public:
  explicit GraphKernelSplitterWithPy(bool is_dynamic = false)
      : GraphKernelSplitter(is_dynamic ? "dyn_graph_kernel_splitter" : "graph_kernel_splitter"),
        is_dynamic_(is_dynamic) {}
  ~GraphKernelSplitterWithPy() = default;
  std::shared_ptr<SplitSchemer> GetSplitSchema(const std::string &processor) override;

 private:
  bool is_dynamic_;
};
using GraphKernelSplitterWithPyPtr = std::shared_ptr<GraphKernelSplitterWithPy>;
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_ADAPTER_GRAPH_KERNEL_SPLITTER_WITH_PY_H_
