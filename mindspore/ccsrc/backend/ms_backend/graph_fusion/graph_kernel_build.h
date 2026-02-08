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
#ifndef MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_GRAPH_KERNEL_BUILD_H_
#define MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_GRAPH_KERNEL_BUILD_H_

#include <vector>
#include <string>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <map>
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/framework_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/kash/kernel_pack.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/graph_kernel_builder.h"
#include "backend/ms_backend/graph_fusion/core/graph_kernel_splitter.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_splitter_with_py.h"

namespace mindspore {
namespace graphkernel {
class SafeSplitSchemer : public CommonSplitSchemer {
 public:
  SafeSplitSchemer() = default;
  ~SafeSplitSchemer() = default;
  bool Split(const FuncGraphPtr &func_graph) override;

 protected:
  void Run(const FuncGraphPtr &func_graph);
  void SplitNodes(const FuncGraphPtr &func_graph);
};

class SafeGraphKernelSplitter : public GraphKernelSplitter {
 public:
  SafeGraphKernelSplitter() = default;
  ~SafeGraphKernelSplitter() = default;
  std::shared_ptr<SplitSchemer> GetSplitSchema(const std::string &) override {
    return std::make_shared<SafeSplitSchemer>();
  }
};

class KernelCompilerGraphKernelSplitter : public GraphKernelSplitter {
 public:
  KernelCompilerGraphKernelSplitter() = default;
  ~KernelCompilerGraphKernelSplitter() = default;
  std::shared_ptr<SplitSchemer> GetSplitSchema(const std::string &) override {
    return std::make_shared<SplitByJsonSchemer>(address_node_map_, json_desc_str_);
  }

  void SetAddressNodeMap(const std::map<std::string, AnfNodePtr> &address_node_map) {
    address_node_map_ = address_node_map;
  }

  void SetJson(const std::string &json_desc_str) { json_desc_str_ = json_desc_str; }
  std::map<std::string, AnfNodePtr> address_node_map_;
  std::string json_desc_str_;
};

class GraphKernelBuild : public opt::Pass {
 public:
  GraphKernelBuild() : Pass("graph_kernel_build") {}
  ~GraphKernelBuild() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  void Init();
  bool Process(const FuncGraphPtr &func_graph, int iter);
  kernel::JsonNodePair CollectNode(const AnfNodePtr &node) const;
  // Collect graph kernel nodes in main graph.
  void CollectNodes(const FuncGraphPtr &func_graph, std::vector<kernel::JsonNodePair> *nodes) const;
  // Collect graph kernel nodes that do not have compile cache, which means these nodes need to be compiled.
  std::vector<kernel::JsonNodePair> CollectNotCachedNodes(const std::vector<kernel::JsonNodePair> &nodes);
  // Parallel compiling.
  void ParallelBuild(const std::vector<kernel::JsonNodePair> &nodes);
  // Split nodes that compiled failed.
  bool SplitNodes(const std::vector<kernel::JsonNodePair> &nodes);
  // Split nodes that compiled failed.
  bool SplitNodesByKernelCompiler(const std::vector<kernel::JsonNodePair> &nodes);

  SafeGraphKernelSplitter splitter_;  // used to split nodes that compile failed
  kernel::KernelMeta *bin_map_{nullptr};
  std::shared_ptr<kernel::GraphKernelBuilder> kernel_builder_{nullptr};
  std::unordered_map<std::string, kernel::KernelPackPtr> kernel_pack_;  // compile cache
};
}  // namespace graphkernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_GRAPH_KERNEL_BUILD_H_
