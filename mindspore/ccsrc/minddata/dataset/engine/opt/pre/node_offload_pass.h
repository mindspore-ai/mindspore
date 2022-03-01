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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PRE_NODE_OFFLOAD_PASS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PRE_NODE_OFFLOAD_PASS_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "minddata/dataset/engine/opt/pass.h"

namespace mindspore {
namespace dataset {
class DatasetOp;

/// \class NodeOffloadPass
/// \brief This is a tree pass that will offload nodes.  It uses offload_nodes to first identify which
///     nodes should be offloaded, adds the nodes' namea to the offload list, then removes the nodes from the ir tree.
class NodeOffloadPass : public IRTreePass {
  /// \class OffloadNodes
  /// \brief This is a NodePass whose job is to identify which nodes should be offloaded.
  class OffloadNodes : public IRNodePass {
   public:
    /// \brief Constructor
    OffloadNodes();
    /// \brief Destructor
    ~OffloadNodes() = default;

    /// \brief Perform MapNode offload check
    /// \param[in] node The node being visited
    /// \param[in, out] modified Indicator if the node was changed at all
    /// \return Status The status code returned
    Status Visit(std::shared_ptr<MapNode> node, bool *const modified) override;

    /// \brief Access selected offload nodes for removal.
    /// \return All the nodes to be removed by offload.
    std::vector<std::shared_ptr<DatasetNode>> nodes_to_offload() { return nodes_to_offload_; }

   private:
    /// \brief Vector of nodes to offload
    std::vector<std::shared_ptr<DatasetNode>> nodes_to_offload_;
    /// \brief Vector of supported offload operations
    const std::set<std::string> supported_ops_{
      "HwcToChw",        "Normalize",          "RandomColorAdjust", "RandomHorizontalFlip",
      "RandomSharpness", "RandomVerticalFlip", "Rescale",           "TypeCast"};
    /// \brief std::map indicating if the map op for the input column is at the end of the pipeline
    std::map<std::string, bool> end_of_pipeline_;
    /// \brief bool indicating whether the auto_offload config option is enabled
    bool auto_offload_;
  };

 public:
  /// \brief Constructor
  NodeOffloadPass();

  /// \brief Destructor
  ~NodeOffloadPass() = default;

  /// \brief Runs an offload_nodes pass first to find out which nodes to offload, then offloads them.
  /// \param[in, out] root_ir The tree to operate on.
  /// \param[in, out] modified Indicates if the tree was modified.
  /// \return Status The status code returned
  Status RunOnTree(std::shared_ptr<DatasetNode> root_ir, bool *const modified) override;
  /// \brief Getter
  /// \return JSON of offload
  nlohmann::json GetOffloadJson() { return offload_json_list_; }

 private:
  /// \brief JSON instance containing single offload op.
  nlohmann::json offload_json_;

  /// \brief JSON instance containing all offload ops.
  nlohmann::json offload_json_list_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PRE_NODE_OFFLOAD_PASS_H_
