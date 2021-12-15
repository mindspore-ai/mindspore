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

#include "minddata/dataset/engine/opt/pre/node_offload_pass.h"
#include "minddata/dataset/engine/ir/datasetops/map_node.h"
#include "minddata/dataset/engine/ir/datasetops/batch_node.h"

namespace mindspore {
namespace dataset {
NodeOffloadPass::OffloadNodes::OffloadNodes() : prev_map_offloaded_(true) {}

// Perform MapNode offload check.
Status NodeOffloadPass::OffloadNodes::Visit(std::shared_ptr<MapNode> node, bool *const modified) {
  *modified = false;
  // Check if this node is set to offload and add to nodes_to_offload_.
  if (node->GetOffload() == true) {
    if (IS_OUTPUT_ON(mindspore::INFO)) {
      std::string operations = "operations=[";
      auto op_list = node->operations();
      size_t op_size = op_list.size();
      for (int i = 0; i < op_size; i++) {
        operations += op_list[i]->Name();
        if (i < op_size - 1) {
          operations += std::string(", ");
        }
      }
      operations += "]";
      MS_LOG(INFO) << "The offload of map(" + operations + ") is true, and heterogeneous acceleration will be enabled.";
    }
    if (prev_map_offloaded_) {
      nodes_to_offload_.push_back(std::static_pointer_cast<DatasetNode>(node));
    } else {
      MS_LOG(WARNING) << "Invalid use of offload in map, ignoring offload flag. Ops will be run in CPU pipeline";
      node->SetOffload(false);
      *modified = true;
    }
  } else {
    // Since map nodes are visited in reverse order, no other map ops can be offloaded after this.
    prev_map_offloaded_ = false;
  }
  return Status::OK();
}

// constructor
NodeOffloadPass::NodeOffloadPass() {}

// Walk the tree to collect the nodes to offload, fill the offload_json object, then remove the node.
Status NodeOffloadPass::RunOnTree(std::shared_ptr<DatasetNode> root_ir, bool *const modified) {
  MS_LOG(INFO) << "Pre pass: node offload pass started.";
  // Create the offload node pass which can identify which nodes need to be offloaded.
  std::unique_ptr<NodeOffloadPass::OffloadNodes> offload_nodes = std::make_unique<NodeOffloadPass::OffloadNodes>();
  RETURN_IF_NOT_OK(offload_nodes->Run(root_ir, modified));

  // Update modified flag if there were any nodes identified to be offloaded
  if (offload_nodes->nodes_to_offload().empty() == false) {
    *modified = true;
  }

  // Then, execute the offloading of any nodes that were set up to be offloaded
  for (auto node : offload_nodes->nodes_to_offload()) {
    RETURN_IF_NOT_OK(node->to_json(&offload_json_));
    offload_json_["op_type"] = node->Name();

    // Add the single offloaded node to the list of offloaded nodes and remove the node from the ir tree
    offload_json_list_.push_back(offload_json_);
    RETURN_IF_NOT_OK(node->Drop());
  }
  MS_LOG(INFO) << "Pre pass: offload node removal pass complete.";
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
