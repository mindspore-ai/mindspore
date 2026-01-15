/**
 * Copyright 2025-2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_UTILS_HYPER_OFFLOAD_HYPER_OFFLOAD_OPERATION_H_
#define MINDSPORE_CCSRC_UTILS_HYPER_OFFLOAD_HYPER_OFFLOAD_OPERATION_H_

#include <vector>
#include <optional>
#include <algorithm>

#include "ir/anf.h"

namespace mindspore {
namespace utils {
namespace hyper_offload {

struct NodeOperations {
  CNodePtrList operations;
};

class HyperOffloadOperations {
 public:
  // Initialize the operations list with a given size.
  void Init(size_t size);

  // Add an operation to be executed before the main execution sequence.
  void AddPreOperation(const CNodePtr &node);

  // Add an operation to be executed after the operation at the specified index.
  void AddOperationAfter(size_t index, const CNodePtr &node);

  // Add an operation to be executed before the operation at the specified index.
  void AddOperationBefore(size_t index, const CNodePtr &node);

  // Merge operations from another HyperOffloadOperations object.
  void MergeFrom(const HyperOffloadOperations &other);

  // Move a specific node ahead by a certain distance.
  void MoveAhead(const CNodePtr &node, size_t prefetch_distance);

  // Get the operations to be executed before the main sequence.
  const NodeOperations &GetPreOperations() const;

  // Get the operations corresponding to the main execution sequence.
  const std::vector<NodeOperations> &GetOperations() const;

  // Get all H2D nodes in the operations.
  CNodePtrList GetAllH2DNodes() const;

 private:
  // Operations corresponding to each position in the original execution sequence
  std::vector<NodeOperations> operations_;

  // Operations to be inserted before the entire execution sequence
  NodeOperations pre_operations_;

  struct AttachPosition {
    bool is_pre = true;
    // Valid only when is_pre=false, indicates attached after original_order[index]
    size_t index = 0;
  };

  // Remove a node from the operations.
  void RemoveNode(const CNodePtr &node);

  // Get the attachment position of a node.
  AttachPosition GetAttachPosition(const CNodePtr &node) const;
};

}  // namespace hyper_offload
}  // namespace utils
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_HYPER_OFFLOAD_HYPER_OFFLOAD_OPERATION_H_
