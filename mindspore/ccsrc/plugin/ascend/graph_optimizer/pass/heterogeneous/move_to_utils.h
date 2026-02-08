/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_HETEROGENEOUS_MOVE_TO_UTILS_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_HETEROGENEOUS_MOVE_TO_UTILS_
#include "include/backend/common/kernel_graph/kernel_graph.h"

namespace mindspore {
namespace opt {
struct MoveToInfo {
  // Specify where the data should be moved to.
  const char *to_;
  // Data source node.
  AnfNodePtr data_previous_node_;
  // Data user node.
  CNodePtr data_following_node_;
  // Input index of data user node.
  size_t input_index_;
  // After which the MoveTo kernel should be launched, can be nullptr.
  CNodePtr control_previous_node_;
  // Before which the MoveTo kernel should be launched, can be nullptr.
  CNodePtr control_following_node_;
};

struct MoveAssignInfo {
  // Specify where the data should be moved to.
  const char *to_;
  // The parameter that be assigned.
  ParameterPtr parameter_;
  // The value assigned to parameter.
  AnfNodePtr value_;
  // After which the MoveAssing kernel should be launched, can be nullptr.
  CNodePtr control_previous_node_;
  // Before which the MoveTo kernel should be launched, can be nullptr.
  CNodePtr control_following_node_;
};

class MoveToUtils {
 public:
  static CNodePtr InsertMoveTo(const KernelGraphPtr &kernel_graph, const MoveToInfo &info);
  static CNodePtr InsertMoveAssign(const KernelGraphPtr &kernel_graph, const MoveAssignInfo &info);

  static CNodePtr InsertDependNode(const KernelGraphPtr &kernel_graph, const CNodePtr &pre_node,
                                   const CNodePtr &post_node);
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_HETEROGENEOUS_MOVE_TO_UTILS_
