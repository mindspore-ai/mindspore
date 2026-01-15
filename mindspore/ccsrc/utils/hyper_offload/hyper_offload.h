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

#ifndef MINDSPORE_CCSRC_UTILS_HYPER_OFFLOAD_HYPER_OFFLOAD_H_
#define MINDSPORE_CCSRC_UTILS_HYPER_OFFLOAD_HYPER_OFFLOAD_H_

#include <vector>

#include "ir/anf.h"
#include "include/utils/visible.h"
#include "utils/hyper_offload/hyper_offload_strategy.h"
#include "utils/hyper_offload/hyper_offload_operation.h"
#include "utils/hyper_offload/hyper_offload_utils.h"

namespace mindspore {
namespace utils {
namespace hyper_offload {
// Event insertion hint: describes logical position "insert after a node"
struct EventPlan {
  CNodePtr send_insert_after;
  CNodePtr recv_insert_after;
};

struct HyperOffloadInput {
  FuncGraphPtr graph;
  CNodePtrList exec_order;
  bool enable_offload_parameter = false;
  bool enable_offload_activation = false;
  HandleNewNodeCallback callback;
};

struct HyperOffloadPlan {
  FuncGraphPtr graph;
  CNodePtrList exec_order;
  HyperOffloadOperations hyper_offload_operations;
  std::vector<EventPlan> event_plan;
};

class COMMON_EXPORT HyperOffloadOptimizer {
 public:
  HyperOffloadOptimizer() = default;
  ~HyperOffloadOptimizer() = default;

  // Run the hyper offload optimizer.
  static HyperOffloadPlan Run(const HyperOffloadInput &input);
  // Generate the event plan for synchronization.
  static std::vector<EventPlan> GenerateEventPlan(const FuncGraphPtr &graph, const CNodePtrList &exec_order);

 private:
  // Generate offload operations for parameters.
  static HyperOffloadOperations GenerateParameterOperations(const FuncGraphPtr &graph, const CNodePtrList &exec_order,
                                                            const HandleNewNodeCallback &callback);
  // Generate offload operations for activations.
  static HyperOffloadOperations GenerateActivationOperations(const FuncGraphPtr &graph, const CNodePtrList &exec_order,
                                                             const HandleNewNodeCallback &callback);
  // Build the new execution order with offload operations inserted.
  static CNodePtrList BuildExecutionOrder(const CNodePtrList &execution_order,
                                          const HyperOffloadOperations &hyper_offload_operations);
  // Adjust the position of hyper offload nodes.
  static void AdjustHyperOffloadNodePosition(HyperOffloadOperations *ops);
};
}  // namespace hyper_offload
}  // namespace utils
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_HYPER_OFFLOAD_HYPER_OFFLOAD_H_
