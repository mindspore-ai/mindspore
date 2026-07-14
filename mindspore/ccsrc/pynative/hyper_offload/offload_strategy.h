/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PYNATIVE_HYPER_OFFLOAD_OFFLOAD_STRATEGY_H_
#define MINDSPORE_CCSRC_PYNATIVE_HYPER_OFFLOAD_OFFLOAD_STRATEGY_H_

#include <vector>
#include <memory>
#include <unordered_map>
#include "ir/anf.h"
#include "utils/hyper_offload/hyper_offload.h"
#include "utils/hyper_offload/hyper_offload_operation.h"

namespace mindspore {
namespace pynative {
namespace hyper_offload {

// Actions to perform on a specific tensor
enum class TensorAction {
  kNone,
  kD2HWithEvent,         // Start D2H copy with event synchronization
  kH2DWithEvent,         // Start H2D copy with event synchronization
  kWaitD2HAndSwitch,     // Wait D2H complete and switch address to host
  kWaitH2DAndSwitch,     // Wait H2D complete and switch address to device
  kWaitH2DAndFreeCache,  // Wait H2D complete and free cache
  kFreeDevice,           // Only OffloadAsync (skip d2h when already offloaded, just free device memory)
};

// Tensor source: from op input or output
enum class TensorSource {
  kInput,  // From op inputs
  kOutput  // From op output
};

// Single operation entry describing which tensor and what action
// The tensor is located by (source, index) from current op's inputs/output
struct TensorOpEntry {
  size_t tensor_id{0};  // Tensor identifier for executor cache
  TensorAction action{TensorAction::kNone};
  TensorSource source{TensorSource::kOutput};
  size_t index{0};  // Input index or output index
};

// Converted strategy from HyperOffloadPlan
// Core: exec_id -> vector<TensorOpEntry>
class OffloadStrategy {
 public:
  OffloadStrategy() = default;
  ~OffloadStrategy() = default;

  // Convert HyperOffloadPlan to executable strategy
  void ConvertFromPlan(const utils::hyper_offload::HyperOffloadPlan &plan);

  // Get pre-execution operations for a specific execution position (called before op runs)
  // @param exec_id: Execution position (0-based index in execution order)
  // @return: List of tensor operations at this position, nullptr if none
  const std::vector<TensorOpEntry> *GetPreOpsAtPosition(size_t exec_id) const;

  // Get post-execution operations for a specific execution position (called after op runs)
  // @param exec_id: Execution position (0-based index in execution order)
  // @return: List of tensor operations at this position, nullptr if none
  const std::vector<TensorOpEntry> *GetOpsAtPosition(size_t exec_id) const;

  // Check if a position has operations
  bool HasOpsAtPosition(size_t exec_id) const;
  bool HasPreOpsAtPosition(size_t exec_id) const;

  // Check if strategy is ready
  bool IsReady() const { return is_ready_; }

  // Debug print
  std::string DebugString() const;

 private:
  // Core data: exec_id -> list of post-operations (executed after op)
  std::unordered_map<size_t, std::vector<TensorOpEntry>> position_ops_;

  // Core data: exec_id -> list of pre-operations (executed before op)
  std::unordered_map<size_t, std::vector<TensorOpEntry>> pre_position_ops_;

  // Is strategy ready for use
  bool is_ready_{false};
};

using OffloadStrategyPtr = std::shared_ptr<OffloadStrategy>;

}  // namespace hyper_offload
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PYNATIVE_HYPER_OFFLOAD_OFFLOAD_STRATEGY_H_
