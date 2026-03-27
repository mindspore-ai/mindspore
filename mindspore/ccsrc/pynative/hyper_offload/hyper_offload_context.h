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

#ifndef MINDSPORE_CCSRC_PYNATIVE_HYPER_OFFLOAD_OFFLOAD_CONTEXT_H_
#define MINDSPORE_CCSRC_PYNATIVE_HYPER_OFFLOAD_OFFLOAD_CONTEXT_H_

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "ir/value.h"
#include "pynative/hyper_offload/op_tracer.h"
#include "pynative/hyper_offload/offload_executor.h"
#include "pynative/hyper_offload/offload_strategy.h"
#include "include/pynative/utils/runtime/stream.h"

namespace mindspore {
namespace pynative {
namespace hyper_offload {

// State of the HyperOffloadContext
enum class OffloadState {
  kIdle,      // Not active, do nothing
  kWarmUp,    // Warm-up phase, trace the graph
  kExecuting  // Replay phase, execute with offload strategy
};

// HyperOffloadContext manages the lifecycle of offload tracing and execution.
// It maintains a state machine and dispatches OnOpExecuted to different logic.
class HyperOffloadContext {
 public:
  HyperOffloadContext() = default;
  ~HyperOffloadContext() = default;

  // State transition methods (called from Python side)
  // Start tracing the graph (enter TRACING state)
  void StartOffload();

  // End tracing and build the graph (enter READY state)
  void EndOffload();

  // Async versions - push tasks to frontend_stage for async execution
  void StartOffloadAsync();
  void EndOffloadAsync();

  // Reset to IDLE state
  void Reset();

  // Core dispatch function called by OpRunner's PreHook (before op execution)
  // Used in EXECUTING state to perform prefetch or wait operations before op runs
  void OnOpWillExecute(const std::string &op_name, const std::vector<ValuePtr> &inputs);

  // Core dispatch function called by OpRunner's PostHook (after op execution)
  // Dispatches to different logic based on current state
  void OnOpExecuted(const std::string &op_name, const std::vector<ValuePtr> &inputs, const ValuePtr &output);

  // Finalize the traced graph by setting output (called by OffloadManager::Analyse)
  bool FinalizeGraph();

  // Set strategy (called by OffloadManager::Analyse after warm-up)
  void SetStrategy(const OffloadStrategyPtr &strategy) { strategy_ = strategy; }

  // Getters
  OffloadState state() const { return state_; }
  OpTracerPtr tracer() const { return tracer_; }
  OffloadExecutorPtr executor() const { return executor_; }
  FuncGraphPtr GetTracedGraph() const { return tracer_ != nullptr ? tracer_->GetTracedGraph() : nullptr; }
  size_t total_traced_ops() const { return total_traced_ops_; }
  hal::StreamPtr copy_stream() const { return copy_stream_; }

  // Check if context is active (not IDLE)
  bool IsActive() const { return state_ != OffloadState::kIdle; }

 private:
  // Validate that the replay operator matches the warm-up sequence at exec_id.
  void CheckReplayOp(size_t exec_id, const std::string &op_name) const;

  // Handle op execution in TRACING state
  void HandleTracing(const std::string &op_name, const std::vector<ValuePtr> &inputs, const ValuePtr &output);

  // Handle pre-execution in EXECUTING state (PreHook)
  void HandlePreExecuting(const std::string &op_name, const std::vector<ValuePtr> &inputs);

  // Handle post-execution in EXECUTING state (PostHook)
  void HandleExecuting(const std::string &op_name, const std::vector<ValuePtr> &inputs, const ValuePtr &output);

  // Initialize compute and copy streams (called once on first EXECUTING state)
  void InitStreams();

  // Current state
  OffloadState state_{OffloadState::kIdle};

  // OpTracer for graph tracing (used in TRACING state)
  OpTracerPtr tracer_{nullptr};

  // OffloadExecutor for executing offload/load operations (used in EXECUTING state)
  OffloadExecutorPtr executor_{nullptr};

  // Copy stream (dedicated for D2H/H2D, shared across steps)
  hal::StreamPtr copy_stream_{nullptr};

  // Total ops traced during warm-up (fixed after warm-up completes)
  size_t total_traced_ops_{0};

  // Current execution position within a step (reset to 0 at StartOffload)
  size_t current_op_index_{0};

  // Step counter for warm-up phase (forward=0, backward=1, etc.)
  // Incremented at each EndOffload during warm-up
  size_t warmup_step_counter_{0};

  // Step counter for executing phase (forward=0, backward=1, etc.)
  // Incremented at each EndOffload during executing
  size_t exec_step_counter_{0};

  // Offload strategy converted from HyperOffloadPlan
  OffloadStrategyPtr strategy_{nullptr};
};

using HyperOffloadContextPtr = std::shared_ptr<HyperOffloadContext>;

// Static PreHook registration for OpRunner
// This function is registered as PreHook and dispatches to HyperOffloadContext
void OffloadPreHook(const std::string &op_name, const std::vector<ValuePtr> &inputs);

// Static PostHook registration for OpRunner
// This function is registered as PostHook and dispatches to HyperOffloadContext
void OffloadPostHook(const std::string &op_name, const std::vector<ValuePtr> &inputs, const ValuePtr &output);

}  // namespace hyper_offload
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PYNATIVE_HYPER_OFFLOAD_OFFLOAD_CONTEXT_H_
