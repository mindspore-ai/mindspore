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

#include "pynative/hyper_offload/hyper_offload_context.h"
#include <mutex>
#include <functional>
#include "utils/log_adapter.h"
#include "include/runtime/pipeline/pipeline.h"
#include "include/pynative/utils/pyboost/op_runner.h"
#include "include/pynative/utils/runtime/op_executor.h"
#include "include/pynative/utils/runtime/op_backend/op_backend.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "pynative/hyper_offload/offload_manager.h"
#include "pynative/hyper_offload/offload_task.h"

namespace mindspore {
namespace pynative {
namespace hyper_offload {

void HyperOffloadContext::StartOffload() {
  if (state_ != OffloadState::kIdle) {
    MS_LOG(EXCEPTION) << "HyperOffloadContext::StartOffload called in non-IDLE state, current state: "
                      << static_cast<int>(state_);
  }

  // Check if another context is already active
  if (OffloadManager::GetCurrentContext() != nullptr) {
    MS_LOG(EXCEPTION) << "Another HyperOffloadContext is already active! "
                      << "Only one context can be active at a time.";
  }

  bool is_ready = OffloadManager::IsOffloadReady();
  if (tracer_ == nullptr) {
    if (!is_ready) {
      tracer_ = std::make_shared<OpTracer>();
    } else {
      MS_LOG(EXCEPTION) << "HyperOffloadContext::StartOffload called in non-IDLE state, current state: " << this << " "
                        << static_cast<int>(state_);
    }
  }

  if (!is_ready) {
    // First step: warm-up phase
    state_ = OffloadState::kWarmUp;
    MS_LOG(DEBUG) << "HyperOffloadContext entered WARM_UP state.";
  } else {
    // Second step and beyond: executing phase
    state_ = OffloadState::kExecuting;

    // Initialize streams and executor on first entering EXECUTING state
    InitStreams();
    if (executor_ == nullptr) {
      executor_ = std::make_shared<OffloadExecutor>(copy_stream_);
    }

    MS_LOG(DEBUG) << "HyperOffloadContext entered EXECUTING state.";
  }

  // Set current context for PostHook to use
  OffloadManager::SetCurrentContext(this);
}

void HyperOffloadContext::EndOffload() {
  if (state_ == OffloadState::kIdle) {
    MS_LOG(EXCEPTION) << "HyperOffloadContext::EndOffload called in IDLE state."
                      << "which mean this cell input or output not requires grad!";
  }

  if (state_ == OffloadState::kWarmUp) {
    // Warm-up: accumulate total traced ops, increment step counter
    total_traced_ops_ = current_op_index_;
    ++warmup_step_counter_;
    MS_LOG(DEBUG) << "HyperOffloadContext warm-up step " << warmup_step_counter_
                  << " completed, total traced ops: " << total_traced_ops_;
  } else if (state_ == OffloadState::kExecuting) {
    ++exec_step_counter_;
    if (exec_step_counter_ == warmup_step_counter_ && current_op_index_ != total_traced_ops_) {
      MS_LOG(EXCEPTION) << "Dynamic graph structure detected! Operator count mismatch: "
                        << "traced " << total_traced_ops_ << " operators in warm-up, "
                        << "but executed " << current_op_index_ << " operators. "
                        << "Hyper offload does not support dynamic graph structures.";
    }
    MS_LOG(DEBUG) << "HyperOffloadContext executing step " << exec_step_counter_ << " completed.";
    // Reset exec_step_counter when a full iteration completes
    if (exec_step_counter_ >= warmup_step_counter_) {
      exec_step_counter_ = 0;
      current_op_index_ = 0;
    }
  }
  // Clear current context
  OffloadManager::ClearCurrentContext();
  state_ = OffloadState::kIdle;
}

void HyperOffloadContext::Reset() {
  // Clear current context if this is the active one
  if (OffloadManager::GetCurrentContext() == this) {
    OffloadManager::ClearCurrentContext();
  }

  state_ = OffloadState::kIdle;
  tracer_ = nullptr;
  strategy_ = nullptr;

  // Reset executor cache but keep executor instance
  if (executor_ != nullptr) {
    executor_->Reset();
  }

  total_traced_ops_ = 0;
  current_op_index_ = 0;
  warmup_step_counter_ = 0;
  exec_step_counter_ = 0;

  MS_LOG(INFO) << "HyperOffloadContext reset to IDLE state.";
}

void HyperOffloadContext::OnOpWillExecute(const std::string &op_name, const std::vector<ValuePtr> &inputs) {
  // PreHook only handles EXECUTING state
  if (state_ != OffloadState::kWarmUp) {
    return;
  }
  HandlePreExecuting(op_name, inputs);
}

void HyperOffloadContext::OnOpExecuted(const std::string &op_name, const std::vector<ValuePtr> &inputs,
                                       const ValuePtr &output) {
  switch (state_) {
    case OffloadState::kIdle:
      break;
    case OffloadState::kWarmUp:
      HandleTracing(op_name, inputs, output);
      break;
    case OffloadState::kExecuting:
      HandleExecuting(op_name, inputs, output);
      break;
    default:
      MS_LOG(WARNING) << "HyperOffloadContext::OnOpExecuted in unknown state.";
      break;
  }
}

void HyperOffloadContext::HandleTracing(const std::string &op_name, const std::vector<ValuePtr> &inputs,
                                        const ValuePtr &output) {
  MS_EXCEPTION_IF_NULL(tracer_);

  // Trace the operator
  tracer_->TraceOp(op_name, inputs, output);

  MS_LOG(DEBUG) << "Traced op[" << current_op_index_ << "]: " << op_name;
  ++current_op_index_;
}

namespace {
// Helper to get tensor from inputs or output by source and index
tensor::TensorPtr GetTensorFromEntry(const TensorOpEntry &entry, const std::vector<ValuePtr> &inputs,
                                     const ValuePtr &output) {
  if (entry.source == TensorSource::kOutput) {
    // Single tensor output
    if (output->isa<tensor::Tensor>()) {
      return output->cast<tensor::TensorPtr>();
    }
    // Multi-output case: ValueTuple/ValueList (assume single level)
    if (output->isa<ValueSequence>()) {
      const auto &seq = output->cast<ValueSequencePtr>();
      const auto &elements = seq->value();
      if (entry.index >= elements.size()) {
        MS_LOG(WARNING) << "Invalid output index " << entry.index << ", outputs size=" << elements.size();
        return nullptr;
      }
      const auto &element = elements[entry.index];
      if (element != nullptr && element->isa<tensor::Tensor>()) {
        return element->cast<tensor::TensorPtr>();
      }
    }
  } else {
    if (entry.index >= inputs.size()) {
      MS_LOG(WARNING) << "Invalid input index " << entry.index << ", inputs size=" << inputs.size();
      return nullptr;
    }
    const auto &input = inputs[entry.index];
    if (input != nullptr && input->isa<tensor::Tensor>()) {
      return input->cast<tensor::TensorPtr>();
    }
  }
  return nullptr;
}
}  // namespace

void HyperOffloadContext::HandlePreExecuting(const std::string &, const std::vector<ValuePtr> &) {}

void HyperOffloadContext::HandleExecuting(const std::string &op_name, const std::vector<ValuePtr> &inputs,
                                          const ValuePtr &output) {
  size_t exec_id = current_op_index_;
  ++current_op_index_;

  // Query strategy by exec_id and execute corresponding actions
  if (strategy_ == nullptr || !strategy_->IsReady()) {
    MS_LOG(DEBUG) << "Executing op[" << exec_id << "]: " << op_name << " (no strategy)";
    return;
  }

  const auto *ops = strategy_->GetOpsAtPosition(exec_id);
  if (ops == nullptr || ops->empty()) {
    MS_LOG(DEBUG) << "Executing op[" << exec_id << "]: " << op_name << " (no ops at this position)";
    return;
  }

  MS_EXCEPTION_IF_NULL(executor_);

  // Execute each operation at this position
  for (const auto &entry : *ops) {
    switch (entry.action) {
      case TensorAction::kD2HWithEvent: {
        tensor::TensorPtr tensor = GetTensorFromEntry(entry, inputs, output);
        MS_EXCEPTION_IF_NULL(tensor);
        executor_->D2HWithEvent(entry.tensor_id, tensor);
        MS_LOG(DEBUG) << "op[" << exec_id << "] " << op_name << ": D2HWithEvent for tensor_id=" << entry.tensor_id;
        break;
      }
      case TensorAction::kH2DWithEvent: {
        executor_->H2DWithEvent(entry.tensor_id);
        MS_LOG(DEBUG) << "op[" << exec_id << "] " << op_name << ": H2DWithEvent for tensor_id=" << entry.tensor_id;
        break;
      }
      case TensorAction::kWaitD2HAndSwitch: {
        executor_->WaitD2HAndSwitch(entry.tensor_id);
        MS_LOG(DEBUG) << "op[" << exec_id << "] " << op_name << ": WaitD2HAndSwitch for tensor_id=" << entry.tensor_id;
        break;
      }
      case TensorAction::kWaitH2DAndSwitch: {
        executor_->WaitH2DAndSwitch(entry.tensor_id);
        MS_LOG(DEBUG) << "op[" << exec_id << "] " << op_name << ": WaitH2DAndSwitch for tensor_id=" << entry.tensor_id;
        break;
      }
      case TensorAction::kWaitH2DAndFreeCache: {
        executor_->WaitH2DAndFreeCache(entry.tensor_id);
        MS_LOG(DEBUG) << "op[" << exec_id << "] " << op_name
                      << ": WaitH2DAndFreeCache for tensor_id=" << entry.tensor_id;
        break;
      }
      case TensorAction::kFreeDevice: {
        tensor::TensorPtr tensor = GetTensorFromEntry(entry, inputs, output);
        MS_EXCEPTION_IF_NULL(tensor);
        executor_->FreeDevice(entry.tensor_id);
        MS_LOG(DEBUG) << "op[" << exec_id << "] " << op_name << ": FreeDevice for tensor_id=" << entry.tensor_id;
        break;
      }
      default:
        break;
    }
  }
}

bool HyperOffloadContext::FinalizeGraph() {
  current_op_index_ = 0;
  if (tracer_ == nullptr) {
    MS_LOG(WARNING) << "HyperOffloadContext::FinalizeGraph called but tracer is null.";
    return false;
  }
  bool success = tracer_->FinalizeGraph();
  MS_LOG(INFO) << "HyperOffloadContext graph finalized.";
  return success;
}

void HyperOffloadContext::InitStreams() {
  if (copy_stream_ != nullptr) {
    return;  // Already initialized
  }

  // Create a dedicated copy stream with default priority
  // Note: compute stream is obtained dynamically via hal::CurrentStream() when needed
  copy_stream_ = std::make_shared<hal::Stream>(0);  // priority = 0
  MS_EXCEPTION_IF_NULL(copy_stream_);

  MS_LOG(DEBUG) << "HyperOffloadContext copy_stream initialized. copy_stream_id=" << copy_stream_->stream_id();
}

void HyperOffloadContext::StartOffloadAsync() {
  auto task = std::make_shared<OffloadAsyncTask>([this]() { StartOffload(); }, runtime::TaskType::kFrontendTask);
  GilReleaseWithCheck no_gil;
  static bool need_sync = runtime::OpExecutor::NeedSync();
  if (need_sync && !runtime::OpExecutor::GetInstance().async_for_graph()) {
    runtime::Pipeline::Get().WaitForward();
    task->Run();
  } else {
    runtime::Pipeline::Get().frontend_stage()->Push(task);
  }
}

void HyperOffloadContext::EndOffloadAsync() {
  auto task = std::make_shared<OffloadAsyncTask>([this]() { EndOffload(); }, runtime::TaskType::kFrontendTask);
  GilReleaseWithCheck no_gil;
  static bool need_sync = runtime::OpExecutor::NeedSync();
  if (need_sync && !runtime::OpExecutor::GetInstance().async_for_graph()) {
    runtime::Pipeline::Get().WaitForward();
    task->Run();
  } else {
    runtime::Pipeline::Get().frontend_stage()->Push(task);
  }
}

}  // namespace hyper_offload
}  // namespace pynative
}  // namespace mindspore
