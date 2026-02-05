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

#include "pynative/hyper_offload/offload_manager.h"
#include "include/runtime/pipeline/pipeline.h"
#include "include/pynative/utils/pyboost/op_runner.h"
#include "include/pynative/utils/runtime/op_backend/op_backend.h"
#include "utils/log_adapter.h"
#include "utils/ir_dump/anf_ir_dump.h"
#include "pynative/backward/grad_utils.h"
#include "pynative/hyper_offload/offload_task.h"
#include "mindspore/ccsrc/frontend/jit/ps/resource.h"
#include "mindspore/ccsrc/include/frontend/jit/ps/action_interface.h"
#include "pynative/hyper_offload/offload_manager.h"

namespace mindspore {
namespace pynative {
namespace hyper_offload {

using VmEvalPtr = std::shared_ptr<std::function<BaseRef(const VectorRef &)>>;

OffloadManager &OffloadManager::GetInstance() {
  static OffloadManager instance;
  return instance;
}

void OffloadRunOPPreHook(const BackendOpRunInfoPtr &op_run_info) {
  // Get op_name from BackendOpRunInfo
  const std::string &op_name = op_run_info->base_op_run_info.op_name;

  // Get inputs from BackendOpRunInfo (already vector<ValuePtr>)
  const std::vector<ValuePtr> &inputs = op_run_info->base_op_run_info.expanded_input_values;

  auto sliding_window_mgr = OffloadManager::GetInstance().GetSlidingWindowMgr();
  if (sliding_window_mgr != nullptr) {
    sliding_window_mgr->PreProcess(op_name, inputs);
  }

  if (OffloadManager::skip_offload()) {
    return;
  }
  auto current_ctx = OffloadManager::GetCurrentContext();
  if (current_ctx == nullptr) {
    return;
  }
  current_ctx->OnOpWillExecute(op_name, inputs);
}

void OffloadRunOPPostHook(const BackendOpRunInfoPtr &op_run_info, const VectorRef &outputs) {
  // Get op_name from BackendOpRunInfo
  const std::string &op_name = op_run_info->base_op_run_info.op_name;

  // Get inputs from BackendOpRunInfo (already vector<ValuePtr>)
  const std::vector<ValuePtr> &inputs = op_run_info->base_op_run_info.expanded_input_values;

  // Convert VectorRef outputs to ValuePtr
  // Use abstract to determine if this is a multi-output op
  const auto &output_abs = op_run_info->base_op_run_info.abstract;

  bool is_multi_output = (output_abs != nullptr && output_abs->isa<abstract::AbstractSequence>());

  ValuePtr output_value = AutoGradUtil::VectorRefToValue(outputs, false, is_multi_output);

  auto sliding_window_mgr = OffloadManager::GetInstance().GetSlidingWindowMgr();
  if (sliding_window_mgr != nullptr) {
    sliding_window_mgr->PostProcess(op_name, inputs, output_value);
  }

  if (OffloadManager::skip_offload()) {
    return;
  }
  auto current_ctx = OffloadManager::GetCurrentContext();
  if (current_ctx == nullptr) {
    return;
  }
  current_ctx->OnOpExecuted(op_name, inputs, output_value);
}

void SlidingWindowPreHook(const std::string &op_name, const std::vector<ValuePtr> &inputs) {
  auto sliding_window_mgr = OffloadManager::GetInstance().GetSlidingWindowMgr();
  if (sliding_window_mgr != nullptr) {
    sliding_window_mgr->PreProcess(op_name, inputs);
  }
}

void SlidingWindowPostHook(const std::string &op_name, const std::vector<ValuePtr> &inputs, const ValuePtr &output) {
  auto sliding_window_mgr = OffloadManager::GetInstance().GetSlidingWindowMgr();
  if (sliding_window_mgr != nullptr) {
    sliding_window_mgr->PostProcess(op_name, inputs, output);
  }
}

void OffloadPreHook(const std::string &op_name, const std::vector<ValuePtr> &inputs) {
  // Get current active context directly (set by StartOffload, cleared by EndOffload)
  auto *current_ctx = OffloadManager::GetCurrentContext();
  if (current_ctx == nullptr) {
    // No active context, skip
    return;
  }

  current_ctx->OnOpWillExecute(op_name, inputs);
}

void OffloadPostHook(const std::string &op_name, const std::vector<ValuePtr> &inputs, const ValuePtr &output) {
  // Get current active context directly (set by StartOffload, cleared by EndOffload)
  auto *current_ctx = OffloadManager::GetCurrentContext();
  if (current_ctx == nullptr) {
    // No active context, skip
    return;
  }

  current_ctx->OnOpExecuted(op_name, inputs, output);
}

void ForwardVmOpPreHook(const std::string &op_name, const std::vector<ValuePtr> &inputs) {
  if (!OffloadManager::IsOffloadReady()) {
    SlidingWindowPreHook(op_name, inputs);
  }
  // if (!OffloadManager::skip_offload()) {
  //   OffloadPreHook(op_name, inputs);
  // }
}

void ForwardVmOpPostHook(const std::string &op_name, const std::vector<ValuePtr> &inputs, const ValuePtr &output) {
  if (!OffloadManager::IsOffloadReady()) {
    SlidingWindowPostHook(op_name, inputs, output);
  }
  // if (!OffloadManager::skip_offload()) {
  //   OffloadPostHook(op_name, inputs, output);
  // }
}

// Static registrar to register PreHook and PostHook at program startup
struct OffloadHookRegistrar {
  OffloadHookRegistrar() {
    // Register PreHook for PyBoost OpRunner (pyboost operators)
    // is_enabled: fast check if there is an active context
    // handler: the actual hook function
    kernel::pyboost::OpRunner::RegisterPreHook([]() { return OffloadManager::HasActiveContext(); },  // is_enabled
                                               OffloadPreHook);                                      // handler
    MS_LOG(DEBUG) << "OffloadPreHook registered to OpRunner.";

    // Register PostHook for PyBoost OpRunner (pyboost operators)
    kernel::pyboost::OpRunner::RegisterPostHook([]() { return OffloadManager::HasActiveContext(); },  // is_enabled
                                                OffloadPostHook);                                     // handler
    MS_LOG(DEBUG) << "OffloadPostHook registered to OpRunner.";

    kernel::pyboost::OpRunner::RegisterPreHook([]() { return !OffloadManager::IsOffloadReady(); },
                                               SlidingWindowPreHook);
    MS_LOG(DEBUG) << "SlidingWindowPreHook registered to OpRunner.";

    kernel::pyboost::OpRunner::RegisterPostHook([]() { return !OffloadManager::IsOffloadReady(); },
                                                SlidingWindowPostHook);
    MS_LOG(DEBUG) << "SlidingWindowPostHook registered to OpRunner.";

    // Register PreHook for OpBackend (RunOp operators)
    compile::OpBackend::RegisterPreHook(OffloadRunOPPreHook);
    MS_LOG(DEBUG) << "OffloadPreHook registered to OpBackend.";

    // Register PostHook for OpBackend (RunOp operators)
    compile::OpBackend::RegisterPostHook(OffloadRunOPPostHook);
    MS_LOG(DEBUG) << "OffloadPostHook registered to OpBackend.";
  }
};

// This static instance triggers registration at program startup
static OffloadHookRegistrar g_offload_hook_registrar;

void OffloadManager::SetSkipOffloadAsync(bool skip) {
  auto task = std::make_shared<OffloadAsyncTask>([skip]() { skip_offload_ = skip; }, runtime::TaskType::kFrontendTask);
  static bool need_sync = runtime::OpExecutor::NeedSync();
  if (need_sync && !runtime::OpExecutor::GetInstance().async_for_graph()) {
    runtime::Pipeline::Get().WaitForward();
    task->Run();
  } else {
    runtime::Pipeline::Get().frontend_stage()->Push(task);
  }
}

HyperOffloadContextPtr OffloadManager::GetOrCreateOffloadCtx(size_t offload_key) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = contexts_.find(offload_key);
  if (it != contexts_.end()) {
    // Context already exists, return it
    MS_LOG(DEBUG) << "Returning existing HyperOffloadContext for key: " << offload_key;
    return it->second;
  }

  if (!offload_ready_) {
    // Warm-up phase: create a new context
    auto ctx = std::make_shared<HyperOffloadContext>();
    contexts_[offload_key] = ctx;
    MS_LOG(DEBUG) << "Created new HyperOffloadContext for key: " << offload_key
                  << ", total count: " << contexts_.size();
    return ctx;
  }

  // Replay phase but context not found: error
  MS_LOG(EXCEPTION) << "No HyperOffloadContext found for key " << offload_key << " in replay phase!";
}

void OffloadManager::Analyse() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (offload_ready_) {
    MS_LOG(DEBUG) << "Offload already ready, skipping Analyse.";
    return;
  }
  runtime::Pipeline::Get().WaitFrontend();
  // Mark warm-up as completed
  runtime::Pipeline::Get().WaitFrontend();
  offload_ready_ = true;

  MS_LOG(INFO) << "OffloadManager::Analyse completed. Offload is now ready.";
  MS_LOG(INFO) << "Total contexts: " << contexts_.size();

  // Finalize sliding window manager
  if (sliding_window_mgr_ != nullptr) {
    sliding_window_mgr_->Finalize();
    sliding_window_mgr_ = nullptr;
  }

  // Finalize all graphs and generate strategies
  for (const auto &[key, ctx] : contexts_) {
    // Finalize graph: set output (return node)
    bool success = ctx->FinalizeGraph();
    if (!success) {
      MS_LOG(DEBUG) << "OffloadManager::Analyse failed to finalize graph for key " << key;
      continue;
    }

    // Dump the finalized graph
    auto graph = ctx->GetTracedGraph();
    DumpIR("offload_graph_" + std::to_string(key) + ".ir", graph);
    MS_LOG(DEBUG) << "Dumped offload graph for key " << key;
    utils::hyper_offload::HyperOffloadInput input;
    input.graph = graph;
    input.exec_order = ctx->tracer()->GetExecutionOrder();
    input.enable_offload_activation = true;
    utils::hyper_offload::HyperOffloadOptimizer optimizer(input);
    auto plan = optimizer.Run();
    auto strategy = std::make_shared<OffloadStrategy>();
    strategy->ConvertFromPlan(plan);
    ctx->SetStrategy(strategy);
  }
}

void OffloadManager::Reset() {
  std::lock_guard<std::mutex> lock(mutex_);

  // Reset all contexts
  for (auto &item : contexts_) {
    const auto &ctx = item.second;
    ctx->Reset();
  }
  contexts_.clear();

  // Reset state
  offload_ready_ = false;

  MS_LOG(INFO) << "OffloadManager reset.";
}

void OffloadManager::ClearResources() {
  std::lock_guard<std::mutex> lock(mutex_);

  // Clear all contexts
  contexts_.clear();

  // Reset state
  offload_ready_ = false;
}
}  // namespace hyper_offload
}  // namespace pynative
}  // namespace mindspore
