/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_RUNTIME_CORE_GRAPH_EXECUTOR_PIPELINE_RUNTIME_PIPELINE_H_
#define MINDSPORE_CCSRC_RUNTIME_CORE_GRAPH_EXECUTOR_PIPELINE_RUNTIME_PIPELINE_H_

#include <set>
#include "include/backend/visible.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "backend/ms_backend/runtime/graph_executor/pipeline/async_lf_queue.h"

namespace mindspore {
namespace runtime {
// Singleton pipeline managing multi-stage asynchronous processing tasks
// Uses lock-free queues for thread-safe operations between stages: infer -> resize -> launch.
class BACKEND_EXPORT RuntimePipeline {
 public:
  static RuntimePipeline &GetInstance();

  AsyncLFQueuePtr &infer_queue() { return infer_queue_; }
  AsyncLFQueuePtr &resize_queue() { return resize_queue_; }
  AsyncLFQueuePtr &launch_queue() { return launch_queue_; }

  // Suspends all pipeline queue, can not push element to a queue which is in pause status.
  void PauseAll();
  // Continue all pipeline queue which is in pause status.
  void ContinueAll();

  // Blocks until all queued tasks complete processing.
  void WaitAll();
  // Waits for worker threads to terminate (shutdown sequence).
  void WorkerJoin();

  // Pre-fork preparation in parent process (flushes buffers).
  void ParentBeforeFork();
  // Post-fork cleanup in child process (resets resources).
  void ChildAfterFork();

  void AddDeviceContext(const device::DeviceContext *device_context);

  const std::set<const device::DeviceContext *> &GetAllDeviceContexts() const;

  // Bind device and set device context for async pipeline threads.
  void BindDevice();

 private:
  RuntimePipeline();
  ~RuntimePipeline() = default;
  DISABLE_COPY_AND_ASSIGN(RuntimePipeline);

  AsyncLFQueuePtr infer_queue_;
  AsyncLFQueuePtr resize_queue_;
  AsyncLFQueuePtr launch_queue_;

  std::set<const device::DeviceContext *> device_contexts_;
};
}  // namespace runtime
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_CORE_GRAPH_EXECUTOR_PIPELINE_RUNTIME_PIPELINE_H_
