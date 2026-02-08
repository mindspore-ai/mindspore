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

#include "backend/ms_backend/runtime/graph_executor/pipeline/runtime_pipeline.h"
#include <memory>

namespace mindspore {
namespace runtime {
RuntimePipeline &RuntimePipeline::GetInstance() {
  static RuntimePipeline instance;
  return instance;
}

RuntimePipeline::RuntimePipeline()
    : infer_queue_(std::make_unique<AsyncLFQueue>("infer_queue")),
      resize_queue_(std::make_unique<AsyncLFQueue>("resize_queue")),
      launch_queue_(std::make_unique<AsyncLFQueue>("launch_queue")) {}

void RuntimePipeline::PauseAll() {
  infer_queue_->Pause();
  resize_queue_->Pause();
  launch_queue_->Pause();
}

void RuntimePipeline::ContinueAll() {
  infer_queue_->Continue();
  resize_queue_->Continue();
  launch_queue_->Continue();
}

void RuntimePipeline::WaitAll() {
  infer_queue_->Wait();
  resize_queue_->Wait();
  launch_queue_->Wait();
}

void RuntimePipeline::WorkerJoin() {
  infer_queue_->WorkerJoin();
  resize_queue_->WorkerJoin();
  launch_queue_->WorkerJoin();
}

void RuntimePipeline::ChildAfterFork() {
  MS_LOG(DEBUG) << "RuntimePipeline reinitialize after fork start.";
  if (infer_queue_ != nullptr) {
    (void)infer_queue_.release();
    infer_queue_ = std::make_unique<AsyncLFQueue>("infer_queue");
  }
  if (resize_queue_ != nullptr) {
    (void)resize_queue_.release();
    resize_queue_ = std::make_unique<AsyncLFQueue>("resize_queue");
  }
  if (launch_queue_ != nullptr) {
    (void)launch_queue_.release();
    launch_queue_ = std::make_unique<AsyncLFQueue>("launch_queue");
  }
  device_contexts_.clear();
  MS_LOG(DEBUG) << "RuntimePipeline reinitialize after fork end.";
}

void RuntimePipeline::ParentBeforeFork() { PauseAll(); }

void RuntimePipeline::AddDeviceContext(const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(device_context);
  (void)device_contexts_.insert(device_context);
}

const std::set<const device::DeviceContext *> &RuntimePipeline::GetAllDeviceContexts() const {
  return device_contexts_;
}

void RuntimePipeline::BindDevice() {
  infer_queue_->BindDevice(device_contexts_);
  resize_queue_->BindDevice(device_contexts_);
  launch_queue_->BindDevice(device_contexts_);
}

}  // namespace runtime
}  // namespace mindspore
