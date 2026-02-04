/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "include/runtime/pipeline/pipeline.h"
#include <memory>
#include "include/runtime/pipeline/async_rqueue.h"
#include "pybind_api/gil_scoped_long_running.h"

namespace mindspore {
namespace runtime {
Pipeline &Pipeline::Get() {
  static Pipeline instance;
  return instance;
}

Pipeline::Pipeline()
    : frontend_stage_(std::make_unique<AsyncRQueue>("frontend_queue", runtime::kThreadWaitLevel::kLevelFrontend)),
      bprop_stage_(std::make_unique<AsyncRQueue>("bprop_queue", kThreadWaitLevel::kLevelGrad)),
      backend_stage_(std::make_unique<AsyncRQueue>("backend_queue", kThreadWaitLevel::kLevelBackend)),
      launch_stage_(std::make_unique<AsyncRQueue>("launch_queue", kThreadWaitLevel::kLevelDevice)),
      stress_detect_(std::make_unique<AsyncRQueue>("stress_detect_queue", kThreadWaitLevel::kLevelStressDetect)) {}

void Pipeline::WaitAll() {
  GilReleaseWithCheck gil_release;
  frontend_stage_->Wait();
  bprop_stage_->Wait();
  backend_stage_->Wait();
  launch_stage_->Wait();
}

void Pipeline::WaitForward() {
  GilReleaseWithCheck gil_release;
  frontend_stage_->Wait();
  backend_stage_->Wait();
  launch_stage_->Wait();
}

void Pipeline::WaitFrontend() {
  GilReleaseWithCheck gil_release;
  frontend_stage_->Wait();
}

void Pipeline::WaitBackend() {
  GilReleaseWithCheck gil_release;
  frontend_stage_->Wait();
  backend_stage_->Wait();
}

void Pipeline::WaitBpropStage() {
  GilReleaseWithCheck gil_release;
  bprop_stage()->Wait();
}

void Pipeline::WaitFrontendAndBprop() {
  GilReleaseWithCheck gil_release;
  frontend_stage_->Wait();
  bprop_stage_->Wait();
}

void Pipeline::SetSpin(bool spin) {
  frontend_stage_->SetSpin(spin);
  bprop_stage_->SetSpin(spin);
  backend_stage_->SetSpin(spin);
  launch_stage_->SetSpin(spin);
}

void Pipeline::ChildAfterFork() {
  MS_LOG(DEBUG) << "Pipeline reinitialize after fork start.";
  AsyncRQueue::ChildAfterForkPre();
  frontend_stage_->ChildAfterFork();
  bprop_stage_->ChildAfterFork();
  backend_stage_->ChildAfterFork();
  launch_stage_->ChildAfterFork();
  MS_LOG(DEBUG) << "Pipeline reinitialize after fork end.";
}

void Pipeline::ParentBeforeFork() {
  WaitAll();
  GilReleaseWithCheck gil_release;
  stress_detect_->Wait();
  MS_LOG(DEBUG) << "ParentBeforeFork start";
  frontend_stage_->ParentBeforeFork();
  bprop_stage_->ParentBeforeFork();
  backend_stage_->ParentBeforeFork();
  launch_stage_->ParentBeforeFork();
  stress_detect_->ParentBeforeFork();
  MS_LOG(DEBUG) << "ParentBeforeFork end";
}

void Pipeline::DisablePipeline() {
  auto disable_pipeline = [](const AsyncRQueuePtr &stage) { stage->SetMultiThreadDisabled(true); };
  disable_pipeline(frontend_stage_);
  disable_pipeline(bprop_stage_);
  disable_pipeline(backend_stage_);
  disable_pipeline(launch_stage_);
  disable_pipeline(stress_detect_);
}

void Pipeline::DisableMultiThreadAfterFork() { Pipeline::Get().DisablePipeline(); }

void Pipeline::WorkerJoin() {
  GilReleaseWithCheck gil_release;
  frontend_stage_->WorkerJoin();
  bprop_stage_->WorkerJoin();
  backend_stage_->WorkerJoin();
  launch_stage_->WorkerJoin();
  stress_detect_->WorkerJoin();
}
}  // namespace runtime
}  // namespace mindspore
