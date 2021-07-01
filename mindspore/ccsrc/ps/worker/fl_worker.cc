/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "ps/worker/fl_worker.h"
#include "utils/ms_exception.h"

namespace mindspore {
namespace ps {
namespace worker {
void FLWorker::Run() {
  worker_num_ = PSContext::instance()->worker_num();
  server_num_ = PSContext::instance()->server_num();
  scheduler_ip_ = PSContext::instance()->scheduler_ip();
  scheduler_port_ = PSContext::instance()->scheduler_port();
  worker_step_num_per_iteration_ = PSContext::instance()->worker_step_num_per_iteration();
  PSContext::instance()->cluster_config().scheduler_host = scheduler_ip_;
  PSContext::instance()->cluster_config().scheduler_port = scheduler_port_;
  PSContext::instance()->cluster_config().initial_worker_num = worker_num_;
  PSContext::instance()->cluster_config().initial_server_num = server_num_;
  MS_LOG(INFO) << "Initialize cluster config for worker. Worker number:" << worker_num_
               << ", Server number:" << server_num_ << ", Scheduler ip:" << scheduler_ip_
               << ", Scheduler port:" << scheduler_port_;

  worker_node_ = std::make_shared<core::WorkerNode>();
  MS_EXCEPTION_IF_NULL(worker_node_);

  worker_node_->RegisterEventCallback(core::ClusterEvent::SCHEDULER_TIMEOUT, [this]() {
    Finalize();
    try {
      MS_LOG(EXCEPTION)
        << "Event SCHEDULER_TIMEOUT is captured. This is because scheduler node is finalized or crashed.";
    } catch (std::exception &e) {
      MsException::Instance().SetException();
    }
  });
  worker_node_->RegisterEventCallback(core::ClusterEvent::NODE_TIMEOUT, [this]() {
    Finalize();
    try {
      MS_LOG(EXCEPTION)
        << "Event NODE_TIMEOUT is captured. This is because some server nodes are finalized or crashed after the "
           "network building phase.";
    } catch (std::exception &e) {
      MsException::Instance().SetException();
    }
  });

  InitializeFollowerScaler();
  worker_node_->Start();
  std::this_thread::sleep_for(std::chrono::milliseconds(kWorkerSleepTimeForNetworking));
  return;
}

void FLWorker::Finalize() {
  MS_EXCEPTION_IF_NULL(worker_node_);
  worker_node_->Finish();
  worker_node_->Stop();
}

bool FLWorker::SendToServer(uint32_t server_rank, void *data, size_t size, core::TcpUserCommand command,
                            std::shared_ptr<std::vector<unsigned char>> *output) {
  // If the worker is in safemode, do not communicate with server.
  while (safemode_.load()) {
    std::this_thread::yield();
  }

  std::shared_ptr<unsigned char[]> message;
  std::unique_ptr<unsigned char[]> message_addr = std::make_unique<unsigned char[]>(size);
  MS_EXCEPTION_IF_NULL(message_addr);
  message = std::move(message_addr);
  MS_EXCEPTION_IF_NULL(message);

  uint64_t src_size = size;
  uint64_t dst_size = size;
  int ret = memcpy_s(message.get(), dst_size, data, src_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
    return false;
  }

  if (output != nullptr) {
    while (true) {
      if (!worker_node_->Send(core::NodeRole::SERVER, server_rank, message, size, static_cast<int>(command), output)) {
        MS_LOG(ERROR) << "Sending message to server " << server_rank << " failed.";
        return false;
      }
      if (*output == nullptr) {
        MS_LOG(WARNING) << "Response from server " << server_rank << " is empty.";
        return false;
      }

      if (std::string(reinterpret_cast<char *>((*output)->data()), (*output)->size()) == kClusterSafeMode) {
        MS_LOG(INFO) << "The server " << server_rank << " is in safemode.";
        std::this_thread::sleep_for(std::chrono::milliseconds(kWorkerRetryDurationForSafeMode));
      } else {
        break;
      }
    }
  } else {
    if (!worker_node_->Send(core::NodeRole::SERVER, server_rank, message, size, static_cast<int>(command))) {
      MS_LOG(ERROR) << "Sending message to server " << server_rank << " failed.";
      return false;
    }
  }
  return true;
}

uint32_t FLWorker::server_num() const { return server_num_; }

uint32_t FLWorker::worker_num() const { return worker_num_; }

uint64_t FLWorker::worker_step_num_per_iteration() const { return worker_step_num_per_iteration_; }

void FLWorker::SetIterationRunning() {
  MS_LOG(INFO) << "Worker iteration starts.";
  worker_iteration_state_ = IterationState::kRunning;
}

void FLWorker::SetIterationCompleted() {
  MS_LOG(INFO) << "Worker iteration completes.";
  worker_iteration_state_ = IterationState::kCompleted;
}

void FLWorker::InitializeFollowerScaler() {
  if (!worker_node_->InitFollowerScaler()) {
    MS_LOG(EXCEPTION) << "Initializing follower elastic scaler failed.";
    return;
  }

  // Set scaling barriers before scaling.
  worker_node_->RegisterFollowerScalerBarrierBeforeScaleOut("WorkerPipeline",
                                                            std::bind(&FLWorker::ProcessBeforeScalingOut, this));
  worker_node_->RegisterFollowerScalerBarrierBeforeScaleIn("WorkerPipeline",
                                                           std::bind(&FLWorker::ProcessBeforeScalingOut, this));

  // Set handlers after scheduler scaling operations are done.
  worker_node_->RegisterFollowerScalerHandlerAfterScaleOut("WorkerPipeline",
                                                           std::bind(&FLWorker::ProcessAfterScalingOut, this));
  worker_node_->RegisterFollowerScalerHandlerAfterScaleIn("WorkerPipeline",
                                                          std::bind(&FLWorker::ProcessAfterScalingIn, this));
  worker_node_->RegisterCustomEventCallback(static_cast<uint32_t>(CustomEvent::kIterationRunning),
                                            std::bind(&FLWorker::HandleIterationRunningEvent, this));
  worker_node_->RegisterCustomEventCallback(static_cast<uint32_t>(CustomEvent::kIterationCompleted),
                                            std::bind(&FLWorker::HandleIterationCompletedEvent, this));
}

void FLWorker::HandleIterationRunningEvent() {
  MS_LOG(INFO) << "Server iteration starts, safemode is " << safemode_.load();
  server_iteration_state_ = IterationState::kRunning;
  if (safemode_.load() == true) {
    safemode_ = false;
  }
}

void FLWorker::HandleIterationCompletedEvent() {
  MS_LOG(INFO) << "Server iteration completes";
  server_iteration_state_ = IterationState::kCompleted;
}

void FLWorker::ProcessBeforeScalingOut() {
  MS_LOG(INFO) << "Starting Worker scaling out barrier.";
  while (server_iteration_state_.load() != IterationState::kCompleted ||
         worker_iteration_state_.load() != IterationState::kCompleted) {
    std::this_thread::yield();
  }
  MS_LOG(INFO) << "Ending Worker scaling out barrier. Switch to safemode.";
  safemode_ = true;
}

void FLWorker::ProcessBeforeScalingIn() {
  MS_LOG(INFO) << "Starting Worker scaling in barrier.";
  while (server_iteration_state_.load() != IterationState::kCompleted ||
         worker_iteration_state_.load() != IterationState::kCompleted) {
    std::this_thread::yield();
  }
  MS_LOG(INFO) << "Ending Worker scaling in barrier. Switch to safemode.";
  safemode_ = true;
}

void FLWorker::ProcessAfterScalingOut() {
  if (worker_node_ == nullptr) {
    return;
  }

  MS_LOG(INFO) << "Cluster scaling out completed. Reinitialize for worker.";
  server_num_ = worker_node_->server_num();
  worker_num_ = worker_node_->worker_num();
  MS_LOG(INFO) << "After scheduler scaling out, worker number is " << worker_num_ << ", server number is "
               << server_num_ << ". Exit safemode.";
  std::this_thread::sleep_for(std::chrono::milliseconds(kWorkerSleepTimeForNetworking));
  safemode_ = false;
}

void FLWorker::ProcessAfterScalingIn() {
  if (worker_node_ == nullptr) {
    return;
  }

  MS_LOG(INFO) << "Cluster scaling in completed. Reinitialize for worker.";
  server_num_ = worker_node_->server_num();
  worker_num_ = worker_node_->worker_num();
  MS_LOG(INFO) << "After scheduler scaling in, worker number is " << worker_num_ << ", server number is " << server_num_
               << ". Exit safemode.";
  std::this_thread::sleep_for(std::chrono::milliseconds(kWorkerSleepTimeForNetworking));
  safemode_ = false;
}
}  // namespace worker
}  // namespace ps
}  // namespace mindspore
