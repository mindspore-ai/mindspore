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

#include "ps/core/follower_scaler.h"
#include "ps/core/communicator/tcp_communicator.h"

namespace mindspore {
namespace ps {
namespace core {
FollowerScaler::FollowerScaler(AbstractNode *node)
    : node_(node), scaling_state_(NodeScaleState::kNormal), running_(true) {
  process_before_scale_out_thread_ = std::thread([&]() {
    while (running_.load()) {
      std::unique_lock<std::mutex> lock(scale_out_mtx_);
      scale_out_cv_.wait(
        lock, [&]() -> bool { return !running_.load() || scaling_state_.load() == NodeScaleState::kPreparing; });
      if (!running_.load()) {
        break;
      }
      ProcessBeforeScaleOut();
    }
  });

  process_before_scale_in_thread_ = std::thread([&]() {
    while (running_.load()) {
      std::unique_lock<std::mutex> lock(scale_in_mtx_);
      scale_in_cv_.wait(
        lock, [&]() -> bool { return !running_.load() || scaling_state_.load() == NodeScaleState::kPreparing; });
      // In scaling in scenario, abstract node will trigger CLUSTER_SCALE_IN_DONE event in the same thread if this node
      // is the one to be scaled in, so we need to release the lock here to avoid dead lock.
      lock.unlock();
      if (!running_.load()) {
        break;
      }
      ProcessBeforeScaleIn();
    }
  });

  process_after_scale_out_thread_ = std::thread([&]() {
    while (running_.load()) {
      std::unique_lock<std::mutex> lock(scale_out_mtx_);
      scale_out_cv_.wait(
        lock, [&]() -> bool { return !running_.load() || scaling_state_.load() == NodeScaleState::kScaling; });
      if (!running_.load()) {
        break;
      }
      ProcessAfterScaleOut();
    }
  });

  process_after_scale_in_thread_ = std::thread([&]() {
    while (running_.load()) {
      std::unique_lock<std::mutex> lock(scale_in_mtx_);
      scale_in_cv_.wait(
        lock, [&]() -> bool { return !running_.load() || scaling_state_.load() == NodeScaleState::kScaling; });
      if (!running_.load()) {
        break;
      }
      ProcessAfterScaleIn();
    }
  });
}

FollowerScaler::~FollowerScaler() {
  running_ = false;
  scale_out_cv_.notify_all();
  scale_in_cv_.notify_all();
  if (process_before_scale_out_thread_.joinable()) {
    process_before_scale_out_thread_.join();
  }
  if (process_before_scale_in_thread_.joinable()) {
    process_before_scale_in_thread_.join();
  }
  if (process_after_scale_out_thread_.joinable()) {
    process_after_scale_out_thread_.join();
  }
  if (process_after_scale_in_thread_.joinable()) {
    process_after_scale_in_thread_.join();
  }
}

void FollowerScaler::RegisterScaleEventCallbacks() {
  ready_for_scale_out_event_callback_ = [&]() -> void {
    // Notify the thread which will call the barriers.
    std::unique_lock<std::mutex> lock(scale_out_mtx_);
    scaling_state_ = NodeScaleState::kPreparing;
    scale_out_cv_.notify_all();
  };

  ready_for_scale_in_event_callback_ = [&]() -> void {
    std::unique_lock<std::mutex> lock(scale_in_mtx_);
    scaling_state_ = NodeScaleState::kPreparing;
    scale_in_cv_.notify_all();
  };

  scale_out_done_event_callback_ = [&]() -> void {
    std::unique_lock<std::mutex> lock(scale_out_mtx_);
    scaling_state_ = NodeScaleState::kScaling;
    scale_out_cv_.notify_all();
  };

  scale_in_done_event_callback_ = [&]() -> void {
    std::unique_lock<std::mutex> lock(scale_in_mtx_);
    scaling_state_ = NodeScaleState::kScaling;
    scale_in_cv_.notify_all();
  };

  MS_EXCEPTION_IF_NULL(node_);
  node_->RegisterEventCallback(core::ClusterEvent::READY_FOR_SCALE_OUT, ready_for_scale_out_event_callback_);
  node_->RegisterEventCallback(core::ClusterEvent::READY_FOR_SCALE_IN, ready_for_scale_in_event_callback_);
  node_->RegisterEventCallback(core::ClusterEvent::CLUSTER_SCALE_OUT_DONE, scale_out_done_event_callback_);
  node_->RegisterEventCallback(core::ClusterEvent::CLUSTER_SCALE_IN_DONE, scale_in_done_event_callback_);
}

void FollowerScaler::ProcessBeforeScaleOut() {
  for (auto &barrier : barriers_before_scale_out_) {
    MS_LOG(INFO) << "Calling barrier before scaling out for " << barrier.first;
    barrier.second();
  }
  scaling_state_ = NodeScaleState::kWaiting;
  // Notify scheduler that this node is ready for elastic scaling out.
  node_->set_ready_for_scale_out();
}

void FollowerScaler::ProcessBeforeScaleIn() {
  for (auto &barrier : barriers_before_scale_in_) {
    MS_LOG(INFO) << "Calling barrier before scaling in for " << barrier.first;
    barrier.second();
  }
  scaling_state_ = NodeScaleState::kWaiting;
  // Notify scheduler that this node is ready for elastic scaling in.
  node_->set_ready_for_scale_in();
}

void FollowerScaler::ProcessAfterScaleOut() {
  MS_LOG(INFO) << "Scaling out operation in scheduler is done. Do scaling out for this node.";
  for (auto &handler : handlers_after_scale_out_) {
    MS_LOG(INFO) << "Calling scaling out handler for " << handler.first;
    handler.second();
  }
  scaling_state_ = NodeScaleState::kNormal;
  // Notify scheduler that scaling out of this node is done.
  node_->set_scale_out_done();
}

void FollowerScaler::ProcessAfterScaleIn() {
  MS_LOG(INFO) << "Scaling in operation in scheduler is done. Do scaling in for this node.";
  for (auto &handler : handlers_after_scale_in_) {
    MS_LOG(INFO) << "Calling scaling in handler for " << handler.first;
    handler.second();
  }
  scaling_state_ = NodeScaleState::kNormal;
  // Notify scheduler that scaling out of this node is done.
  node_->set_scale_in_done();
}

void FollowerScaler::RegisterBarrierBeforeScaleOut(const std::string &module, const BarrierBeforeScaleOut &barrier) {
  (void)barriers_before_scale_out_.try_emplace(module, barrier);
}

void FollowerScaler::RegisterBarrierBeforeScaleIn(const std::string &module, const BarrierBeforeScaleIn &barrier) {
  (void)barriers_before_scale_in_.try_emplace(module, barrier);
}

void FollowerScaler::RegisterHandlerAfterScaleOut(const std::string &module, const HandlerAfterScaleOut &handler) {
  (void)handlers_after_scale_out_.try_emplace(module, handler);
}

void FollowerScaler::RegisterHandlerAfterScaleIn(const std::string &module, const HandlerAfterScaleIn &handler) {
  (void)handlers_after_scale_in_.try_emplace(module, handler);
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
