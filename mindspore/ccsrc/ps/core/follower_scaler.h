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

#ifndef MINDSPORE_CCSRC_PS_CORE_FOLLOWER_SCALER_H_
#define MINDSPORE_CCSRC_PS_CORE_FOLLOWER_SCALER_H_

#include <map>
#include <mutex>
#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <functional>
#include <condition_variable>
#include "utils/log_adapter.h"
#include "ps/core/abstract_node.h"

namespace mindspore {
namespace ps {
namespace core {
class AbstractNode;
// Scaling state machine: kNormal->kPreparing->kWaiting->kScaling->kNormal
enum class NodeScaleState {
  // This state means the server/worker node is not involved with scaling operations.
  kNormal,
  // This state means the server/worker node is preparing for scaling. The barriers will be called when
  // server/worker node is in this state.
  kPreparing,
  // After barriers complete, the server/worker node switches into this state. This means this node is ready for
  // scaling. When in this state, server/worker node is in safemode.
  kWaiting,
  // Server/worker node will switch to this state after scheduler's scaling out/in operation is done.
  // When in this state, server/worker node can't send/receive messages.
  kScaling
};

// The class helps worker/server node to elastic scale while running a training job. In this class, the scaling events
// are triggered by scheduler and caught by worker/server.

// Modules which are involved with elastic scaling should register handlers to this class. After scheduler receives
// elastic scaling messages from user or cluster manager, it triggers events and the handlers will be called so that
// every module's consistency is guaranteed.
class FollowerScaler {
 public:
  explicit FollowerScaler(AbstractNode *node);
  ~FollowerScaler();

  // The methods called after the events READY_FOR_SCALE_OUT/READY_FOR_SCALE_IN are triggered.
  void ProcessBeforeScaleOut();
  void ProcessBeforeScaleIn();

  // The methods called after the events CLUSTER_SCALE_OUT_DONE/CLUSTER_SCALE_IN_DONE are triggered.
  void ProcessAfterScaleOut();
  void ProcessAfterScaleIn();

  void RegisterBarrierBeforeScaleOut(const std::string &module, const BarrierBeforeScaleOut &barrier);
  void RegisterBarrierBeforeScaleIn(const std::string &module, const BarrierBeforeScaleIn &barrier);
  void RegisterHandlerAfterScaleOut(const std::string &module, const HandlerAfterScaleOut &handler);
  void RegisterHandlerAfterScaleIn(const std::string &module, const HandlerAfterScaleIn &handler);

  // Register the scaling event callbacks to the node.
  void RegisterScaleEventCallbacks();

 private:
  AbstractNode *node_;

  std::atomic<NodeScaleState> scaling_state_;

  // Callbacks for scaling events should not be blocked so we notify a thread to call
  // barriers(barriers_before_scale_out_/barriers_before_scale_in_) or
  // handlers(handlers_after_scale_out_/handlers_after_scale_in_).
  std::atomic_bool running_;
  std::thread process_before_scale_out_thread_;
  std::thread process_before_scale_in_thread_;
  std::thread process_after_scale_out_thread_;
  std::thread process_after_scale_in_thread_;

  // Variables for signals of scaling out/in operations.
  std::mutex scale_out_mtx_;
  std::mutex scale_in_mtx_;
  std::condition_variable scale_out_cv_;
  std::condition_variable scale_in_cv_;

  // Barriers and handlers for scale out/in events.
  std::map<std::string, BarrierBeforeScaleOut> barriers_before_scale_out_;
  std::map<std::string, BarrierBeforeScaleIn> barriers_before_scale_in_;
  std::map<std::string, HandlerAfterScaleOut> handlers_after_scale_out_;
  std::map<std::string, HandlerAfterScaleIn> handlers_after_scale_in_;

  std::function<void(void)> ready_for_scale_out_event_callback_;
  std::function<void(void)> ready_for_scale_in_event_callback_;
  std::function<void(void)> scale_out_done_event_callback_;
  std::function<void(void)> scale_in_done_event_callback_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_FOLLOWER_SCALER_H_
