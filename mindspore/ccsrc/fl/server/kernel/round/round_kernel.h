/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FL_SERVER_KERNEL_ROUND_ROUND_KERNEL_H_
#define MINDSPORE_CCSRC_FL_SERVER_KERNEL_ROUND_ROUND_KERNEL_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <queue>
#include <utility>
#include <chrono>
#include <thread>
#include <unordered_map>
#include "kernel/common_utils.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "fl/server/common.h"
#include "fl/server/local_meta_store.h"
#include "fl/server/distributed_count_service.h"
#include "fl/server/distributed_metadata_store.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
constexpr uint64_t kReleaseDuration = 100;
// RoundKernel contains the main logic of server handling messages from workers. One iteration has multiple round
// kernels to represent the process. They receive and parse messages from the server communication module. After
// handling these messages, round kernels allocate response data and send it back.

// For example, the main process of federated learning is:
// startFLJob round->updateModel round->getModel round.
class RoundKernel {
 public:
  RoundKernel();
  virtual ~RoundKernel();

  // Initialize RoundKernel with threshold_count which means that for every iteration, this round needs threshold_count
  // messages.
  virtual void InitKernel(size_t threshold_count) = 0;

  // Launch the round kernel logic to handle the message passed by the communication module.
  virtual bool Launch(const uint8_t *req_data, size_t len,
                      const std::shared_ptr<ps::core::MessageHandler> &message) = 0;

  // Some rounds could be stateful in a iteration. Reset method resets the status of this round.
  virtual bool Reset() = 0;

  // The counter event handlers for DistributedCountService.
  // The callbacks when first message and last message for this round kernel is received.
  // These methods is called by class DistributedCountService and triggered by counting server.
  virtual void OnFirstCountEvent(const std::shared_ptr<ps::core::MessageHandler> &message);
  virtual void OnLastCountEvent(const std::shared_ptr<ps::core::MessageHandler> &message);

  // Called when this round is finished. This round timer's Stop method will be called.
  void StopTimer() const;

  // Called after this iteration(including all rounds) is finished. All rounds' Reset method will
  // be called.
  void FinishIteration(bool is_last_iter_valid, const std::string &reason = "") const;

  // Set round kernel name, which could be used in round kernel's methods.
  void set_name(const std::string &name);

  // Set callbacks to be called under certain triggered conditions.
  void set_stop_timer_cb(const StopTimerCb &timer_stopper);

  void Summarize();

  void IncreaseTotalClientNum();

  void IncreaseAcceptClientNum();

  size_t total_client_num() const;

  size_t accept_client_num() const;

  size_t reject_client_num() const;

  void InitClientVisitedNum();

  void InitClientUploadLoss();

  void UpdateClientUploadLoss(const float upload_loss);

  float upload_loss() const;

 protected:
  // Send response to client, and the data can be released after the call.
  void SendResponseMsg(const std::shared_ptr<ps::core::MessageHandler> &message, const void *data, size_t len);
  // Send response to client, and the data will be released by cb after finished send msg.
  void SendResponseMsgInference(const std::shared_ptr<ps::core::MessageHandler> &message, const void *data, size_t len,
                                ps::core::RefBufferRelCallback cb);

  // Round kernel's name.
  std::string name_;

  // The current received message count for this round in this iteration.
  size_t current_count_;

  StopTimerCb stop_timer_cb_;

  // Members below are used for allocating and releasing response data on the heap.

  // To ensure the performance, we use another thread to release data on the heap. So the operation on the data should
  // be threadsafe.
  std::atomic_bool running_;

  std::atomic<size_t> total_client_num_;
  std::atomic<size_t> accept_client_num_;

  std::atomic<float> upload_loss_;
};
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_KERNEL_ROUND_ROUND_KERNEL_H_
