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

#include "fl/server/kernel/round/round_kernel.h"

#include <chrono>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "fl/server/iteration.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
RoundKernel::RoundKernel() : name_(""), current_count_(0), running_(true) {}

RoundKernel::~RoundKernel() { running_ = false; }

void RoundKernel::OnFirstCountEvent(const std::shared_ptr<ps::core::MessageHandler> &) { return; }

void RoundKernel::OnLastCountEvent(const std::shared_ptr<ps::core::MessageHandler> &) { return; }

void RoundKernel::StopTimer() const {
  if (stop_timer_cb_) {
    stop_timer_cb_();
  }
  return;
}

void RoundKernel::FinishIteration(bool is_last_iter_valid, const std::string &in_reason) const {
  std::string reason = in_reason;
  if (is_last_iter_valid) {
    reason = "Round " + name_ + " finished! This iteration is valid. Proceed to next iteration.";
  }
  Iteration::GetInstance().NotifyNext(is_last_iter_valid, reason);
}

void RoundKernel::set_name(const std::string &name) { name_ = name; }

void RoundKernel::set_stop_timer_cb(const StopTimerCb &timer_stopper) { stop_timer_cb_ = timer_stopper; }

void RoundKernel::SendResponseMsg(const std::shared_ptr<ps::core::MessageHandler> &message, const void *data,
                                  size_t len) {
  if (!verifyResponse(message, data, len)) {
    return;
  }
  IncreaseTotalClientNum();
  if (!message->SendResponse(data, len)) {
    MS_LOG(WARNING) << "Sending response failed.";
    return;
  }
  CalculateSendData(len);
}

void RoundKernel::SendResponseMsgInference(const std::shared_ptr<ps::core::MessageHandler> &message, const void *data,
                                           size_t len, ps::core::RefBufferRelCallback cb) {
  if (!verifyResponse(message, data, len)) {
    return;
  }
  IncreaseTotalClientNum();
  if (!message->SendResponseInference(data, len, cb)) {
    MS_LOG(WARNING) << "Sending response failed.";
    return;
  }
  CalculateSendData(len);
}

bool RoundKernel::verifyResponse(const std::shared_ptr<ps::core::MessageHandler> &message, const void *data,
                                 size_t len) {
  if (message == nullptr) {
    MS_LOG(WARNING) << "The message handler is nullptr.";
    return false;
  }
  if (data == nullptr || len == 0) {
    std::string reason = "The output of the round " + name_ + " is empty.";
    MS_LOG(WARNING) << reason;
    if (!message->SendResponse(reason.c_str(), reason.size())) {
      MS_LOG(WARNING) << "Sending response failed.";
    }
    return false;
  }
  return true;
}

void RoundKernel::IncreaseTotalClientNum() { total_client_num_ += 1; }

void RoundKernel::IncreaseAcceptClientNum() { accept_client_num_ += 1; }

void RoundKernel::Summarize() {
  if (name_ == "startFLJob" || name_ == "updateModel" || name_ == "getModel") {
    MS_LOG(INFO) << "Round kernel " << name_ << " total client num is: " << total_client_num_
                 << ", accept client num is: " << accept_client_num_
                 << ", reject client num is: " << (total_client_num_ - accept_client_num_);
  }

  if (name_ == "updateModel" && accept_client_num() > 0) {
    MS_LOG(INFO) << "Client Upload avg Loss: " << (upload_loss_ / accept_client_num());
  }
  RecordSendData(send_data_time_, send_data_);
  RecordReceiveData(receive_data_time_, receive_data_);
}

size_t RoundKernel::total_client_num() const { return total_client_num_; }

size_t RoundKernel::accept_client_num() const { return accept_client_num_; }

size_t RoundKernel::reject_client_num() const { return total_client_num_ - accept_client_num_; }

void RoundKernel::InitClientVisitedNum() {
  total_client_num_ = 0;
  accept_client_num_ = 0;
}

void RoundKernel::InitClientUploadLoss() { upload_loss_ = 0.0f; }

void RoundKernel::UpdateClientUploadLoss(const float upload_loss) { upload_loss_ = upload_loss_ + upload_loss; }

float RoundKernel::upload_loss() const { return upload_loss_; }

void RoundKernel::CalculateSendData(size_t send_len) {
  uint64_t second_time_stamp =
    std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  if (send_data_time_ == 0) {
    send_data_ = send_len;
    send_data_time_ = second_time_stamp;
    return;
  }
  if (second_time_stamp == send_data_time_) {
    send_data_ += send_len;
  } else {
    RecordSendData(send_data_time_, send_data_);
    send_data_time_ = second_time_stamp;
    send_data_ = send_len;
  }
}

void RoundKernel::CalculateReceiveData(size_t receive_len) {
  uint64_t second_time_stamp =
    std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  if (receive_data_time_ == 0) {
    receive_data_time_ = second_time_stamp;
    receive_data_ = receive_len;
    return;
  }
  if (second_time_stamp == receive_data_time_) {
    receive_data_ += receive_len;
  } else {
    RecordReceiveData(receive_data_time_, receive_data_);
    receive_data_time_ = second_time_stamp;
    receive_data_ = receive_len;
  }
}

void RoundKernel::RecordSendData(uint64_t time_stamp_second, size_t send_data) {
  std::lock_guard<std::mutex> lock(send_data_rate_mutex_);
  send_data_and_time_[time_stamp_second] = send_data;
}

void RoundKernel::RecordReceiveData(uint64_t time_stamp_second, size_t receive_data) {
  std::lock_guard<std::mutex> lock(receive_data_rate_mutex_);
  receive_data_and_time_[time_stamp_second] = receive_data;
}

std::map<uint64_t, size_t> RoundKernel::GetSendData() {
  std::lock_guard<std::mutex> lock(send_data_rate_mutex_);
  return send_data_and_time_;
}

std::map<uint64_t, size_t> RoundKernel::GetReceiveData() {
  std::lock_guard<std::mutex> lock(receive_data_rate_mutex_);
  return receive_data_and_time_;
}

void RoundKernel::ClearData() {
  std::lock_guard<std::mutex> lock(send_data_rate_mutex_);
  std::lock_guard<std::mutex> lock2(receive_data_rate_mutex_);
  send_data_and_time_.clear();
  receive_data_and_time_.clear();
}
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
