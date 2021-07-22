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

#include "fl/server/kernel/round/round_kernel.h"
#include <mutex>
#include <queue>
#include <chrono>
#include <thread>
#include <utility>
#include <string>
#include <vector>

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
RoundKernel::RoundKernel() : name_(""), current_count_(0), required_count_(0), error_reason_(""), running_(true) {
  release_thread_ = std::thread([&]() {
    while (running_.load()) {
      std::unique_lock<std::mutex> release_lock(release_mtx_);
      // Detect whether there's any data needs to be released every 100 milliseconds.
      if (heap_data_to_release_.empty()) {
        release_lock.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(kReleaseDuration));
        continue;
      }

      AddressPtr addr_ptr = heap_data_to_release_.front();
      heap_data_to_release_.pop();
      release_lock.unlock();

      std::unique_lock<std::mutex> heap_data_lock(heap_data_mtx_);
      if (heap_data_.count(addr_ptr) == 0) {
        MS_LOG(ERROR) << "The data is not stored.";
        continue;
      }
      // Manually release unique_ptr data.
      heap_data_[addr_ptr].reset(nullptr);
      (void)heap_data_.erase(heap_data_.find(addr_ptr));
    }
  });
}

RoundKernel::~RoundKernel() {
  running_ = false;
  if (release_thread_.joinable()) {
    release_thread_.join();
  }
}

void RoundKernel::OnFirstCountEvent(const std::shared_ptr<ps::core::MessageHandler> &) { return; }

void RoundKernel::OnLastCountEvent(const std::shared_ptr<ps::core::MessageHandler> &) { return; }

void RoundKernel::StopTimer() const {
  if (stop_timer_cb_) {
    stop_timer_cb_();
  }
  return;
}

void RoundKernel::FinishIteration() const {
  if (finish_iteration_cb_) {
    finish_iteration_cb_(true, "");
  }
  return;
}

void RoundKernel::Release(const AddressPtr &addr_ptr) {
  if (addr_ptr == nullptr) {
    MS_LOG(ERROR) << "Data to be released is empty.";
    return;
  }
  std::unique_lock<std::mutex> lock(release_mtx_);
  heap_data_to_release_.push(addr_ptr);
  return;
}

void RoundKernel::set_name(const std::string &name) { name_ = name; }

void RoundKernel::set_stop_timer_cb(const StopTimerCb &timer_stopper) { stop_timer_cb_ = timer_stopper; }

void RoundKernel::set_finish_iteration_cb(const FinishIterCb &finish_iteration_cb) {
  finish_iteration_cb_ = finish_iteration_cb;
}

void RoundKernel::GenerateOutput(const std::vector<AddressPtr> &outputs, const void *data, size_t len) {
  if (data == nullptr) {
    MS_LOG(ERROR) << "The data is nullptr.";
    return;
  }

  if (outputs.empty()) {
    MS_LOG(ERROR) << "Generating output failed. Outputs size is empty.";
    return;
  }

  std::unique_ptr<unsigned char[]> output_data = std::make_unique<unsigned char[]>(len);
  if (output_data == nullptr) {
    MS_LOG(ERROR) << "Output data is nullptr.";
    return;
  }

  size_t dst_size = len;
  int ret = memcpy_s(output_data.get(), dst_size, data, len);
  if (ret != 0) {
    MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
    return;
  }
  outputs[0]->addr = output_data.get();
  outputs[0]->size = len;

  std::unique_lock<std::mutex> lock(heap_data_mtx_);
  (void)heap_data_.insert(std::make_pair(outputs[0], std::move(output_data)));
  return;
}
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
