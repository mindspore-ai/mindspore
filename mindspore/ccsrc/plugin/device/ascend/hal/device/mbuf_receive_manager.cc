/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/hal/device/mbuf_receive_manager.h"
#include <ctime>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include "include/common/utils/utils.h"
#include "ir/tensor.h"
#include "plugin/device/ascend/hal/device/ascend_data_queue.h"
#include "utils/file_utils.h"
#include "utils/ms_context.h"
#include "utils/shape_utils.h"

namespace mindspore::device::ascend {

namespace {
// Data may not be received when the process exits; reserve a timeout period.
constexpr std::chrono::milliseconds stop_time_out{100};
}  // namespace

MbufDataHandler::MbufDataHandler(MbufFuncType func, uint32_t device_id, string channel_name, size_t capacity,
                                 int32_t timeout)
    : func_(func), device_id_(device_id), channel_name_(channel_name), capacity_(capacity), timeout_(timeout) {
  MS_LOG(INFO) << "Channel " << channel_name_ << " begins the construction process.";
  acl_handle_ = acltdtCreateChannelWithCapacity(device_id_, channel_name_.c_str(), capacity_);
  if (acl_handle_ == nullptr) {
    MS_LOG(ERROR) << "Channel " << channel_name_ << "failed to create mbuf Channel.";
    return;
  }
  future_ = promise_.get_future();
  thread_ = std::thread(&MbufDataHandler::HandleData, this);
}

MbufDataHandler::~MbufDataHandler() {
  MS_LOG(INFO) << "Channel " << channel_name_ << " begins the destruction process.";
  // Stop the child thread from receiving data
  stop_receive_.store(true, std::memory_order_acq_rel);
  if (thread_.joinable()) {
    thread_.join();
  }
  MbufReceiveError thread_status = future_.get();
  if (thread_status != MbufReceiveError::Success) {
    MS_LOG(ERROR) << "Channel " << channel_name_ << " having a problem during operation. Error code: " << thread_status;
  }
  if (acl_handle_) {
    aclError status = acltdtDestroyChannel(acl_handle_);
    if (status != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Channel " << channel_name_ << " failed destroy acl channel. Error code: " << status;
      return;
    }
  } else {
    MS_LOG(WARNING) << "Channel " << channel_name_ << ": acl handle has been destroyed.";
  }
}

bool MbufDataHandler::ReceiveAndProcessData(const ScopeAclTdtDataset &scope_acl_dataset) {
  aclError status = acltdtReceiveTensor(acl_handle_, scope_acl_dataset.Get(), timeout_);
  if (status == ACL_ERROR_RT_QUEUE_EMPTY) {
    MS_LOG(DEBUG) << "Channel " << channel_name_ << " didn't receive any data during " << timeout_ << "ms.";
  } else if (status != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Channel " << channel_name_ << " failed to receive tensor. Error code is " << status;
    promise_.set_value(MbufReceiveError::AclError);
    return false;
  }

  if (status == ACL_SUCCESS) {
    func_(scope_acl_dataset.Get());
  }
  return true;
}

bool MbufDataHandler::QueryChannelSize(size_t *size) {
  aclError status = acltdtQueryChannelSize(acl_handle_, size);
  if (status != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Channel " << channel_name_ << " failed to QueryChannelSize. Error code is " << status;
    promise_.set_value(MbufReceiveError::AclError);
    return false;
  }
  return true;
}

void MbufDataHandler::HandleData() {
  MS_LOG(INFO) << "Channel " << channel_name_ << " starts executing HandleData.";
  ScopeAclTdtDataset scope_acl_dataset;
  if (scope_acl_dataset.Get() == nullptr) {
    MS_LOG(ERROR) << "Channel " << channel_name_ << " failed to create aclDateaset.";
    promise_.set_value(MbufReceiveError::AclError);
    return;
  }

  while (!stop_receive_.load()) {
    if (!ReceiveAndProcessData(scope_acl_dataset)) {
      return;
    }
  }

  size_t channel_size = 0;
  auto start = std::chrono::high_resolution_clock::now();
  if (!QueryChannelSize(&channel_size)) {
    return;
  }
  while (channel_size > 0 && std::chrono::high_resolution_clock::now() - start < stop_time_out) {
    if (!ReceiveAndProcessData(scope_acl_dataset)) {
      return;
    }
    if (!QueryChannelSize(&channel_size)) {
      return;
    }
  }
  if (channel_size > 0) {
    MS_LOG(ERROR) << "Channel " << channel_name_ << " has stopped receiving data, and " << channel_size
                  << " pieces of data have not been received.";
    promise_.set_value(MbufReceiveError::Timeout);
    return;
  }
  promise_.set_value(MbufReceiveError::Success);
}

}  // namespace mindspore::device::ascend
