/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "include/backend/distributed/ps/ps_cache/ps_data_prefetch.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ps {
const size_t kTimeoutLoopCount = 40;
const int64_t kLongestTimeToWait = 30;

PsDataPrefetch &PsDataPrefetch::GetInstance() {
  static PsDataPrefetch instance;
  return instance;
}

void PsDataPrefetch::CreateDataChannel(const std::string &channel_name, size_t step_num) {
  if (cache_enable_ == false) {
    return;
  }
  MS_LOG(INFO) << "PS cache creates data channel(channel name:" << channel_name << ", step num:" << step_num << ").";
  auto iter = ps_data_channel_map_.find(channel_name);
  if (iter != ps_data_channel_map_.end()) {
    MS_LOG(WARNING) << "The ps data channel already exists, channel name:" << channel_name;
    auto channel = iter->second;
    MS_ERROR_IF_NULL_WO_RET_VAL(channel);
    channel->set_step_num(step_num);
  } else {
    auto channel = std::make_shared<PsDataChannel>(channel_name, step_num);
    (void)ps_data_channel_map_.emplace(channel_name, channel);
  }
}

std::shared_ptr<PsDataChannel> PsDataPrefetch::ps_data_channel(const std::string &channel_name) const {
  auto iter = ps_data_channel_map_.find(channel_name);
  if (iter == ps_data_channel_map_.end()) {
    MS_LOG(ERROR) << "The ps data channel does not exist, channel name:" << channel_name;
    return nullptr;
  }
  return iter->second;
}

bool PsDataPrefetch::PrefetchData(const std::string &channel_name, void *data, const size_t data_size,
                                  const std::string &data_type) {
  if (cache_enable_ == false) {
    return true;
  }
  // In ps cache mode, input ids are from dataset and data type transmitted from minddata must be 'int32'
  const std::string supported_data_type = "int32";
  if (data_type != supported_data_type) {
    MS_LOG(ERROR) << "Parameter server cache mode need input id with data type[int32], but got[" << data_type << "]";
    invalid_data_type_ = true;
    return false;
  }
  if (data == nullptr) {
    MS_LOG(WARNING) << "No data prefetch.";
    return true;
  }

  if (!need_wait_) {
    return true;
  }

  auto channel = ps_data_channel(channel_name);
  MS_ERROR_IF_NULL(channel);
  channel->set_data(data, data_size);
  std::unique_lock<std::mutex> locker(data_mutex_);
  data_ready_ = true;
  data_process_.notify_one();

  for (size_t i = 0; i < kTimeoutLoopCount; ++i) {
    if (data_prefetch_.wait_for(locker, std::chrono::seconds(kLongestTimeToWait),
                                [this] { return data_ready_ == false || need_wait_ == false; })) {
      return true;
    } else {
      MS_LOG(INFO) << "Waiting for ps data process, channel name:" << channel_name << "...(" << i << " / "
                   << kTimeoutLoopCount << ")";
    }
  }
  MS_LOG(ERROR) << "Ps cache data process timeout, suggest to enlarge the cache size.";
  return false;
}

bool PsDataPrefetch::FinalizeData(const std::string &channel_name) {
  if (cache_enable_ == false) {
    return true;
  }
  auto channel = ps_data_channel(channel_name);
  MS_ERROR_IF_NULL(channel);
  channel->ResetData();
  std::unique_lock<std::mutex> locker(data_mutex_);
  data_ready_ = false;
  data_prefetch_.notify_one();
  if (!need_wait_) {
    return true;
  }

  for (size_t i = 0; i < kTimeoutLoopCount; ++i) {
    if (data_process_.wait_for(locker, std::chrono::seconds(kLongestTimeToWait),
                               [this] { return data_ready_ == true || need_wait_ == false; })) {
      return true;
    } else {
      MS_LOG(INFO) << "Waiting for ps data prefetch, channel name:" << channel_name << "...(" << i << " / "
                   << kTimeoutLoopCount << ")";
    }
  }
  MS_LOG(ERROR) << "Ps cache data prefetch timeout.";
  return false;
}

bool PsDataPrefetch::QueryData(const std::string &channel_name, void **data_ptr) const {
  if (invalid_data_type_) {
    return false;
  }
  if (data_ptr == nullptr) {
    return false;
  }
  auto channel = ps_data_channel(channel_name);
  if (channel == nullptr) {
    *data_ptr = nullptr;
    return true;
  }
  *data_ptr = const_cast<void *>(channel->data());
  return true;
}

size_t PsDataPrefetch::data_size(const std::string &channel_name) const {
  auto channel = ps_data_channel(channel_name);
  if (channel == nullptr) {
    return 0;
  }
  return channel->data_size();
}

void PsDataPrefetch::NotifyFinalize() {
  std::lock_guard<std::mutex> lock(finalize_mutex_);
  if (!need_wait_) {
    return;
  }

  need_wait_ = false;
  WakeAllChannel();
  data_prefetch_.notify_one();
  data_process_.notify_one();
}

bool PsDataPrefetch::TryWakeChannel(const std::string &channel_name) const {
  auto channel = ps_data_channel(channel_name);
  if (channel == nullptr) {
    return false;
  }
  channel->TryWakeChannel();
  return true;
}

void PsDataPrefetch::WakeAllChannel() const {
  for (auto iter = ps_data_channel_map_.begin(); iter != ps_data_channel_map_.end(); ++iter) {
    auto channel = iter->second;
    if (channel == nullptr) {
      return;
    }
    channel->TryWakeChannel(true);
  }
}
}  // namespace ps
}  // namespace mindspore
