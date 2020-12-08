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

#include "ps/ps_cache/ps_data/ps_data_prefetch.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ps {
void PsDataPrefetch::CreateDataChannel(const std::string &channel_name, size_t step_num) {
  if (cache_enable_ == false) {
    return;
  }
  MS_LOG(INFO) << "PS cache creates data channel(channel name:" << channel_name << ", step num:" << step_num << ").";
  auto iter = ps_data_channel_map_.find(channel_name);
  if (iter != ps_data_channel_map_.end()) {
    MS_LOG(WARNING) << "The ps data channel already exists, channel name:" << channel_name;
    auto channel = iter->second;
    MS_EXCEPTION_IF_NULL(channel);
    channel->set_step_num(step_num);
  } else {
    auto channel = std::make_shared<PsDataChannel>(channel_name, step_num);
    MS_EXCEPTION_IF_NULL(channel);
    (void)ps_data_channel_map_.emplace(channel_name, channel);
  }
}

std::shared_ptr<PsDataChannel> PsDataPrefetch::ps_data_channel(const std::string &channel_name) const {
  auto iter = ps_data_channel_map_.find(channel_name);
  if (iter == ps_data_channel_map_.end()) {
    MS_LOG(EXCEPTION) << "The ps data channel does not exist, channel name:" << channel_name;
  }
  return iter->second;
}

void PsDataPrefetch::PrefetchData(const std::string &channel_name, void *data, const size_t data_size) {
  if (cache_enable_ == false) {
    return;
  }
  if (data == nullptr) {
    MS_LOG(WARNING) << "No data prefetch.";
    return;
  }
  auto channel = ps_data_channel(channel_name);
  MS_EXCEPTION_IF_NULL(channel);
  channel->set_data(data, data_size);
  std::unique_lock<std::mutex> locker(data_mutex_);
  data_ready_ = true;
  data_process_.notify_one();
  for (int i = 0; i < 10; i++) {
    if (data_prefetch_.wait_for(locker, std::chrono::seconds(30), [this] { return data_ready_ == false; })) {
      return;
    } else {
      MS_LOG(INFO) << "Waiting for ps data process, channel name:" << channel_name << "...(" << i << " / 10)";
    }
  }
  MS_LOG(EXCEPTION) << "Ps cache data process timeout, suggest to enlarge the cache size.";
}

void PsDataPrefetch::FinalizeData(const std::string &channel_name) {
  if (cache_enable_ == false) {
    return;
  }
  auto channel = ps_data_channel(channel_name);
  MS_EXCEPTION_IF_NULL(channel);
  channel->ResetData();
  std::unique_lock<std::mutex> locker(data_mutex_);
  data_ready_ = false;
  data_prefetch_.notify_one();
  for (int i = 0; i < 10; i++) {
    if (data_process_.wait_for(locker, std::chrono::seconds(30), [this] { return data_ready_ == true; })) {
      return;
    } else {
      MS_LOG(INFO) << "Waiting for ps data prefetch, channel name:" << channel_name << "...(" << i << " / 10)";
    }
  }
  MS_LOG(EXCEPTION) << "Ps cache data prefetch timeout.";
}

void *PsDataPrefetch::data(const std::string &channel_name) const {
  auto channel = ps_data_channel(channel_name);
  MS_EXCEPTION_IF_NULL(channel);
  return channel->data();
}

size_t PsDataPrefetch::data_size(const std::string &channel_name) const {
  auto channel = ps_data_channel(channel_name);
  MS_EXCEPTION_IF_NULL(channel);
  return channel->data_size();
}

void PsDataPrefetch::TryWakeChannel(const std::string &channel_name) {
  auto channel = ps_data_channel(channel_name);
  MS_EXCEPTION_IF_NULL(channel);
  channel->TryWakeChannel();
}
}  // namespace ps
}  // namespace mindspore
