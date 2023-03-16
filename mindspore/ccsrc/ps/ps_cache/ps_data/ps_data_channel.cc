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

#include "include/backend/distributed/ps/ps_cache/ps_data_channel.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ps {
void PsDataChannel::TryLockChannel() {
  // The prefetch order of data needs to be consistent with the graph execution order.
  // Example: if graph execution order is graph1 --> graph2 --> graph1 -->graph2,
  // then the data prefetch order needs be channel1 --> channel 2 --> channel1 --> channel2.
  if ((current_data_step_ != 0) && (current_data_step_ % step_num_ == 0)) {
    MS_LOG(INFO) << "Lock channel:" << channel_name_;
    std::unique_lock<std::mutex> locker(channel_mutex_);
    channel_.wait(locker, [this] { return channel_open_; });
    channel_open_ = false;
  }
  current_data_step_++;
}

void PsDataChannel::TryWakeChannel(bool force_wake) {
  if (force_wake || ((current_graph_step_ != 0) && (current_graph_step_ % step_num_ == 0))) {
    MS_LOG(INFO) << "Wake up channel:" << channel_name_;
    std::lock_guard<std::mutex> locker(channel_mutex_);
    channel_open_ = true;
    channel_.notify_one();
  }
  current_graph_step_++;
}

void PsDataChannel::set_data(const void *data, const size_t data_size) {
  MS_EXCEPTION_IF_NULL(data);
  TryLockChannel();
  data_ = const_cast<void *>(data);
  data_size_ = data_size;
}
}  // namespace ps
}  // namespace mindspore
