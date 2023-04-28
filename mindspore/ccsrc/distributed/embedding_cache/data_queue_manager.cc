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

#include "include/backend/distributed/embedding_cache/data_queue_manager.h"

namespace mindspore {
namespace distributed {
DataQueueManager &DataQueueManager::GetInstance() {
  static DataQueueManager instance;
  return instance;
}

void DataQueueManager::CreateDataQueue(const std::string &channel_name, size_t sink_size, size_t capacity) {
  if (channels_to_queues_.find(channel_name) != channels_to_queues_.end()) {
    return;
  }

  std::unique_lock<std::mutex> lock(mtx_);
  auto ids_data_queue = std::make_shared<BlockingQueue<IdDataInfo>>(capacity);
  auto indices_data_queue = std::make_shared<BlockingQueue<IndexDataInfo>>(capacity);
  (void)channels_to_queues_.emplace(channel_name, std::make_pair(ids_data_queue, indices_data_queue));

  (void)channels_to_sink_sizes_.emplace(channel_name, sink_size);
}

const IdsAndIndicesDataQueuePair &DataQueueManager::GetDataQueue(const std::string &channel_name) const {
  const auto &iter = channels_to_queues_.find(channel_name);
  if (iter != channels_to_queues_.end()) {
    return iter->second;
  }

  MS_LOG(EXCEPTION) << "Can not find data queue for channel: " << channel_name;
}

size_t DataQueueManager::GetSinkSize(const std::string &channel_name) const {
  const auto &iter = channels_to_sink_sizes_.find(channel_name);
  if (iter != channels_to_sink_sizes_.end()) {
    return iter->second;
  }

  MS_LOG(EXCEPTION) << "Can not find sink sizes for channel: " << channel_name;
}

void DataQueueManager::CloseAllQueues() {
  std::unique_lock<std::mutex> lock(mtx_);
  if (closed_) {
    return;
  }

  closed_ = true;
  for (auto &item : channels_to_queues_) {
    auto &queue_pair = item.second;
    queue_pair.first->Close();
    queue_pair.second->Close();
  }
}
}  // namespace distributed
}  // namespace mindspore
