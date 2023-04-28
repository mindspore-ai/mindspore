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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_DATA_QUEUE_MANAGER_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_DATA_QUEUE_MANAGER_H_

#include <memory>
#include <utility>
#include <string>
#include "include/backend/distributed/embedding_cache/blocking_queue.h"
#include "include/backend/distributed/embedding_cache/embedding_cache_utils.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace distributed {
using IdsDataQueue = BlockingQueue<IdDataInfo>;
using IndicesDataQueue = BlockingQueue<IndexDataInfo>;
using IdsAndIndicesDataQueuePair = std::pair<std::shared_ptr<IdsDataQueue>, std::shared_ptr<IndicesDataQueue>>;

// This class is used to manage the Cache prefetch queue in the Embedding Cache mode.
class BACKEND_EXPORT DataQueueManager {
 public:
  static DataQueueManager &GetInstance();

  void CreateDataQueue(const std::string &channel_name, size_t sink_size, size_t capacity);

  const IdsAndIndicesDataQueuePair &GetDataQueue(const std::string &channel_name) const;

  size_t GetSinkSize(const std::string &channel_name) const;

  void CloseAllQueues();

  bool IsClosed() { return closed_.load(); }

 private:
  DataQueueManager() = default;
  ~DataQueueManager() = default;
  DISABLE_COPY_AND_ASSIGN(DataQueueManager);

  mindspore::HashMap<std::string, IdsAndIndicesDataQueuePair> channels_to_queues_;

  mindspore::HashMap<std::string, size_t> channels_to_sink_sizes_;

  std::atomic_bool closed_{false};

  std::mutex mtx_;
};
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_DATA_QUEUE_MANAGER_H_
