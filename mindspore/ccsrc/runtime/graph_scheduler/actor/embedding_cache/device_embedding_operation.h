/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_ACTOR_EMBEDDING_CACHE_DEVICE_EMBEDDING_OPERATION_H_
#define MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_ACTOR_EMBEDDING_CACHE_DEVICE_EMBEDDING_OPERATION_H_

#include "distributed/embedding_cache/embedding_cache_utils.h"

namespace mindspore {
namespace runtime {
using distributed::HashTableInfo;

class DeviceEmbeddingOperation {
 public:
  virtual ~DeviceEmbeddingOperation() = default;

  // Analyze the hit/miss info of the local host cache and device cache, and calculate the swapping and
  // mapping information of the missing feature id that needs to be inserted into the cache.
  virtual bool CountCacheMissIds(const int *batch_ids, const size_t batch_ids_len) = 0;

  // Pull missing embeddings on the device cache from the local host.
  virtual bool PullCacheFromLocalHostToDevice(const HashTableInfo &hash_info) = 0;

  // Push non-hotspot embeddings on the device cache to the local host cache.
  virtual bool PushCacheFromDeviceToLocalHost(const HashTableInfo &hash_info) = 0;

 private:
  DeviceEmbeddingOperation() = default;

  DISABLE_COPY_AND_ASSIGN(DeviceEmbeddingOperation);
};
}  // namespace runtime
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_ACTOR_EMBEDDING_CACHE_DEVICE_EMBEDDING_OPERATION_H_
