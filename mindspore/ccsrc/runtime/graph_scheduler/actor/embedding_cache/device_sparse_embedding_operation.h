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

#ifndef MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_ACTOR_EMBEDDING_CACHE_DEVICE_SPARSE_EMBEDDING_OPERATION_H_
#define MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_ACTOR_EMBEDDING_CACHE_DEVICE_SPARSE_EMBEDDING_OPERATION_H_

#include "runtime/graph_scheduler/actor/embedding_cache/device_embedding_operation.h"

namespace mindspore {
namespace runtime {
class DeviceSparseEmbeddingOperation : DeviceEmbeddingOperation {
 public:
  DeviceSparseEmbeddingOperation() = default;
  ~DeviceSparseEmbeddingOperation() override = default;

  bool CountCacheMissIds(const int *batch_ids, const size_t batch_ids_len) override;

  bool PullCacheFromLocalHostToDevice(const HashTableInfo &hash_info) override;
  bool PushCacheFromDeviceToLocalHost(const HashTableInfo &hash_info) override;

 private:
  DISABLE_COPY_AND_ASSIGN(DeviceSparseEmbeddingOperation);
};
}  // namespace runtime
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_ACTOR_EMBEDDING_CACHE_DEVICE_SPARSE_EMBEDDING_OPERATION_H_
