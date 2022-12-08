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

#include "runtime/graph_scheduler/actor/embedding_cache/device_sparse_embedding_operation.h"

namespace mindspore {
namespace runtime {
void LookupEmbeddingTable(size_t indices_num, size_t outer_dim_size, size_t first_dim_size, const float *input_addr,
                          const int *indices_addr, float *output_addr) {}

bool DeviceSparseEmbeddingOperation::CountCacheMissIds(int *batch_ids, const size_t batch_ids_len, size_t data_step,
                                                       size_t graph_running_step, bool *device_cache_need_wait_graph,
                                                       bool *host_cache_need_wait_graph) {
  return true;
}

bool DeviceSparseEmbeddingOperation::PullCacheFromLocalHostToDevice(const HashTableInfo &hash_info) { return true; }

bool DeviceSparseEmbeddingOperation::PushCacheFromDeviceToLocalHost(const HashTableInfo &hash_info) { return true; }
}  // namespace runtime
}  // namespace mindspore
