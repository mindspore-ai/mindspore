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

#include <memory>
#include <utility>
#include <set>
#include "runtime/hardware/device_context.h"
#include "distributed/embedding_cache/embedding_cache_utils.h"
#include "runtime/graph_scheduler/actor/embedding_cache/embedding_cache_prefetch_actor.h"

namespace mindspore {
namespace runtime {
using device::DeviceContext;
using distributed::EmbeddingCacheStatisticsInfo;
using distributed::EmbeddingDeviceCache;
using distributed::EmbeddingHostCache;
using distributed::HashTableInfo;

// Maximum number of threads for concurrent accelerated cache processing.
constexpr size_t kMaxThreadNum = 16;
// Maximum number of feature ids processed per thread.
constexpr size_t kMaxIdsPerThread = 10000;

class DeviceEmbeddingOperation {
 public:
  DeviceEmbeddingOperation(EmbeddingCachePrefetchActor *actor, device::DeviceContext *device_context,
                           std::shared_ptr<EmbeddingDeviceCache> emb_dev_cache,
                           std::shared_ptr<EmbeddingHostCache> emb_host_cache,
                           const std::pair<int, int> &local_embedding_slice_bounds,
                           const std::pair<int, int> &local_device_cache_bounds, CNodePtr embedding_cache_lookup_node,
                           CNodePtr embedding_cache_update_node, EmbeddingCacheStatisticsInfo *statistics_info,
                           const size_t &stream_id, std::atomic_bool *running)
      : actor_(actor),
        device_context_(device_context),
        embedding_device_cache_(emb_dev_cache),
        embedding_host_cache_(emb_host_cache),
        local_embedding_slice_bounds_(local_embedding_slice_bounds),
        local_device_cache_bounds_(local_device_cache_bounds),
        embedding_cache_lookup_node_(embedding_cache_lookup_node),
        embedding_cache_update_node_(embedding_cache_update_node),
        statistics_info_(statistics_info),
        stream_id_(stream_id),
        running_(running) {}

  virtual ~DeviceEmbeddingOperation() = default;

  // Do lookup embedding table operation.
  virtual void LookupEmbeddingTable(size_t indices_num, size_t outer_dim_size, size_t first_dim_size,
                                    const float *input_addr, const int *indices_addr, float *output_addr) = 0;

  // Analyze the hit/miss info of the local host cache and device cache, and calculate the swapping and
  // mapping information of the missing feature id that needs to be inserted into the cache.
  virtual bool CountCacheMissIds(int *batch_ids, const size_t batch_ids_len, size_t data_step,
                                 size_t graph_running_step, bool *device_cache_need_wait_graph,
                                 bool *host_cache_need_wait_graph) = 0;

  // Pull missing embeddings on the device cache from the local host.
  virtual bool PullCacheFromLocalHostToDevice(const HashTableInfo &hash_info) = 0;

  // Push non-hotspot embeddings on the device cache to the local host cache.
  virtual bool PushCacheFromDeviceToLocalHost(const HashTableInfo &hash_info) = 0;

 protected:
  // Look up feature weights on Device Embedding Cache:
  // 1. Update the shape of parameter node.
  // 2. Infer shape for embedding cache look up kernel(operator name: 'Gather').
  // 3. Launch embedding cache look up kernel.
  virtual bool LookupDeviceCache(void *indices, void *embedding_cache, size_t indices_num, size_t cache_size,
                                 size_t embedding_size, void *outputs) = 0;

  // Parse the hit and swap out to device cache information of the currently preprocessed id of the local host cache.
  bool ParseHostDataHostToDevice(int id, size_t data_step, size_t graph_running_step, bool *host_cache_need_wait_graph);

  // Parse the swap in information from device cache of the currently preprocessed id of the local host cache.
  bool ParseHostDataDeviceToHost(size_t data_step, size_t graph_running_step, bool *host_cache_need_wait_graph);

  // The actor which owns this operation.
  EmbeddingCachePrefetchActor *actor_;

  // The device interface.
  device::DeviceContext *device_context_;

  // Record the public information of all device embedding cache tables, such as the mapping relationship of id to
  // index, the information that needs to be updated (swap in and swap out), etc.
  std::shared_ptr<EmbeddingDeviceCache> embedding_device_cache_{nullptr};

  // Record the public information of all local host embedding cache tables, such as the mapping relationship of id to
  // index, the information that needs to be updated (swap in and swap out), etc.
  std::shared_ptr<EmbeddingHostCache> embedding_host_cache_{nullptr};

  // Model parallelism is used between multiple workers, and local_embedding_slice_bounds_ records the feature range
  // corresponding to the embedding table slice of the process.
  std::pair<int, int> local_embedding_slice_bounds_;

  // Model parallelism is used between multiple workers, and local_device_cache_bounds_ records the local device cache
  // range corresponding to the embedding table slice of the process.
  std::pair<int, int> local_device_cache_bounds_;

  // The embedding cache look up kernel node(operator name: 'Gather').
  CNodePtr embedding_cache_lookup_node_{nullptr};

  // The embedding cache update kernel node(operator name: 'ScatterUpdate').
  CNodePtr embedding_cache_update_node_{nullptr};

  // The feature ids that have been initialized already.
  std::set<int> initialized_ids_;

  // Statistics on the cache hit rate of the host and device and the information used to update cache.
  EmbeddingCacheStatisticsInfo *statistics_info_;

  // The device stream used to async memcpy operators and launch device kernels, such as embedding cache look up and
  // update kernel.
  size_t stream_id_{0};

  // The flag which indicates whether this actor is running to prefetch cache.
  std::atomic_bool *running_;

 private:
  DISABLE_COPY_AND_ASSIGN(DeviceEmbeddingOperation);
};
}  // namespace runtime
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_ACTOR_EMBEDDING_CACHE_DEVICE_EMBEDDING_OPERATION_H_
