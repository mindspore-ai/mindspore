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

#include <vector>
#include <memory>
#include <utility>
#include <set>
#include "runtime/hardware/device_context.h"
#include "distributed/embedding_cache/embedding_cache_utils.h"
#include "runtime/graph_scheduler/actor/embedding_cache/embedding_cache_prefetch_actor.h"

namespace mindspore {
namespace runtime {
// One and two dimensional shape placeholder.
const ShapeVector kOneDimensionalShape = {1};
const ShapeVector kTwoDimensionalShape = {1, 1};

const size_t kInputIndexZero = 0;
const size_t kInputIndexOne = 1;
const size_t kInputIndexTwo = 2;

using device::DeviceContext;
using distributed::EmbeddingCacheStatisticsInfo;
using distributed::EmbeddingDeviceCache;
using distributed::EmbeddingHostCache;
using distributed::HashTableInfo;
using mindspore::session::KernelGraph;

// Maximum number of threads for concurrent accelerated cache processing.
constexpr size_t kMaxThreadNum = 16;
// Maximum number of feature ids processed per thread.
constexpr size_t kMaxIdsPerThread = 10000;

class DeviceEmbeddingOperation {
 public:
  DeviceEmbeddingOperation(EmbeddingCachePrefetchActor *actor, device::DeviceContext *device_context,
                           const std::pair<int, int> &local_embedding_slice_bounds,
                           const std::pair<int, int> &local_device_cache_bounds,
                           EmbeddingCacheStatisticsInfo *statistics_info, const size_t &stream_id)
      : actor_(actor),
        device_context_(device_context),
        local_embedding_slice_bounds_(local_embedding_slice_bounds),
        local_device_cache_bounds_(local_device_cache_bounds),
        statistics_info_(statistics_info),
        stream_id_(stream_id) {}

  virtual ~DeviceEmbeddingOperation() = default;

  virtual bool Initialize();

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
  // Parse the hit and swap out to device cache information of the currently preprocessed id of the local host cache.
  bool ParseHostDataHostToDevice(int id, size_t data_step, size_t graph_running_step, bool *host_cache_need_wait_graph);

  // Parse the swap in information from device cache of the currently preprocessed id of the local host cache.
  bool ParseHostDataDeviceToHost(size_t data_step, size_t graph_running_step, bool *host_cache_need_wait_graph);

  // Build a CNode of embedding cache look up kernel, which is used to look up local device
  // embedding cache.
  virtual void BuildEmbeddingCacheLookupKernel() = 0;

  // Build a CNode of embedding cache update kernel, which is used to update local
  // device embedding cache.
  virtual void BuildEmbeddingCacheUpdateKernel() = 0;

  // Async copy host memory to device.
  static bool MemcpyHostToDeviceAsync(void *dst, const void *src, size_t size, const DeviceContext *device_context,
                                      size_t stream_id);

  // Async copy device memory to host.
  static bool MemcpyDeviceToHostAsync(void *dst, const void *src, size_t size, const DeviceContext *device_context,
                                      size_t stream_id);

  static ParameterPtr NewParameter(const KernelGraphPtr &graph, TypePtr type, const ShapeVector &shape);

  static ValueNodePtr NewValueNode(int64_t value, const DeviceContext *device_context, size_t stream_id);

  static bool InferOpShape(const CNodePtr &kernel);

  // The actor which owns this operation.
  EmbeddingCachePrefetchActor *actor_;

  // The device interface.
  device::DeviceContext *device_context_;

  // Model parallelism is used between multiple workers, and local_embedding_slice_bounds_ records the feature range
  // corresponding to the embedding table slice of the process.
  std::pair<int, int> local_embedding_slice_bounds_;

  // Model parallelism is used between multiple workers, and local_device_cache_bounds_ records the local device cache
  // range corresponding to the embedding table slice of the process.
  std::pair<int, int> local_device_cache_bounds_;

  // The embedding cache look up kernel node(operator name: 'Gather' for dense mode and 'MapTensorGet' for sparse mode).
  CNodePtr embedding_cache_lookup_node_{nullptr};

  // The embedding cache update kernel node(operator name: 'ScatterUpdate' for dense mode and 'MapTensorPut' for sparse
  // mode).
  CNodePtr embedding_cache_update_node_{nullptr};

  // The feature ids that have been initialized already.
  std::set<int> initialized_ids_;

  // Statistics on the cache hit rate of the host and device and the information used to update cache.
  EmbeddingCacheStatisticsInfo *statistics_info_;

  // Cache embeding cache ops kernel graphs.
  std::vector<KernelGraphPtr> embedding_cache_graphs_;

  // The device stream used to async memcpy operators and launch device kernels, such as embedding cache look up and
  // update kernel.
  size_t stream_id_{0};

 private:
  DISABLE_COPY_AND_ASSIGN(DeviceEmbeddingOperation);
};
}  // namespace runtime
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_ACTOR_EMBEDDING_CACHE_DEVICE_EMBEDDING_OPERATION_H_
