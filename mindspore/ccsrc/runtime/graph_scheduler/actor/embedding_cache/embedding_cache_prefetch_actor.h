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

#ifndef MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_ACTOR_EMBEDDING_CACHE_EMBEDDING_CACHE_PREFETCH_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_ACTOR_EMBEDDING_CACHE_EMBEDDING_CACHE_PREFETCH_ACTOR_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "runtime/graph_scheduler/actor/actor_common.h"
#include "runtime/graph_scheduler/actor/rpc/send_actor.h"
#include "runtime/graph_scheduler/actor/rpc/recv_actor.h"
#include "ir/anf.h"
#include "backend/common/session/kernel_graph.h"

// Note: After the code in ps/ps_cache are removed into runtime/addons/embedding_cache/,
// the follow include file and using declaration of ps will be removed.
#include "ps/ps_cache/ps_data/ps_data_prefetch.h"
#include "ps/ps_cache/embedding_hash_map.h"
#include "ps/ps_cache/ps_cache_manager.h"
#include "ps/ps_context.h"
using mindspore::ps::EmbeddingDeviceCache;
using mindspore::ps::EmbeddingHostCache;
using mindspore::ps::HashTableInfo;
using mindspore::ps::INVALID_INDEX_VALUE;
using mindspore::ps::INVALID_STEP_VALUE;
using mindspore::ps::PsCacheStatisticsInfo;
using mindspore::ps::PSContext;
using mindspore::ps::PsDataPrefetch;

namespace mindspore {
namespace runtime {
// The EmbeddingCachePrefetchActor is used to cache large embedding table scenarios. The cache level is: Device
// Cache->Local Host Cache->Remote Cache. This Actor is used to perform Local and Device Cache hit analysis and cache
// prefetching (the feature weights corresponding to the ids of subsequent batches are assigned in advance Prefetching
// into the Device Cache, so that it is pipelined with the calculation on the Device side), cache prefetching may
// involve RPC communication with the Server side.
class EmbeddingCachePrefetchActor : public ActorBase {
 public:
  explicit EmbeddingCachePrefetchActor(device::DeviceContext *device_context)
      : ActorBase("EmbeddingCachePrefetchActor"), device_context_(device_context) {}

  ~EmbeddingCachePrefetchActor() override = default;

  // Initialize embedding cache prefetch actor.
  // 1. Build and Link rpc actors between local cache and remote cache.
  // 2. Build network connection of rpc actors.
  void Initialize();

  // Perform local cache hit analysis, prefetch the feature vector corresponding to the next batch into the cache.
  void Run();

  // Finalize embedding cache prefetch actor and push latest embedding from local cache to remote cache.
  void Finalize();

 private:
  // When the device cache does not reach 100% hit, the cache needs to be updated, which involves cache insertion and
  // deletion. That is, push the non-hotspot embeddings on the local side to the remote, and pull the missing embeddings
  // on the local side from the remote.
  bool UpdateCache();

  // Push non-hotspot embeddings on local host cache to remote.
  bool PushCacheFromLocalHostToRemote(const HashTableInfo &hash_info);
  // Push non-hotspot embeddings on device cache to local host cache.
  bool PushCacheFromDeviceToLocalHost(const HashTableInfo &hash_info);
  // Pull missing embeddings on local cache from remote.
  bool PullCacheFromRemoteToLocalHost(const HashTableInfo &hash_info);
  // Pull missing embeddings on device cache from local host.
  bool PullCacheFromLocalHostToDevice(const HashTableInfo &hash_info);

  // Insert weights into the local host embedding cache.
  bool InsertLocalHostCache(size_t embedding_size, size_t insert_indices_size, const int *insert_indices,
                            const float *insert_data, float *hash_table_addr);
  // Lookup embeddings from local host embedding cache.
  bool LookupLocalHostCache(size_t embedding_size, size_t indices_num, const float *hash_table_addr,
                            const int *indices_addr, float *output_addr);
  // Do lookup embedding table operation.
  void LookupEmbeddingTable(size_t indices_num, size_t outer_dim_size, size_t first_dim_size, const float *input_addr,
                            const int *indices_addr, float *output_addr);

  // Lookup embedding from Remote and get embeddings via RPC.
  bool PullEembeddingsFromRemote(const int *ids, size_t ids_num, std::vector<float> *outputs);
  // Push the local embedding cache that requires evict to the remote.
  bool PushEmbeddingsToRemote(const int *ids, size_t ids_num, const float *embeddings, size_t embeddings_len);

  // Get the id range of each server's embedding table slice.
  void GetRemoteEmbeddingSliceBound();

  // In a multi-server scenario, the embeddings need to be segmented, and each server saves the embeddings of
  // different feature id ranges. Therefore, when the local side performs the push or pull embeddings operation, the
  // embeddings and ids need to be divided, and then communicate with the corresponding remote: Partition ids by
  // remote embedding slice bound and get unique ids.
  void PartitionIds(const int *ids, size_t ids_num, std::vector<std::vector<int>> *slice_ids_list);
  // Partition ids end embeddings by remote embedding slice bound.
  void PartitionIdsAndEmbeddings(const int *ids, size_t ids_num, const float *embeddings, size_t embeddings_len,
                                 std::vector<std::vector<int>> *slice_ids_list,
                                 std::vector<std::vector<float>> *slice_embeddings_list);

  // Send content to remote, such as ids or embeddings.
  bool SendToRemote(size_t server_rank_id, const void *keys, size_t keys_len, const void *values = nullptr,
                    size_t values_len = 0);
  // Wait response of remote and get return result.
  bool WaitRespFromRemote(size_t server_rank_id, std::vector<float> *outputs);
  // Retrieve embeddings by input ids order.
  bool RetrieveEmbeddings(const int *ids, size_t ids_num, const std::vector<std::vector<int>> &slice_ids_list,
                          const std::vector<std::vector<float>> &slice_embeddings_list, std::vector<float> *outputs);

  // The cache prefetch phase may involve RPC communication with the server, implemented through Send Actor and
  // Recv Actor.
  // Build rpc actors.
  void BuildRpcActors();
  // Link rpc actors by inter-process arrows.
  void LinkRpcActors();

  // Build a CNode of embedding cache look up kernel(operator name: 'Gather'), which is used to look up local device
  // embedding cache.
  void BuildEmbeddingCacheLookupKernel();
  // Build a CNode of embedding cache update kernel(operator name: 'ScatterUpdate'), which is used to update local
  // device embedding cache.
  void BuildEmbeddingCacheUpdateKernel();

  // Look up feature weights on Device Embedding Cache:
  // 1. Update the shape of parameter node.
  // 2. Infer shape for embedding cache look up kernel(operator name: 'Gather').
  // 3. Launch embedding cache look up kernel.
  bool LookupDeviceCache(void *indices, void *embedding_cache, size_t indices_num, size_t cache_size,
                         size_t embedding_size, void *outputs);

  // Update feature weights on Device Embedding Cache:
  // 1. Update the shape of parameter node.
  // 2. Infer shape for embedding cache update kernel(operator name: 'ScatterUpdate').
  // 3. Launch embedding cache update kernel.
  bool UpdateDeviceCache(void *indices, void *update_value, size_t indices_num, size_t cache_size,
                         size_t embedding_size, void *embedding_cache);

  // Record Send Actor and Recv Actor.
  // Key: Inter process edge(Parameter name), Value: Send Actor.
  std::map<std::string, SendActorPtr> send_actors_;
  // Key: Inter process edge(Parameter name), Value: Recv Actor.
  std::map<std::string, RecvActorPtr> recv_actors_;

  // The device interface.
  device::DeviceContext *device_context_;
  // The device stream used to async memcpy operators and launch device kernels, such as embedding cache look up and
  // update kernel.
  size_t stream_id_;

  // The embedding cache look up kernel node(operator name: 'Gather').
  CNodePtr embedding_cache_lookup_node_;
  // The embedding cache update kernel node(operator name: 'ScatterUpdate').
  CNodePtr embedding_cache_update_node_;

  // Full Embedding table row num, not less than the total number of feature ids.
  size_t vocab_size_{0};

  // Embedding cache size(row number of embedding cache) of local host cache.
  size_t local_host_cache_size_{0};

  // Record the hash table meta info for all embedding tables.
  std::map<std::string, HashTableInfo> hash_tables_;

  // Record the public information of all device embedding cache tables, such as the mapping relationship of id to
  // index, the information that needs to be updated (swap in and swap out), etc.
  std::shared_ptr<EmbeddingDeviceCache> embedding_device_cache_;
  // Record the public information of all local host embedding cache tables, such as the mapping relationship of id to
  // index, the information that needs to be updated (swap in and swap out), etc.
  std::shared_ptr<EmbeddingHostCache> embedding_host_cache_;

  // Statistics on the cache hit rate of the host and device and the information used to update cache.
  PsCacheStatisticsInfo statistics_info_;

  // Model parallelism is used between multiple workers, and local_embedding_slice_bounds_ records the feature range
  // corresponding to the embedding table slice of the process.
  std::pair<int, int> local_embedding_slice_bounds_;

  // Model parallelism is used between multiple workers, and local_device_cache_bounds_ records the local device cache
  // range corresponding to the embedding table slice of the process.
  std::pair<int, int> local_device_cache_bounds_;

  // In a multi-server scenario, the embeddings need to be segmented, and each server saves the embeddings of
  // different feature id ranges, remote_embedding_slice_bounds_ records the feature range of the embedding table
  // slice on each server.
  std::vector<std::pair<size_t, size_t>> remote_embedding_slice_bounds_;

  // Total server number of cluster.
  size_t server_num_;

  // The flag which indicates whether this actor is running to prefetch cache.
  std::atomic_bool running_{false};
};

using EmbeddingCachePrefetchActorPtr = std::shared_ptr<EmbeddingCachePrefetchActor>;
}  // namespace runtime
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_ACTOR_EMBEDDING_CACHE_EMBEDDING_CACHE_PREFETCH_ACTOR_H_
