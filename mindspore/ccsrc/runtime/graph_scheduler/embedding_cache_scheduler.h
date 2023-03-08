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

#ifndef MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_EMBEDDING_CACHE_SCHEDULER_H_
#define MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_EMBEDDING_CACHE_SCHEDULER_H_

#include <memory>
#include <vector>
#include <string>
#include "utils/ms_utils.h"
#include "include/backend/kernel_graph.h"
#include "runtime/hardware/device_context.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace runtime {
using device::DeviceContext;
class EmbeddingCachePrefetchActor;

// EmbeddingCacheScheduler could be used to build, schedule and finalize embedding cache prefetch actor
// to cache large embedding table of a large recommendation network model. The cache level is:
// Device Cache->Local Host Cache->Remote Cache. The embedding cache prefetch actor is used to perform Local
// and Device Cache hit analysis and cache prefetching.
class BACKEND_EXPORT EmbeddingCacheScheduler {
 public:
  static EmbeddingCacheScheduler &GetInstance();

  // Build and initialize embedding cache prefetch actor and save it by embedding_cache_prefetch_actor_.
  void Initialize();

  // Set device address for embedding cache parameter.
  void SetEmbedCachedParamAddress(const DeviceContext *device_context, const KernelGraphPtr &graph);
  // Set data set channel name, used for multi dataset mode, such as predict after train.
  void SetDataSetChannel(const std::string &actor_id, const std::vector<KernelGraphPtr> &graphs);

  // Initialize all embedding storage instances.
  void InitEmbeddingStorage(const std::vector<AnfNodePtr> &parameters);

  // 1. Build network connection between local and remote cache for embedding cache prefetch actor.
  // 2. Schedule and Run embedding cache prefetch actor.
  // Since the embedding cache prefetch actor is spinning, and the actor is not in the actor set, start the actor in the
  // Schedule interface.
  void Schedule();

  // Record the number of global steps executed by the compute graph.
  void IncreaseGraphStep(const std::string &actor_id) const;

  // Synchronize latest embedding table in local cache to remote.
  void SyncEmbeddingTable() const;

  // Finalize embedding cache prefetch actor.
  void Finalize(bool sync_embedding_table = true);

 private:
  EmbeddingCacheScheduler() = default;
  ~EmbeddingCacheScheduler() = default;
  DISABLE_COPY_AND_ASSIGN(EmbeddingCacheScheduler);

  // Get ids number in a batch, not batch size.
  void ParseBatchIdsNum(const KernelGraphPtr &graph);

  // Allocate device and local host memory for embedding cache table.
  void AllocMemForEmbeddingCacheTable(const DeviceContext *device_context, const KernelGraphPtr &graph);

  // Embedding cache prefetch actor.
  std::shared_ptr<EmbeddingCachePrefetchActor> embedding_cache_prefetch_actor_;

  // The flag indicates whether already parse batch ids number.
  bool parsed_batch_ids_num_{false};

  // The flag indicates whether already allocate memory for embedding cache tables.
  bool allocated_embed_cache_mem_{false};

  // The flag indicates whether the EmbeddingCacheScheduler is initialized.
  bool initialized_{false};
  // The flag indicates whether the EmbeddingCacheScheduler is scheduled.
  bool scheduled_{false};
  // The flag indicates whether the EmbeddingCacheScheduler is finalized.
  bool finalized_{false};
  // Ensure that the Finalize function is multithreaded safe.
  std::mutex finalize_mutex_;

  // Record data set channel name, used for multi dataset mode, such as predict after train.
  // Key: data prepare actor id for an actor set, Value: data set channel name.
  mindspore::HashMap<std::string, std::string> data_prepare_aid_to_data_channel_;
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_EMBEDDING_CACHE_SCHEDULER_H_
