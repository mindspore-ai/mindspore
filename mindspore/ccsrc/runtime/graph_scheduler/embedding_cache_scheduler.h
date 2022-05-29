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

#include "runtime/graph_scheduler/actor/embedding_cache/embedding_cache_prefetch_actor.h"

namespace mindspore {
namespace runtime {
// EmbeddingCacheScheduler could be used to build, schedule and finalize embedding cache prefetch actor
// to cache large embedding table of a large recommendation network model. The cache level is:
// Device Cache->Local Host Cache->Remote Cache. The embedding cache prefetch actor is used to perform Local
// and Device Cache hit analysis and cache prefetching.
class EmbeddingCacheScheduler {
 public:
  EmbeddingCacheScheduler() = default;
  ~EmbeddingCacheScheduler() = default;
  DISABLE_COPY_AND_ASSIGN(EmbeddingCacheScheduler);

  // Build and initialize embedding cache prefetch actor and save it by embedding_cache_prefetch_actor_.
  void Initialize();

  // 1. Build network connection between local and remote cache for embedding cache prefetch actor.
  // 2. Schedule and Run embedding cache prefetch actor.
  // Since the embedding cache prefetch actor is spinning, and the actor is not in the actor set, start the actor in the
  // Schedule interface.
  void Schedule() const;

  // Synchronize latest embedding table in local cache to remote.
  void SyncEmbeddingTable() const;

  // Finalize embedding cache prefetch actor.
  void Finalize();

 private:
  // Embedding cache prefetch actor.
  EmbeddingCachePrefetchActorPtr embedding_cache_prefetch_actor_;

  // The flag indicates whether the EmbeddingCacheScheduler is initialized.
  bool initialized_{false};
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_EMBEDDING_CACHE_SCHEDULER_H_
