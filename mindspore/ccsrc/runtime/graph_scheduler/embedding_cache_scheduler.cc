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

#include "runtime/graph_scheduler/embedding_cache_scheduler.h"
#include <string>
#include <memory>

namespace mindspore {
namespace runtime {
void EmbeddingCacheScheduler::Initialize() {
  if (!ps::PSContext::instance()->cache_enable() || !distributed::cluster::ClusterContext::instance()->initialized() ||
      !ps::PSContext::instance()->is_worker()) {
    return;
  }

  // Get or Create device context.
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  uint32_t device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_context =
    device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_name, device_id});
  MS_EXCEPTION_IF_NULL(device_context);
  device_context->Initialize();

  // Create and initialize EmbeddingCachePrefetchActor.
  embedding_cache_prefetch_actor_ = std::make_shared<EmbeddingCachePrefetchActor>(device_context);
  MS_EXCEPTION_IF_NULL(embedding_cache_prefetch_actor_);

  initialized_ = true;
}

void EmbeddingCacheScheduler::Schedule() const {
  if (!initialized_) {
    return;
  }

  // 1. Initialize embedding cache prefetch actor and build network connection inter process.
  MS_EXCEPTION_IF_NULL(embedding_cache_prefetch_actor_);
  embedding_cache_prefetch_actor_->Initialize();

  // 2. Spawn embedding cache prefetch actor.
  auto actor_manager = ActorMgr::GetActorMgrRef();
  MS_EXCEPTION_IF_NULL(actor_manager);
  // Bind single thread to execute embedding cache prefetch actor.
  (void)actor_manager->Spawn(embedding_cache_prefetch_actor_, false);

  // 3. Run embedding cache prefetch actor.
  ActorDispatcher::Send(embedding_cache_prefetch_actor_->GetAID(), &EmbeddingCachePrefetchActor::Run);
}

void EmbeddingCacheScheduler::Finalize() {
  if (!initialized_) {
    return;
  }

  MS_EXCEPTION_IF_NULL(embedding_cache_prefetch_actor_);
  // Stop the embedding cache prefetch_actor.
  embedding_cache_prefetch_actor_->Finalize();

  initialized_ = false;
}
}  // namespace runtime
}  // namespace mindspore
