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

#include "distributed/embedding_cache/embedding_storage/embedding_storage.h"
#include <map>
#include <string>
#include "distributed/embedding_cache/cache_strategy/lru_cache.h"
#include "distributed/persistent/storage/local_file.h"
#if defined(__linux__) && defined(WITH_BACKEND)
#include "include/backend/distributed/ps/ps_context.h"
#include "include/backend/distributed/cluster/cluster_context.h"
#endif
#include "utils/file_utils.h"

namespace mindspore {
namespace distributed {
namespace storage {
namespace {
// The environment variable used to set the embedding remote persistent file storage path.
constexpr auto kEnvEmbeddingRemoteStoragePath = "MS_EMBEDDING_REMOTE_STORAGE_PATH";
// Default value for embedding remote persistent file storage path.
constexpr auto kDefaultEmbeddingRemoteStoragePath = "./embedding_storage";

// Get embedding remote persistent file storage path from environment variable.
std::string GetEmbeddingRemoteStoragePath() {
  std::string stoage_path = common::GetEnv(kEnvEmbeddingRemoteStoragePath);
  if (stoage_path.empty()) {
    return kDefaultEmbeddingRemoteStoragePath;
  }

  return stoage_path;
}
}  // namespace

template <typename KeyType, typename ValueType, typename Allocator>
void EmbeddingStorage<KeyType, ValueType, Allocator>::Initialize(const DeviceAddress *device_address) {
  // 1. Get rank id.
#if defined(__linux__) && defined(WITH_BACKEND)
  if (!ps::PSContext::instance()->cache_enable() || !distributed::cluster::ClusterContext::instance()->initialized() ||
      !ps::PSContext::instance()->is_server()) {
    return;
  }

  auto node = distributed::cluster::ClusterContext::instance()->node();
  MS_EXCEPTION_IF_NULL(node);
  uint32_t rank_id = node->rank_id();
#else
  uint32_t rank_id = 0;
#endif

  // 2. Create the host memory cache instance.
  cache_ = std::make_unique<LRUCache<KeyType, int>>(cache_capacity_);
  MS_EXCEPTION_IF_NULL(cache_);

  // 3. Create the persistent storage instance.
  std::string storage_file_root_path = GetEmbeddingRemoteStoragePath();
  const std::string kEmbeddingStorageFilePrefix = "embedding_table_";
  std::string storage_file_path = storage_file_root_path + "/rank_" + std::to_string(rank_id) + "/" +
                                  kEmbeddingStorageFilePrefix + std::to_string(embedding_key_);
  if (!FileIOUtils::IsFileOrDirExist(storage_file_path)) {
    FileIOUtils::CreateDirRecursive(storage_file_path);
  }
  auto ret = FileUtils::GetRealPath(storage_file_path.c_str());
  if (!ret.has_value()) {
    MS_LOG(EXCEPTION) << "Cannot get real path of persistent storage file for parameter.";
  }
  std::string storage_file_real_path = ret.value();

  std::map<std::string, std::string> config_map;
  (void)config_map.emplace(kFileStoragePath, storage_file_real_path);
  (void)config_map.emplace(kElementSize, std::to_string(embedding_dim_));

  storage_ = std::make_unique<LocalFile<KeyType, ValueType>>(config_map);
  MS_EXCEPTION_IF_NULL(storage_);
  storage_->Initialize();
}

template <typename KeyType, typename ValueType, typename Allocator>
void EmbeddingStorage<KeyType, ValueType, Allocator>::Finalize() {
  MS_EXCEPTION_IF_NULL(cache_);
  cache_ = nullptr;
  MS_EXCEPTION_IF_NULL(storage_);
  storage_->Finalize();
  storage_ = nullptr;
}

template class EmbeddingStorage<int32_t, bool>;
template class EmbeddingStorage<int32_t, int8_t>;
template class EmbeddingStorage<int32_t, int16_t>;
template class EmbeddingStorage<int32_t, int32_t>;
template class EmbeddingStorage<int32_t, int64_t>;
template class EmbeddingStorage<int32_t, uint8_t>;
template class EmbeddingStorage<int32_t, uint16_t>;
template class EmbeddingStorage<int32_t, uint32_t>;
template class EmbeddingStorage<int32_t, uint64_t>;
template class EmbeddingStorage<int32_t, float16>;
template class EmbeddingStorage<int32_t, float>;
template class EmbeddingStorage<int32_t, double>;

template class EmbeddingStorage<int64_t, bool>;
template class EmbeddingStorage<int64_t, int8_t>;
template class EmbeddingStorage<int64_t, int16_t>;
template class EmbeddingStorage<int64_t, int32_t>;
template class EmbeddingStorage<int64_t, int64_t>;
template class EmbeddingStorage<int64_t, uint8_t>;
template class EmbeddingStorage<int64_t, uint16_t>;
template class EmbeddingStorage<int64_t, uint32_t>;
template class EmbeddingStorage<int64_t, uint64_t>;
template class EmbeddingStorage<int64_t, float16>;
template class EmbeddingStorage<int64_t, float>;
template class EmbeddingStorage<int64_t, double>;

template class EmbeddingStorage<int32_t, float, std::allocator<uint8_t>>;
template class EmbeddingStorage<int64_t, float, std::allocator<uint8_t>>;
}  // namespace storage
}  // namespace distributed
}  // namespace mindspore
