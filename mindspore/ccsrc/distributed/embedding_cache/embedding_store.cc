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
#include <map>
#include <string>
#include <utility>

#include "distributed/embedding_cache/embedding_store.h"
#include "distributed/persistent/storage/local_file.h"
#include "distributed/persistent/storage/file_io_utils.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace distributed {
template <typename K, typename V>
bool EmbeddingStore<K, V>::Initialize() {
  value_size_ = emb_dim_ * sizeof(V);
  key_size_ = sizeof(K);
  cache_ = std::make_unique<EmbeddingLRUCache<K, V>>(cache_capacity_);

  std::string storage_file_path = GetEmbeddingRemoteStoragePath();
  if (!distributed::storage::FileIOUtils::IsFileOrDirExist(storage_file_path)) {
    distributed::storage::FileIOUtils::CreateDir(storage_file_path);
  }
  auto ret = FileUtils::GetRealPath(storage_file_path.c_str());
  if (!ret.has_value()) {
    MS_LOG(ERROR) << "Cannot get real path of persistent storage file for parameter.";
    return false;
  }
  std::string real_storage_file_path = ret.value();

  std::map<std::string, std::string> config_map;
  config_map[distributed::storage::kFileStoragePath] = real_storage_file_path;
  storage_ = std::make_unique<storage::LocalFile>(config_map, key_size_, value_size_);
  return true;
}

std::string GetEmbeddingRemoteStoragePath() {
  std::string stoage_path = common::GetEnv(kEnvEmbeddingRemoteStoragePath);
  if (stoage_path.empty()) {
    return kDefaultEmbeddingRemoteStoragePath;
  }

  return stoage_path;
}

template class EmbeddingStore<size_t, float>;
template class EmbeddingStore<size_t, double>;
template class EmbeddingStore<size_t, int64_t>;
template class EmbeddingStore<size_t, size_t>;
}  // namespace distributed
}  // namespace mindspore
