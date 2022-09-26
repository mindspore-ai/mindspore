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
  cache_ = std::make_unique<EmbeddingLRUCache<K, V>>(cache_capacity_, value_size_);
  if (!cache_->Initialize()) {
    MS_LOG(ERROR) << "Cannot initialize cache";
    return false;
  }

  std::string storage_file_root_path = GetEmbeddingRemoteStoragePath();
  std::string storage_file_path = storage_file_root_path + "/" + name_;
  if (!distributed::storage::FileIOUtils::IsFileOrDirExist(storage_file_path)) {
    distributed::storage::FileIOUtils::CreateDirRecursive(storage_file_path);
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
  storage_->Initialize();
  return true;
}

template <typename K, typename V>
bool EmbeddingStore<K, V>::Get(const void *input, size_t key_num, const void *keys, void *values) {
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(keys);
  MS_EXCEPTION_IF_NULL(values);

  // 1. Get data from cache, save miss keys.
  size_t cache_miss_num = 0;
  cache_miss_keys_.resize(key_num);
  cache_miss_indices_.resize(key_num);

  if (!cache_->Get(input, key_num, keys, values, &cache_miss_num, cache_miss_keys_.data(),
                   cache_miss_indices_.data())) {
    MS_LOG(ERROR) << "Cannot get data from cache.";
    return false;
  }

  if (cache_miss_num == 0) {
    return true;
  }

  // 2. Get data of miss keys from storage
  MS_LOG(INFO) << "Embedding store read miss data from storage, num: " << cache_miss_num;
  size_t storage_miss_num = 0;
  storage_miss_indices_.resize(cache_miss_num);
  storage_output_buf_.resize(cache_miss_num * value_size_);
  storage_->Read(cache_miss_num, static_cast<const int32_t *>(cache_miss_keys_.data()), storage_output_buf_.data(),
                 &storage_miss_num, storage_miss_indices_.data());
  if (storage_miss_num > 0) {
    MS_LOG(ERROR) << "Miss some key from storage. num: " << storage_miss_num;
    return false;
  }

  // 3. Copy data of miss keys to values
  for (size_t i = 0; i < cache_miss_num; i++) {
    auto ret = memcpy_s(AddressOffset(values, cache_miss_indices_[i] * value_size_), value_size_,
                        AddressOffset(storage_output_buf_.data(), i * value_size_), value_size_);
    if (ret != 0) {
      MS_LOG(ERROR) << "Failed to copy storage data to return values.";
      return false;
    }
  }

  return true;
}

template <typename K, typename V>
bool EmbeddingStore<K, V>::Get(size_t key_num, const void *keys, void *values) {
  MS_EXCEPTION_IF_NULL(keys);
  MS_EXCEPTION_IF_NULL(values);

  size_t storage_miss_num = 0;
  storage_miss_indices_.resize(key_num);

  storage_->Read(key_num, static_cast<const int32_t *>(keys), values, &storage_miss_num, storage_miss_indices_.data());
  if (storage_miss_num > 0) {
    MS_LOG(INFO) << "Miss some key from storage. num: " << storage_miss_num;
    // After impl flush interface, all data will be in storage, it can not miss data from here.
    return true;
  }

  return true;
}

template <typename K, typename V>
bool EmbeddingStore<K, V>::Put(void *input, size_t key_num, const void *keys, const void *values) {
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(keys);
  MS_EXCEPTION_IF_NULL(values);

  size_t evicted_num = 0;
  evicted_keys_.resize(key_num);
  evicted_values_buf_.resize(key_num * value_size_);
  if (!cache_->Put(input, key_num, keys, values, &evicted_num, evicted_keys_.data(), evicted_values_buf_.data())) {
    MS_LOG(ERROR) << "Cannot put data to cache.";
    return false;
  }

  if (evicted_num == 0) {
    return true;
  }

  // put evicted data to storage
  MS_LOG(INFO) << "Embedding store Write evicted data to storage, num: " << evicted_num;
  storage_->Write(evicted_values_buf_.data(), evicted_num, static_cast<const int32_t *>(evicted_keys_.data()));
  return true;
}

std::string GetEmbeddingRemoteStoragePath() {
  std::string stoage_path = common::GetEnv(kEnvEmbeddingRemoteStoragePath);
  if (stoage_path.empty()) {
    return kDefaultEmbeddingRemoteStoragePath;
  }

  return stoage_path;
}

template class EmbeddingStore<int32_t, float>;
template class EmbeddingStore<int32_t, double>;
template class EmbeddingStore<int32_t, int64_t>;
template class EmbeddingStore<int32_t, size_t>;
}  // namespace distributed
}  // namespace mindspore
