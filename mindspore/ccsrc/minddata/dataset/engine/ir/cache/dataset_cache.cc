/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/ir/cache/dataset_cache.h"

#include <memory>
#include <string>
#include <optional>
#include <vector>

#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/ir/cache/dataset_cache_impl.h"
#endif

namespace mindspore::dataset {
#ifndef ENABLE_ANDROID
Status DatasetCache::from_json(nlohmann::json json_obj, std::shared_ptr<DatasetCache> *cache) {
  if (json_obj.find("cache") != json_obj.end()) {
    nlohmann::json json_cache = json_obj["cache"];
    CHECK_FAIL_RETURN_UNEXPECTED(json_cache.find("session_id") != json_cache.end(), "Failed to find session_id");
    CHECK_FAIL_RETURN_UNEXPECTED(json_cache.find("cache_memory_size") != json_cache.end(),
                                 "Failed to find cache_memory_size");
    CHECK_FAIL_RETURN_UNEXPECTED(json_cache.find("spill") != json_cache.end(), "Failed to find spill");
    session_id_type id = static_cast<session_id_type>(json_cache["session_id"]);
    uint64_t mem_sz = json_cache["cache_memory_size"];
    bool spill = json_cache["spill"];
    std::optional<std::vector<char>> hostname_c = std::nullopt;
    std::optional<int32_t> port = std::nullopt;
    std::optional<int32_t> num_connections = std::nullopt;
    std::optional<int32_t> prefetch_sz = std::nullopt;
    if (json_cache.find("hostname") != json_cache.end()) {
      std::optional<std::string> hostname = json_cache["hostname"];
      hostname_c = std::vector<char>(hostname->begin(), hostname->end());
    }
    if (json_cache.find("port") != json_cache.end()) port = json_cache["port"];
    if (json_cache.find("num_connections") != json_cache.end()) num_connections = json_cache["num_connections"];
    if (json_cache.find("prefetch_size") != json_cache.end()) prefetch_sz = json_cache["prefetch_size"];
    *cache = std::make_shared<DatasetCacheImpl>(id, mem_sz, spill, hostname_c, port, num_connections, prefetch_sz);
  }
  return Status::OK();
}
#endif
}  // namespace mindspore::dataset
