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

#include "fl/server/consistent_hash_ring.h"

namespace mindspore {
namespace fl {
namespace server {
bool ConsistentHashRing::Insert(uint32_t rank) {
  for (uint32_t i = 0; i < virtual_node_num_; i++) {
    std::string physical_node_hash_key = std::to_string(rank) + "#" + std::to_string(i);
    size_t hash_value = std::hash<std::string>()(physical_node_hash_key);
    MS_LOG(DEBUG) << "Insert virtual node " << physical_node_hash_key << " for node " << rank << ", hash value is "
                  << hash_value;
    if (ring_.count(hash_value) != 0) {
      MS_LOG(INFO) << "Virtual node " << physical_node_hash_key << " is already mapped to the ring.";
      continue;
    }
    ring_[hash_value] = rank;
  }
  return true;
}

bool ConsistentHashRing::Erase(uint32_t rank) {
  for (auto iterator = ring_.begin(); iterator != ring_.end();) {
    if (iterator->second == rank) {
      (void)ring_.erase(iterator++);
    } else {
      ++iterator;
    }
  }
  return true;
}

uint32_t ConsistentHashRing::Find(const std::string &key) {
  size_t hash_value = std::hash<std::string>()(key);
  auto iterator = ring_.lower_bound(hash_value);
  if (iterator == ring_.end()) {
    // If the virtual node is not found clockwise, the key will be mapped to the first virtual node on the ring.
    iterator = ring_.begin();
  }
  return iterator->second;
}
}  // namespace server
}  // namespace fl
}  // namespace mindspore
