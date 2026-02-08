/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_MEMORY_MEM_POOL_ADDRESS_DISCRETIZER_H_
#define MINDSPORE_CCSRC_RUNTIME_MEMORY_MEM_POOL_ADDRESS_DISCRETIZER_H_

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <algorithm>

namespace mindspore {
namespace device {
namespace tracker {
class AddressDiscretizer {
 public:
  explicit AddressDiscretizer(const std::vector<uintptr_t> &addresses) {
    // Sort addresses to maintain relative order
    sorted_addresses_ = addresses;
    std::sort(sorted_addresses_.begin(), sorted_addresses_.end());

    // Create mapping from address to discrete id
    uint32_t id = 0;
    for (size_t i = 0; i < sorted_addresses_.size(); ++i) {
      auto iter = addr_to_id_.find(sorted_addresses_[i]);
      if (iter == addr_to_id_.end()) {
        addr_to_id_[sorted_addresses_[i]] = id;
        id++;
      }
    }
  }

  uint32_t GetDiscreteId(uintptr_t address) const {
    auto it = addr_to_id_.find(address);
    return it != addr_to_id_.end() ? it->second : UINT32_MAX;
  }

  uintptr_t GetOriginalAddress(uint32_t id) const { return id < sorted_addresses_.size() ? sorted_addresses_[id] : 0; }

  size_t GetDiscretizedCount() const { return addr_to_id_.size(); }

 private:
  std::vector<uintptr_t> sorted_addresses_;
  std::unordered_map<uintptr_t, uint32_t> addr_to_id_;
};
}  // namespace tracker
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_MEMORY_MEM_POOL_ADDRESS_DISCRETIZER_H_
