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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_CXX_API_MODEL_POOL_RESOURCE_MANAGER_H_
#define MINDSPORE_LITE_SRC_RUNTIME_CXX_API_MODEL_POOL_RESOURCE_MANAGER_H_
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include <mutex>
#include "include/api/status.h"

namespace mindspore {
class ResourceManager {
 public:
  static ResourceManager *GetInstance();
  ~ResourceManager() = default;

  std::vector<int> ParseCpuCoreList(size_t *percentage);
  Status DistinguishPhysicalAndLogical(std::vector<int> *physical_cores, std::vector<int> *logical_cores);
  Status DistinguishPhysicalAndLogicalByNuma(std::vector<std::vector<int>> *numa_physical_cores,
                                             std::vector<std::vector<int>> *numa_logical_cores);

 private:
  ResourceManager() = default;
  std::mutex manager_mutex_;
  size_t can_use_core_num_ = 0;
  int core_num_ = 0;
  bool can_use_all_resource_ = true;
  std::vector<int> cpu_cores_;
  std::vector<int> physical_core_ids_;
  std::vector<int> logical_core_ids_;
  std::vector<std::vector<int>> numa_physical_core_ids_;
  std::vector<std::vector<int>> numa_logical_core_ids_;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_RUNTIME_CXX_API_MODEL_POOL_RESOURCE_MANAGER_H_
