/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_FRONTEND_PARALLEL_DEVICE_MANAGER_H_
#define MINDSPORE_CCSRC_INCLUDE_FRONTEND_PARALLEL_DEVICE_MANAGER_H_

#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "frontend/parallel/device.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/group_manager.h"
#include "frontend/parallel/status.h"
#include "frontend/parallel/strategy.h"
#include "utils/convert_utils.h"
#include "include/utils/visible.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace parallel {
class FRONTEND_EXPORT ParallelCommManager {
 public:
  ParallelCommManager() = default;
  ~ParallelCommManager() = default;
  ParallelCommManager(const ParallelCommManager &) = delete;
  ParallelCommManager &operator=(const ParallelCommManager &) = delete;
  static std::shared_ptr<ParallelCommManager> GetInstance();

  std::string RankListName(const std::vector<uint32_t> &ranks) const;
  std::string HashName(const std::string &origin_name) const;

  void SetHcclGroups(const std::vector<uint32_t> &ranks, std::string name, bool flag);
  std::optional<std::pair<std::string, bool>> HcclGroups(const std::vector<uint32_t> &ranks) const;

 private:
  mindspore::HashMap<std::string, std::pair<std::string, bool>> hccl_groups_map_;  // {rank_list: <group_name, flag>}
  inline static std::shared_ptr<ParallelCommManager> group_instance_{nullptr};
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_FRONTEND_PARALLEL_DEVICE_MANAGER_H_
