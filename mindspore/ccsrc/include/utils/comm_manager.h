/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_COMM_MANAGER_H
#define MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_COMM_MANAGER_H

#include <string>
#include <vector>
#include <utility>
#include <map>
#include <memory>
#include "include/utils/visible.h"

namespace mindspore {
class COMMON_EXPORT CommManager {
 public:
  static CommManager &GetInstance() noexcept;
  static bool Register(const std::string &name, const std::shared_ptr<CommManager> &instance);
  static void Clear();

  CommManager(const CommManager &) = delete;
  virtual ~CommManager() = default;

  virtual bool CreateGroupSync(const std::string &group, const std::vector<unsigned int> &rank_id_list) const = 0;
  virtual bool DestroyGroup(const std::string &group) const = 0;
  virtual bool GetRankID(const std::string &group, unsigned int *rank_id) const = 0;
  virtual bool GetRankSize(const std::string &group, unsigned int *rank_size) const = 0;
  virtual uint32_t GetRank() = 0;

 protected:
  explicit CommManager(std::string backend) : backend_(std::move(backend)) {}

  std::string backend_;
};

COMMON_EXPORT uint32_t GetRank();

COMMON_EXPORT bool IsStandAlone();

#define COMM_MANAGER_REG(NAME, INSTANCE) \
  static bool g_CommManager_##NAME##_reg_result = mindspore::CommManager::Register(NAME, INSTANCE)
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_COMM_MANAGER_H
