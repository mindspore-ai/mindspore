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

#include "utils/comm_manager.h"
#include "utils/convert_utils.h"
#ifndef NO_DLIB
#include "hccl/hcom.h"
#endif

namespace mindspore {
CommManager &CommManager::GetInstance() noexcept {
  static CommManager instance("hccl");
  return instance;
}

#ifndef NO_DLIB
#define HCCL_RUN_CHECK(op_name, group, op)                      \
  do {                                                          \
    auto hccl_result = (op);                                    \
    if (hccl_result != tagHcclResult::HCCL_SUCCESS) {           \
      MS_LOG(ERROR) << op_name << " failed: #" << group << "#"; \
      return false;                                             \
    }                                                           \
  } while (0)

#define HCCL_GROUP_CHECK_EMPTY(group)                              \
  do {                                                             \
    if (group.length() == 0) {                                     \
      MS_LOG(ERROR) << "The length of group name should not be 0"; \
      return false;                                                \
    }                                                              \
  } while (0)

#define HCCL_GROUP_CHECK_IS_WORLD(group)                                \
  do {                                                                  \
    if (group == "hccl_world_group") {                                  \
      MS_LOG(ERROR) << "The group name should not be hccl_world_group"; \
      return false;                                                     \
    }                                                                   \
  } while (0)

bool CommManager::CreateGroupSync(const string &group, const vector<unsigned int> &rank_id_list) const {
  auto rank_size = rank_id_list.size();
  HCCL_GROUP_CHECK_EMPTY(group);
  HCCL_GROUP_CHECK_IS_WORLD(group);
  HCCL_RUN_CHECK(string("create communicate group"), group,
                 hcom_create_group(group.c_str(), UlongToUint(rank_size), vector<unsigned int>(rank_id_list).data()));
  return true;
}

bool CommManager::GetRankID(const string &group, unsigned int *rank_id) const {
  HCCL_GROUP_CHECK_EMPTY(group);
  HCCL_RUN_CHECK(string("get rank_id"), group, hcom_get_rank_id(group.c_str(), rank_id));
  return true;
}

bool CommManager::GetRankSize(const string &group, unsigned int *rank_size) const {
  HCCL_GROUP_CHECK_EMPTY(group);
  HCCL_RUN_CHECK(string("get rank size"), group, hcom_get_rank_size(group.c_str(), rank_size));
  return true;
}

bool CommManager::DestroyGroup(const string &group) const {
  HCCL_GROUP_CHECK_EMPTY(group);
  HCCL_GROUP_CHECK_IS_WORLD(group);
  HCCL_RUN_CHECK(string("destroy communicate group"), group, hcom_destroy_group(group.c_str()));
  return true;
}
#else
bool CommManager::CreateGroupSync(const string &, const vector<unsigned int> &) const { return true; }

bool CommManager::GetRankID(const string &group, unsigned int *rank_id) const { return true; }

bool CommManager::GetRankSize(const string &group, unsigned int *rank_size) const {
  *rank_size = NO_COMM_DLIB_RANK_SIZE;
  return true;
}

bool CommManager::DestroyGroup(const string &group) const { return true; }
#endif
}  // namespace mindspore
