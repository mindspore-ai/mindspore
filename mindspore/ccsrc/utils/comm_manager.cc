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

#if defined(ENABLE_GPU)
#include "runtime/device/gpu/distribution/collective_init.h"
using CollectiveInitializer = mindspore::device::gpu::CollectiveInitializer;
using CreateCommGroupFunc = mindspore::device::gpu::CreateCommGroupFunc;
using GetRankIDByGroupFunc = mindspore::device::gpu::GetRankIDByGroupFunc;
using GetGroupSizeFunc = mindspore::device::gpu::GetGroupSizeFunc;
using DestroyGroupFunc = mindspore::device::gpu::DestroyGroupFunc;
#endif

namespace mindspore {
#ifndef NO_DLIB
CommManager &CommManager::GetInstance() noexcept {
  static CommManager instance("hccl");
  return instance;
}

#define HCCL_RUN_CHECK(op_name, group, op)                      \
  do {                                                          \
    auto hccl_result = (op);                                    \
    if (hccl_result != 0) {                                     \
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
                 HcomCreateGroup(group.c_str(), UlongToUint(rank_size), vector<unsigned int>(rank_id_list).data()));
  return true;
}

bool CommManager::GetRankID(const string &group, unsigned int *rank_id) const {
  HCCL_GROUP_CHECK_EMPTY(group);
  HCCL_RUN_CHECK(string("get rank_id"), group, HcomGetRankId(group.c_str(), rank_id));
  return true;
}

bool CommManager::GetRankSize(const string &group, unsigned int *rank_size) const {
  HCCL_GROUP_CHECK_EMPTY(group);
  HCCL_RUN_CHECK(string("get rank size"), group, HcomGetRankSize(group.c_str(), rank_size));
  return true;
}

bool CommManager::DestroyGroup(const string &group) const {
  HCCL_GROUP_CHECK_EMPTY(group);
  HCCL_GROUP_CHECK_IS_WORLD(group);
  HCCL_RUN_CHECK(string("destroy communicate group"), group, HcomDestroyGroup(group.c_str()));
  return true;
}
#elif defined(ENABLE_GPU)
CommManager &CommManager::GetInstance() noexcept {
  static CommManager instance("nccl");
  return instance;
}

bool CommManager::CreateGroupSync(const string &group, const vector<unsigned int> &rank_id_list) const {
  const void *collective_handle_ = CollectiveInitializer::instance().collective_handle();
  if (!collective_handle_) {
    MS_LOG(EXCEPTION) << "GPU collective handle is not initialized.";
  }
  MS_LOG(INFO) << "Create communication group " << group << " by rank id list " << rank_id_list;
  auto create_comm_group_funcptr =
    reinterpret_cast<CreateCommGroupFunc>(dlsym(const_cast<void *>(collective_handle_), "CreateCommGroup"));
  MS_EXCEPTION_IF_NULL(create_comm_group_funcptr);
  bool ret = (*create_comm_group_funcptr)(group, rank_id_list);
  if (!ret) {
    MS_LOG(ERROR) << "Creating group " << group << "for rank id list" << rank_id_list << "failed.";
    return ret;
  }
  return ret;
}

bool CommManager::GetRankID(const string &group, unsigned int *rank_id) const {
  const void *collective_handle_ = CollectiveInitializer::instance().collective_handle();
  if (!collective_handle_) {
    MS_LOG(EXCEPTION) << "GPU collective handle is not initialized.";
  }
  auto get_rank_id_funcptr =
    reinterpret_cast<GetRankIDByGroupFunc>(dlsym(const_cast<void *>(collective_handle_), "GetRankIDByGroup"));
  MS_EXCEPTION_IF_NULL(get_rank_id_funcptr);
  int rank = (*get_rank_id_funcptr)(group);
  *rank_id = static_cast<unsigned int>(rank);
  MS_LOG(INFO) << "This process rank id is " << *rank_id << " in group " << group;
  return true;
}

bool CommManager::GetRankSize(const string &group, unsigned int *rank_size) const {
  const void *collective_handle_ = CollectiveInitializer::instance().collective_handle();
  if (!collective_handle_) {
    MS_LOG(EXCEPTION) << "GPU collective handle is not initialized.";
  }
  auto get_group_size_funcptr =
    reinterpret_cast<GetGroupSizeFunc>(dlsym(const_cast<void *>(collective_handle_), "GetGroupSize"));
  MS_EXCEPTION_IF_NULL(get_group_size_funcptr);
  int size = (*get_group_size_funcptr)(group);
  *rank_size = static_cast<unsigned int>(size);
  MS_LOG(INFO) << "Group " << group << " size is " << *rank_size;
  return true;
}

bool CommManager::DestroyGroup(const string &group) const {
  const void *collective_handle_ = CollectiveInitializer::instance().collective_handle();
  if (!collective_handle_) {
    MS_LOG(EXCEPTION) << "GPU collective handle is not initialized.";
  }
  auto destroy_group_funcptr =
    reinterpret_cast<DestroyGroupFunc>(dlsym(const_cast<void *>(collective_handle_), "DestroyGroup"));
  MS_EXCEPTION_IF_NULL(destroy_group_funcptr);

  bool ret = (*destroy_group_funcptr)(group);
  if (!ret) {
    MS_LOG(ERROR) << "Destroying group " << group << " failed.";
    return ret;
  }
  return ret;
}
#else
CommManager &CommManager::GetInstance() noexcept {
  static CommManager instance("hccl");
  return instance;
}

bool CommManager::CreateGroupSync(const string &, const vector<unsigned int> &) const { return true; }

bool CommManager::GetRankID(const string &group, unsigned int *rank_id) const { return true; }

bool CommManager::GetRankSize(const string &group, unsigned int *rank_size) const {
  *rank_size = NO_COMM_DLIB_RANK_SIZE;
  return true;
}

bool CommManager::DestroyGroup(const string &group) const { return true; }
#endif
}  // namespace mindspore
