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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_DISTRIBUTION_COLLECTIVE_INIT_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_DISTRIBUTION_COLLECTIVE_INIT_H_

#ifndef _WIN32
#include <dlfcn.h>
#endif
#include <vector>
#include <string>
#include "include/backend/visible.h"

namespace mindspore {
namespace device {
namespace gpu {
using InitMPI = void (*)();
using InitNCCLComm = void (*)();
using GetLocalRankId = int (*)();
using CreateCommGroupFunc = bool (*)(const std::string &, const std::vector<unsigned int> &);
using GetRankIDByGroupFunc = int (*)(const std::string &);
using GetGroupSizeFunc = int (*)(const std::string &);
using DestroyGroupFunc = bool (*)(const std::string &);

class GPU_EXPORT CollectiveInitializer {
 public:
  CollectiveInitializer(CollectiveInitializer const &) = delete;
  CollectiveInitializer &operator=(const CollectiveInitializer &) = delete;
  static CollectiveInitializer &instance();
  bool collective_inited() const;
  const void *collective_handle();
  static void InitCollective();
  static void FinalizeCollective();
  static uint32_t GetRankID(const std::string &group_name);
  static uint32_t GetRankSize(const std::string &group_name);

  // The capsulation of the collective communication APIs for compatibility.
  uint32_t local_rank_id();
  bool CreateCommunicationGroup(const std::string &group_name, const std::vector<uint32_t> &group_ranks);
  bool DestroyCommunicationGroup(const std::string &group_name);
  uint32_t GetRankIDByGroup(const std::string &group_name);
  uint32_t GetGroupSize(const std::string &group_name);

 private:
  CollectiveInitializer() : use_mpi_(false), collective_inited_(false), collective_handle_(nullptr) {}
  ~CollectiveInitializer() = default;

  bool use_mpi_;
  bool collective_inited_;
  void *collective_handle_{nullptr};
};
}  // namespace gpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_DISTRIBUTION_COLLECTIVE_INIT_H_
