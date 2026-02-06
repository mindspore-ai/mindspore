/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_CORE_GRAPH_EXECUTOR_PRE_LAUNCH_PRE_LAUNCH_COMM_H_
#define MINDSPORE_CCSRC_RUNTIME_CORE_GRAPH_EXECUTOR_PRE_LAUNCH_PRE_LAUNCH_COMM_H_

#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <set>
#include "backend/ms_backend/runtime/actors/base/kernel_actor.h"
#include "backend/ms_backend/runtime/actors/base/kernel_runner.h"
#include "backend/ms_backend/runtime/actors/base/actor_set.h"

namespace mindspore {
namespace runtime {

struct CommKernelInfo {
  std::string name;
  std::string group;
  int64_t src_rank{-1};
  int64_t dest_rank{-1};
  std::string ToString() const {
    return name + ", group:" + group + ", src_rank:" + std::to_string(src_rank) +
           ", dest_rank:" + std::to_string(dest_rank);
  }

  bool operator==(const CommKernelInfo &other) const {
    return (name == other.name) && (group == other.group) && (src_rank == other.src_rank) &&
           (dest_rank == other.dest_rank);
  }
};

enum SortedFunc {
  SORTED_BY_SEND_SEQUENTAIL,
  SORTED_BY_RECV_SEQUENTAIL,
  SORTED_BY_SEND_REVERSE,
  SORTED_BY_RECV_REVERSE
};
using LaunchCommNode = std::tuple<runtime::KernelRunnerPtr, CNodePtr, CommKernelInfo, KernelLaunchInfoWithStream>;

// PreLaunchComm is used to launch communication kernel before launch all kernels.
class BACKEND_EXPORT PreLaunchComm {
 public:
  static PreLaunchComm &GetInstance();

  CommKernelInfo GetKernelInfo(const CNodePtr &);
  void SpiltBucket(const std::vector<LaunchCommNode> &, std::vector<LaunchCommNode> *, std::vector<LaunchCommNode> *,
                   std::vector<LaunchCommNode> *, std::vector<LaunchCommNode> *);
  void Launch(std::vector<LaunchCommNode> &, SortedFunc);
  void PreLaunchCommKernel(runtime::ActorSet *);
  void CachePreLaunchOrder(uint32_t graph_id);
  std::vector<uint32_t> GetPreLaunchOrder(bool force_launch);

 private:
  PreLaunchComm() = default;
  ~PreLaunchComm() = default;

  std::set<std::string> is_pre_launch_comm_;
  std::vector<uint32_t> orders_;
};

}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_CORE_GRAPH_EXECUTOR_PRE_LAUNCH_PRE_LAUNCH_COMM_H_
