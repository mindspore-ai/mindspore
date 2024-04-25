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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_PROFILER_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_PROFILER_ACTOR_H_

#include <vector>
#include <set>
#include <mutex>
#include <string>
#include "runtime/graph_scheduler/actor/actor_common.h"
#include "runtime/graph_scheduler/device_tensor_store.h"
#include "runtime/hardware/device_context.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceContext;
using mindspore::kernel::KernelLaunchAddr;

// The debug actor is used to debug and dump kernel info, it gets the kernel real time execution info in the device, so
// it is synchronous and blocked.
class ProfilerActor : public ActorBase {
 public:
  ProfilerActor() : ActorBase("ProfilerActor") {}
  ~ProfilerActor() override = default;

  void AscendStepStart(const std::vector<KernelGraphPtr> &graphs, std::vector<DeviceContext *> device_contexts);

  void AscendStepEnd();

  // The debug on step begin.
  void ProfilerOnStepBegin(const std::vector<KernelGraphPtr> &graphs,
                           const std::vector<AnfNodePtr> &origin_parameters_order,
                           std::vector<DeviceContext *> device_contexts, OpContext<DeviceTensor> *const op_context,
                           const AID *from_aid);

  // The debug on step end.
  void ProfilerOnStepEnd(OpContext<DeviceTensor> *const op_context, const AID *from_aid, int total_running_count_);
  static inline uint64_t current_step{1};

 private:
  // class members
  uint32_t exec_order_ = 0;
  int step_count = 0;
  bool dump_flag = false;
  int is_dataset_sink = 0;

  bool profile_started_ = false;
  DeviceContext *device_ctx_ = nullptr;

  // Support multi-thread.
  std::mutex debug_mutex_;
};

}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_PROFILER_ACTOR_H_
