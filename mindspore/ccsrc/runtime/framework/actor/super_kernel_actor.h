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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_SUPER_KERNEL_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_SUPER_KERNEL_ACTOR_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <unordered_map>
#include "runtime/framework/actor/actor_common.h"
#include "runtime/framework/actor/debug_aware_actor.h"
#include "runtime/hardware/device_context.h"
#include "runtime/framework/device_tensor_store.h"
#include "backend/kernel_compiler/kernel.h"
#include "ir/anf.h"
#include "ir/tensor.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceContext;
using mindspore::device::KernelInfo;
using mindspore::kernel::Address;
using mindspore::kernel::KernelLaunchInfo;
using mindspore::tensor::TensorPtr;

// The Super kernel actor is used to represent the sink executing of graph which is the combination of kernels.
class SuperKernelActor : public DebugAwareActor {
 public:
  SuperKernelActor(const std::string &name, const KernelGraphPtr &graph, const DeviceContext *device_context,
                   const AID &memory_manager_aid, const AID *debug_aid, const AID *recorder_aid)
      : DebugAwareActor(name, KernelTransformType::kSuperKernelActor, recorder_aid, memory_manager_aid, debug_aid),
        graph_(graph) {
    (void)device_contexts_.emplace_back(device_context);
  }
  ~SuperKernelActor() override = default;

  void Init() override;

  // The super kernel actor run when receive the input data.
  void RunOpData(OpData<DeviceTensor> *const input_data, OpContext<DeviceTensor> *const context) override;

  // The super kernel actor run when receive the input control.
  void RunOpControl(AID *const input_control, OpContext<DeviceTensor> *const context) override;

 private:
  friend class GraphScheduler;

  // Send output data and output controls when finish kernel launch.
  void SendOutput(OpContext<DeviceTensor> *const context) const;

  KernelGraphPtr graph_;
};

using SuperKernelActorPtr = std::shared_ptr<SuperKernelActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_KERNEL_ACTOR_H_
