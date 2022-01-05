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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CUSTOM_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CUSTOM_ACTOR_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include "utils/hash_map.h"
#include "runtime/framework/actor/actor_common.h"
#include "runtime/framework/actor/debug_aware_actor.h"
#include "runtime/hardware/device_context.h"
#include "runtime/framework/device_tensor_store.h"
#include "backend/kernel_compiler/kernel.h"
#include "ir/anf.h"
#include "ir/tensor.h"
#include "utils/anf_utils.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceContext;
using mindspore::device::KernelInfo;
using mindspore::kernel::Address;
using mindspore::kernel::KernelLaunchInfo;
using mindspore::tensor::TensorPtr;

class CustomActor : public AbstractActor {
 public:
  CustomActor(const std::string &name, const AnfNodePtr &kernel, const DeviceContext *device_context,
              const AID *recorder_aid)
      : AbstractActor(name, KernelTransformType::kCustomActor, recorder_aid), kernel_(kernel) {
    device_contexts_.push_back(device_context);
  }
  CustomActor(const std::string &name, const AnfNodePtr &kernel, const DeviceContext *device_context,
              const AID *recorder_aid, GraphExecutionStrategy strategy)
      : AbstractActor(name, KernelTransformType::kCustomActor, recorder_aid), kernel_(kernel), strategy_(strategy) {
    device_contexts_.push_back(device_context);
  }
  ~CustomActor() override = default;

  void Init() override;

  const AnfNodeWeakPtr &kernel() const { return kernel_; }

 protected:
  void Run(OpContext<DeviceTensor> *const context) override;

 private:
  friend class GraphScheduler;
  friend class ControlNodeScheduler;

  // The info of kernel.
  AnfNodeWeakPtr kernel_;
  AnfUtils::CustomActorCallback custom_func_ = {};
  GraphExecutionStrategy strategy_{GraphExecutionStrategy::kPipeline};
};

using CustomActorPtr = std::shared_ptr<CustomActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_CUSTOM_ACTOR_H_
