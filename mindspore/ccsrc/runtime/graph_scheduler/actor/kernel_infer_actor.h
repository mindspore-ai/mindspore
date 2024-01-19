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

#ifndef MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_ACTOR_KERNEL_INFER_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_ACTOR_KERNEL_INFER_ACTOR_H_

#include <string>
#include <memory>
#include <vector>
#include "runtime/graph_scheduler/actor/kernel_actor.h"
#include "runtime/hardware/device_context.h"

namespace mindspore {
namespace runtime {
// KernelInferActor is used to Infer the shape output scenario from the dynamic shape asynchronous operator, improving
// the concurrency between dynamic shape operators and improving the performance of the dynamic shape network.
class KernelInferActor : public KernelActor {
 public:
  KernelInferActor(const std::string &name, const CNodePtr &kernel, const DeviceContext *device_context,
                   const AID &memory_manager_aid,
                   const KernelTransformType &type = KernelTransformType::kKernelInferActor)
      : KernelActor(name, kernel, device_context, memory_manager_aid, nullptr, nullptr,
                    GraphExecutionStrategy::kPipeline, {}, {}, type) {}
  ~KernelInferActor() override = default;

  // The actor run when receive the input data.
  void RunOpData(OpData<DeviceTensor> *const input_data, OpContext<DeviceTensor> *const context) override;

  // The memory related operation interface.
  void SendMemoryFreeReq(OpContext<DeviceTensor> *const context) override;

 protected:
  void Run(OpContext<DeviceTensor> *const context) override;
  void Init() override;
  void SendRecorderInfo(OpContext<DeviceTensor> *const context) const override {}
};

using KernelInferActorPtr = std::shared_ptr<KernelInferActor>;
}  // namespace runtime
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_ACTOR_KERNEL_INFER_ACTOR_H_
