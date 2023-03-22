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

#include <string>
#include <memory>
#include <map>
#include <utility>
#include <vector>
#include <queue>
#include "runtime/graph_scheduler/actor/debug_aware_actor.h"
#include "runtime/graph_scheduler/actor/actor_common.h"
#include "runtime/hardware/device_context.h"
#include "ir/anf.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceAddress;
using mindspore::device::DeviceContext;

// The Super kernel actor is used to represent the sink executing of graph which is the combination of kernels.
class SuperKernelActor : public DebugAwareActor {
 public:
  SuperKernelActor(const std::string &name, const KernelGraphPtr &graph, const DeviceContext *device_context,
                   const AID &memory_manager_aid, const AID *debug_aid, const AID *recorder_aid)
      : DebugAwareActor(name, KernelTransformType::kSuperKernelActor, recorder_aid, memory_manager_aid, debug_aid),
        graph_(graph) {
    (void)device_contexts_.emplace_back(device_context);
    input_device_tensors_.resize(graph->input_nodes().size());
  }
  ~SuperKernelActor() override = default;

  size_t FetchInputNodePosition(const AnfNodePtr &intput_node);
  void FetchInputDeviceTensor(OpContext<DeviceTensor> *const context);
  // The debug related operation interface.
  void SendDebugReq(OpContext<DeviceTensor> *const context) override;

  // The memory related operation interface.
  void SendMemoryAllocReq(OpContext<DeviceTensor> *const context) override;
  // The callback after memory alloc finished.
  void OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) override;
  // The input may come from the control actor, so need free the input memory by the dynamic ref count.
  void SendMemoryFreeReq(OpContext<DeviceTensor> *const context) override;

  const KernelGraphPtr &graph() const { return graph_; }

 protected:
  void Init() override;
  void Run(OpContext<DeviceTensor> *const context) override;

 private:
  friend class GraphScheduler;

  bool CopyInputData(const OpContext<DeviceTensor> *context);

  KernelGraphPtr graph_;

  // The device tensors of graph input parameter, which used to compare the recv input data.
  std::vector<DeviceTensorPtr> node_device_tensors_;

  // In the scheduler, check whether the parameters need to be copied after lunch. Only when the parameter has
  // the ref attribute and is directly used by the kernel in the graph, it needs to be copied.
  std::vector<bool> is_parameters_need_copy_;

  // Record the address map of ref node to copy back when running finished.
  std::map<DeviceAddress *, DeviceAddress *> ref_node_addr_map_;

  // The device tensors for memory alloc.
  std::vector<DeviceTensor *> memory_alloc_list_;
  // The lists of device tensors which need free by dynamic ref count, will be cleared at the end of step.
  std::queue<std::vector<DeviceTensor *>> memory_free_lists_;
  // The received input device type and format may be different from the formal parameter in the control flow scenarios,
  // so it needs to be copied from the input data to real data that graph launch needs.
  std::vector<DeviceTensorPtr> copy_input_device_tensors_;
  // The input device tensors for launch.
  std::vector<DeviceTensor *> input_device_tensors_;
};

using SuperKernelActorPtr = std::shared_ptr<SuperKernelActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_KERNEL_ACTOR_H_
