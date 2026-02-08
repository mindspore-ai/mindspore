/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_SUPER_KERNEL_ACTOR_H_
#define MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_SUPER_KERNEL_ACTOR_H_

#include <string>
#include <memory>
#include <map>
#include <utility>
#include <vector>
#include <queue>
#include <set>
#include "backend/ge_backend/runtime/actor/debug_aware_actor.h"
#include "backend/ge_backend/runtime/actor/actor_common.h"
#include "ir/anf.h"
#include "backend/ge_backend/executor/ge_graph_executor.h"

namespace mindspore {
namespace ge_backend {
namespace runtime {
using mindspore::device::DeviceAddress;

struct OutputMemoryInfo {
  size_t size;
  std::string node_full_name;
};

// The Super kernel actor is used to represent the sink executing of graph which is the combination of kernels.
class SuperKernelActor : public DebugAwareActor {
 public:
  SuperKernelActor(const std::string &name, const KernelGraphPtr &graph, const std::string &graph_phase,
                   const AID &memory_manager_aid, const AID *debug_aid, const AID *recorder_aid,
                   const std::shared_ptr<backend::ge_backend::GeGraphExecutor> &graph_executor,
                   KernelTransformType type = KernelTransformType::kSuperKernelActor)
      : DebugAwareActor(name, type, recorder_aid, memory_manager_aid, debug_aid, nullptr),
        graph_(graph),
        graph_phase_(graph_phase),
        is_infer_phase_(IsInferPhase(graph_phase)),
        graph_executor_(graph_executor) {
    input_kernel_tensors_.resize(graph->input_nodes().size());
  }
  ~SuperKernelActor() override = default;

  size_t FetchInputNodePosition(const AnfNodePtr &intput_node);
  virtual void FetchInputDeviceTensor(OpContext<KernelTensor> *const context);
  // The debug related operation interface.
  void SendDebugReq(OpContext<KernelTensor> *const context) override;

  // The memory related operation interface.
  void SendMemoryAllocReq(OpContext<KernelTensor> *const context) override;
  // The callback after memory alloc finished.
  void OnMemoryAllocFinish(OpContext<KernelTensor> *const context) override;
  // The input may come from the control actor, so need free the input memory by the dynamic ref count.
  void SendMemoryFreeReq(OpContext<KernelTensor> *const context) override;
  bool CopyInputData(const OpContext<KernelTensor> *context, const KernelGraphPtr &graph);

  const KernelGraphPtr &graph() const { return graph_; }

 protected:
  void Init() override;
  void Run(OpContext<KernelTensor> *const context) override;
  // The input device tensors for launch.
  std::vector<KernelTensorPtr> input_kernel_tensors_;
  // The device tensors of graph input parameter, which used to compare the recv input data.
  std::vector<KernelTensorPtr> node_kernel_tensors_;
  // The device tensors for memory alloc.
  std::vector<KernelTensorPtr> memory_alloc_list_;
  // The lists of device tensors which need free by dynamic ref count, will be cleared at the end of step.
  std::queue<std::vector<KernelTensorPtr>> memory_free_lists_;

 private:
  bool CopyInputDataPersistedHandle(const KernelTensorPtr &input_kernel_tensor,
                                    const KernelTensorPtr &node_kernel_tensor, size_t i);

  void FetchPersistentDeviceTensor();

  void TrackInputMemory();

  friend class GraphScheduler;
  KernelGraphPtr graph_;

  // The phase of the root graph this super actor belongs to.
  std::string graph_phase_;
  // Whether the super kernel actor is a infer 'prefill' or 'increment' graph or not.
  bool is_infer_phase_;

  // In the scheduler, check whether the parameters need to be copied after lunch. Only when the parameter has
  // the ref attribute and is directly used by the kernel in the graph, it needs to be copied.
  std::vector<bool> is_parameters_need_copy_;

  // Record the address map of ref node to copy back when running finished.
  std::map<KernelTensor *, KernelTensor *> ref_node_addr_map_;

  // The received input device type and format may be different from the formal parameter in the control flow scenarios,
  // so it needs to be copied from the input data to real data that graph launch needs.
  std::vector<KernelTensorPtr> copy_input_kernel_tensors_;
  // Record the device address to the output node of graph.
  std::map<DeviceAddress *, OutputMemoryInfo> device_address_to_node_;
  std::shared_ptr<backend::ge_backend::GeGraphExecutor> graph_executor_;
};

using SuperKernelActorPtr = std::shared_ptr<SuperKernelActor>;
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_KERNEL_ACTOR_H_
