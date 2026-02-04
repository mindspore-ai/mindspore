/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_ANY_TYPE_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_ANY_TYPE_ACTOR_H_

#include <string>
#include <memory>
#include <map>
#include <utility>
#include <vector>
#include "backend/ms_backend/runtime/actors/base/super_kernel_actor.h"
#include "backend/ms_backend/runtime/actors/base/actor_common.h"
#include "include/utils/python_adapter.h"
#include "ir/anf.h"

namespace mindspore {
namespace runtime {
// State is used to mark the state of the actor, which is divided into two states: processing the input of the graph
// and the output of the graph.
enum AnyTypeKernelActorState { kAnyTypeKernelActorInit, kAnyTypeKernelActorSendInput, kAnyTypeKernelActorSendOutput };
using mindspore::device::DeviceContext;
using DataArrowGroupMap = mindspore::HashMap<std::string, std::vector<DataArrowPtr>>;
using ControlArrowGroupMap = mindspore::HashMap<std::string, std::vector<AID *>>;
using TransformFunc =
  std::function<std::vector<AbstractActorPtr>(const KernelGraphPtr &, const KernelGraphPtr &, const DeviceContext *)>;
using ScheduleFunc = std::function<void(const std::vector<AbstractActorPtr> &)>;
// The Any Type kernel actor is used to represent the graph whose data type is uncertain and need compiler when
// the actor run.
// The execution is as follows:
// 1. Receive input
// 2. Send graph input to kernel\superkernel actor
// 3. Receive graph output from kernel\superkernel actor
// 4. Send graph output
class AnyTypeKernelActor : public SuperKernelActor {
 public:
  AnyTypeKernelActor(const std::string &name, const KernelGraphPtr &graph, const DeviceContext *device_context,
                     const AID &memory_manager_aid, const AID *debug_aid, const AID *recorder_aid,
                     KernelTransformType type = KernelTransformType::kAnyTypeKernelActor);
  ~AnyTypeKernelActor() override = default;
  const std::string &current_data_type() const { return current_data_type_; }

 protected:
  void Init() override;
  void Run(OpContext<KernelTensor> *const context) override;

  void FetchInputDeviceTensor(OpContext<KernelTensor> *const context) override;

  KernelGraphPtr CompileRealKernelGraph(OpContext<KernelTensor> *const context);
  void UpdateOutputData(OpData<KernelTensor> *const output_data, const DataArrowPtr &data_arrow,
                        const AnfNodePtr &output_node, OpContext<KernelTensor> *const context) override;
  // Compile the corresponding kernel_graph according to the input tensors and create the kernel actor in super kernel
  // actor.
  void PrepareRunContext(OpContext<KernelTensor> *const context);
  // Clear the elements in super kernel actor and recreate by the new compiler action.
  void ClearElements(OpContext<KernelTensor> *const context);

 private:
  friend class AnyTypeGraphScheduler;

  // Kernel graphs that are actually executed.
  mindspore::HashMap<string, KernelGraphPtr> real_graphs_;
  // The positions of any type parameter in the kernel graph.
  // After graph compiler, a unique key will be generate according to the type of these parameters to save the arrows
  // corresponding to the graph.
  std::vector<size_t> any_type_parameter_indexes_;
  // The data type of any type parameters in the currently received input, the format is like:typeid1_typeid2_typeid3.
  std::string current_data_type_;

  static std::mutex instance_lock_;

  CompileFunc compile_func_;
  KernelGraphPtr model_graph_;

  std::vector<AnfNodePtr> model_output_data_nodes_;
  std::vector<DataArrowPtr> model_output_data_arrows_;

  // The dependent device tensor stores, the dependent expression is pair<index, AnfNode>.
  // Index is the input position, AnfNode is the key of the device tensor store.
  std::vector<std::pair<size_t, AnfNodePtr>> extern_device_tensor_store_keys_;
  // The dependent parameter stores, the dependent expression is pair<index, ParameterInfo>.
  // Index is the input position, ParameterInfo is used to fetch args and device tensor.
  std::vector<std::pair<size_t, ParameterInfo>> extern_parameter_indexs_;
};

using AnyTypeKernelActorPtr = std::shared_ptr<AnyTypeKernelActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_ANY_TYPE_ACTOR_H_
