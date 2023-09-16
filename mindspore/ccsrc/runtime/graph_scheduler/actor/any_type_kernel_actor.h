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
#include "runtime/graph_scheduler/actor/super_kernel_actor.h"
#include "runtime/graph_scheduler/actor/actor_common.h"
#include "include/common/utils/python_adapter.h"
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

  void RunOpData(OpData<DeviceTensor> *const input_data, OpContext<DeviceTensor> *const context) override;
  void RunOpControl(AID *const input_control, OpContext<DeviceTensor> *const context) override;
  const std::string &current_data_type() const { return current_data_type_; }

 protected:
  void Init() override;

  // Hand the graph input.
  // The execution of actor is divided into the following steps:
  // Receive graph inputs:
  // 1. generate type key
  // 2. check whether the corresponding graph already exists, if not found, execute 3, if there is, execute 4
  // 3. compile the corresponding kernel_graph according to the type and generate the corresponding actor_set
  // 4. send graph inputs to kernel actor of current graph
  void RunForGraphInput(OpContext<DeviceTensor> *const context);
  void FetchInputDeviceTensor(OpContext<DeviceTensor> *const context) override;
  void UpdataDynamicShapeParameterForGraphInput(OpContext<DeviceTensor> *const context);
  void SendOutput(OpContext<DeviceTensor> *const context) override;
  void OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) override;

  // Handle the graph output.
  bool CheckGraphOutputRunningCondition(const OpContext<DeviceTensor> *context);
  // Receive graph outputs:
  // 1. find the corresponding arrow according to the current type key, and send the outputs.
  void RunForGraphOutput(OpContext<DeviceTensor> *const context);
  void FetchGraphOutput(OpContext<DeviceTensor> *const context);
  void EraseGraphOutput(OpContext<DeviceTensor> *const context);
  void UpdataDynamicShapeParameterForGraphOutput(OpContext<DeviceTensor> *const context);
  void UpdateOutputData(OpData<DeviceTensor> *const output_data, const DataArrowPtr &data_arrow,
                        const AnfNodePtr &output_node, OpContext<DeviceTensor> *const context) override;

 private:
  friend class AnyTypeGraphScheduler;

  // When the actor receives the input of the graph, it can determine the data type of the parameter and then compile
  // an executable kernel graph and actors.
  mindspore::HashMap<string, std::vector<AbstractActorPtr>> actors_;
  // Kernel graphs that are actually executed.
  mindspore::HashMap<string, KernelGraphPtr> real_graphs_;
  // The positions of any type parameter in the kernel graph.
  // After graph compiler, a unique key will be generate according to the type of these parameters to save the arrows
  // corresponding to the graph.
  std::vector<size_t> any_type_parameter_indexes_;
  // The data type of any type parameters in the currently received input, the format is like:typeid1_typeid2_typeid3.
  std::string current_data_type_;

  // Parameters that have a dynamic shape.
  mindspore::HashMap<std::string, std::vector<AnfNodePtr>> graph_input_backend_parameters_;

  // Arrows send to kernel/superkernel actors of graph.
  mindspore::HashMap<std::string, std::vector<DataArrowPtr>> graph_input_data_arrows_;
  mindspore::HashMap<std::string, std::vector<ControlArrowPtr>> graph_input_control_arrows_;
  // The output_data_nodes_ and output_data_ corresponds to the output_data_arrows_ one by one.
  mindspore::HashMap<std::string, std::vector<AnfNodePtr>> graph_input_data_nodes_;
  // The second of pair indicates the output data flag. See constant prefixed with kOutputDataFalg for details.
  mindspore::HashMap<std::string, std::vector<std::pair<OpDataUniquePtr<DeviceTensor>, size_t>>> graph_input_data_;
  // Record the fusion output index for output data arrow.
  mindspore::HashMap<std::string, mindspore::HashMap<DataArrow *, size_t>> data_arrow_to_graph_input_actor_indexs_;
  // Used to send batch data in the message which RunBatchOpData needs, the key is the actor name of destination actor.
  mindspore::HashMap<std::string, mindspore::HashMap<std::string, std::vector<OpData<DeviceTensor> *>>>
    batch_graph_input_data_;
  mindspore::HashMap<std::string, mindspore::HashMap<std::string, std::vector<DataArrowPtr>>>
    batch_graph_input_data_arrows_;

  // Graph outputs receive from kernel/superkernel actors of graph.
  mindspore::HashMap<int, std::vector<OpData<DeviceTensor> *>> graph_output_op_data_;
  mindspore::HashMap<int, std::vector<AID *>> graph_output_op_control_;
  std::vector<DeviceTensor *> graph_ouput_device_tensors_;
  // In any type kernel actor, the kernel in the model graph will have fallback scenario, the device type of the
  // model graph and the real graph will be different. A new device address needs to be created for the model graph
  // and placed here.
  std::vector<DeviceTensorPtr> fallback_device_tensors_;
  mindspore::HashMap<std::string, size_t> graph_output_data_num_;
  mindspore::HashMap<std::string, size_t> graph_output_control_num_;

  AnyTypeKernelActorState actor_state_{kAnyTypeKernelActorInit};

  static std::mutex instance_lock_;

  CompileFunc compile_func_;
  TransformFunc transform_func_;
  ScheduleFunc schedule_func_;
};

using AnyTypeKernelActorPtr = std::shared_ptr<AnyTypeKernelActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_ANY_TYPE_ACTOR_H_
