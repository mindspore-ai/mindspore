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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_DATA_PREPARE_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_DATA_PREPARE_ACTOR_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <unordered_map>
#include <map>
#include "runtime/framework/graph_compiler.h"
#include "runtime/framework/actor/actor_common.h"
#include "runtime/framework/actor/data_source_actor.h"
#include "runtime/framework/actor/debug_aware_actor.h"
#include "runtime/framework/device_tensor_store.h"
#include "runtime/hardware/device_context.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceContext;

// The data prepare actor is used to prepare data for device tensor store and host tensor queue to represent the begin
// of one step.
class DataPrepareActor : public DebugAwareActor {
 public:
  DataPrepareActor(const std::string &name, const AID &memory_manager_aid, const AID *debug_aid,
                   const GraphCompilerInfo *graph_compiler_info, const HostQueueDSActorPtr &host_data_source_actor,
                   const HostTensorQueuePtr &host_tensor_queue)
      : DebugAwareActor(name, KernelTransformType::kDataPrepareActor, nullptr, memory_manager_aid, debug_aid),
        graph_compiler_info_(graph_compiler_info),
        strategy_(GraphExecutionStrategy::kPipeline),
        host_data_source_actor_(host_data_source_actor),
        host_tensor_queue_(host_tensor_queue),
        loop_count_aid_(nullptr) {}
  ~DataPrepareActor() override = default;

  void Init() override;

  // The process entry of data prepare.
  void PrepareData(const std::vector<std::vector<TensorPtr>> &input_tensors, OpContext<DeviceTensor> *const context);

  // The debug related operation interface.
  void SendDebugReq(OpContext<DeviceTensor> *const context) override;
  void OnDebugFinish(OpContext<DeviceTensor> *const context) override;

  // The continuous memory related operation interface.
  void SendMemoryAllocReq(OpContext<DeviceTensor> *const context) override;
  void OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) override;

 private:
  friend class GraphScheduler;

  // Send output controls when finish data prepare.
  void SendOutput(OpContext<DeviceTensor> *const context);

  void PrepareDataForDeviceTensorStore(const std::vector<std::vector<TensorPtr>> &input_tensors,
                                       OpContext<DeviceTensor> *const context);
  void PrepareDataForHostTensorQueue(const std::vector<std::vector<TensorPtr>> &input_tensors,
                                     OpContext<DeviceTensor> *const context);
  void PrepareDataForStepMode(const std::vector<std::vector<TensorPtr>> &input_tensors,
                              OpContext<DeviceTensor> *const context);

  // Prepare the device data for persistent device tensor of weight node from host tensor.
  void PrepareDataForWeightNode(const AnfNodePtr &backend_node, const AnfNodePtr &front_node, const TensorPtr &tensor,
                                const DeviceContext *device_context, OpContext<DeviceTensor> *const context);
  // Prepare the device data for persistent device tensor of value node.
  void PrepareDataForValueNode(const ValueNodePtr &node, const DeviceContext *device_context,
                               OpContext<DeviceTensor> *const context);
  //  The branch processing of PrepareDataForValueNode that value type is tensor.
  void PrepareDataForValueNodeTensor(const ValueNodePtr &node, const ValuePtr &node_value,
                                     const DeviceContext *device_context, OpContext<DeviceTensor> *const context);

  // The data prepare in the control flow scene.
  void PrepareDeviceTensorStoreForControlNode(const ControlNodeParserPtr &control_node_parser,
                                              const std::vector<TensorPtr> &tensors,
                                              OpContext<DeviceTensor> *const context);
  void PrepareHostTensorQueueForControlNode(const std::vector<TensorPtr> &tensors,
                                            std::vector<TensorPtr> *const host_tensors,
                                            OpContext<DeviceTensor> *const context);
  // In control flow, all weight nodes associated with the host weight parameter need to use the same device tensor.
  void PrepareDataForControlWeightNode(
    const AnfNodePtr &node, const AnfNodePtr &front_node, const TensorPtr &tensor, const DeviceContext *device_context,
    const std::unordered_map<AnfNodePtr, std::vector<AnfNodePtr>> &host_parameter_to_weights,
    OpContext<DeviceTensor> *const context);

  const GraphCompilerInfo *graph_compiler_info_;
  GraphExecutionStrategy strategy_;
  HostQueueDSActorPtr host_data_source_actor_;
  HostTensorQueuePtr host_tensor_queue_;

  // The output controls contain the data source actors and the no input kernel actors.
  std::vector<AID> data_source_aids_;
  std::vector<AID> no_input_kernel_aids_;
  // If has no data source actor and kernel actor, then need send to loop count actor.
  const AID *loop_count_aid_;

  // The nodes need continuous memory, which must allocate in the begin of step running. The first bool of pair
  // expresses the inputs of node need continuous memory, the second bool of pair expresses the outputs of node need
  // continuous memory.
  std::map<std::pair<CNodePtr, DeviceContext *>, std::pair<bool, bool>> continuous_memory_nodes_;
  // The members for continuous memory alloc fetched by continuous_memory_nodes_.
  std::vector<std::vector<DeviceTensorPtr>> continuous_memory_alloc_list_list_;
  std::vector<std::vector<size_t>> size_list_list_;
  std::vector<size_t> total_size_list_;
  std::vector<const DeviceContext *> continuous_memory_device_contexts_;
};  // namespace runtime

using DataPrepareActorPtr = std::shared_ptr<DataPrepareActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_DATA_PREPARE_ACTOR_H_
