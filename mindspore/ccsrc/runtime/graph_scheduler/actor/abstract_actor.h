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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_ABSTRACT_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_ABSTRACT_ACTOR_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <set>
#include <unordered_set>
#include <map>
#include "mindrt/include/actor/op_actor.h"
#include "runtime/graph_scheduler/actor/actor_common.h"
#include "runtime/graph_scheduler/device_tensor_store.h"
#include "runtime/graph_scheduler/device_tensor_copy_store.h"
#include "runtime/hardware/device_context.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceContext;

// The flag of output data.
constexpr size_t kOutputDataFlagInit = 0;
// Indicates that the output data destination is stack actor, and the output data cannot be reused.
constexpr size_t kOutputDataFlagToStack = 1;
// Indicates that the output data is the batch data, and send data in batches to increase the sending performance.
constexpr size_t kOutputDataFlagBatch = 2;
// Indicates that the output data is the last data in the batch.
constexpr size_t kOutputDataFlagLastBatch = 4;
// Indicates that the output data destination is the internal fusion actor, and uses the synchronous sending interface.
constexpr size_t kOutputDataFlagBetweenFusion = 8;
// Indicates that the output data destination is the fusion actor, and needs to use the fusion output index.
constexpr size_t kOutputDataFlagToFusion = 16;

// The abstract common attributes of actors. The actor inheritance relationship:  OpActor --> AbstractActor -->
// MemoryAwareActor --> DebugAwareActor --> KernelActor/DataSourceActor/CopyActor/LoopCountActor/OutputActor.
class AbstractActor : public OpActor<DeviceTensor> {
 public:
  explicit AbstractActor(const std::string &name, KernelTransformType type, const AID *recorder_aid)
      : OpActor(name),
        type_(type),
        recorder_aid_(recorder_aid),
        input_datas_num_(0),
        input_controls_num_(0),
        running_dependent_msg_num_(0),
        parent_fusion_actor_{nullptr},
        memory_alloc_insert_position_{nullptr},
        memory_free_insert_position_{nullptr} {}
  ~AbstractActor() override = default;

  bool IsActive(int msg_num) override { return msg_num >= running_dependent_msg_num_ ? true : false; }

  // The actor run when receive the input data.
  void RunOpData(OpData<DeviceTensor> *const input_data, OpContext<DeviceTensor> *const context) override;
  // The actor run when receive the input control.
  void RunOpControl(AID *const input_control, OpContext<DeviceTensor> *const context) override;
  // The actor run when receive the batch input data.
  void RunBatchOpData(std::vector<OpData<DeviceTensor> *> *const batch_input_data,
                      OpContext<DeviceTensor> *const context);

  // Get the position of node in the actor.
  virtual size_t FetchNodePosition(const KernelWithIndex &node) const { return 0; }

  // Get the member.
  KernelTransformType type() const { return type_; }
  const std::vector<const DeviceContext *> &device_contexts() const { return device_contexts_; }
  const std::vector<AnfNodePtr> &output_data_nodes() const { return output_data_nodes_; }
  const std::vector<std::pair<size_t, AnfNodePtr>> &device_tensor_store_keys() const {
    return device_tensor_store_keys_;
  }
  const std::vector<std::pair<AID, DataArrow *>> &input_data_arrow_aids() const { return input_data_arrow_aids_; }
  const std::vector<std::pair<AID, ControlArrow *>> &input_control_arrow_aids() const {
    return input_control_arrow_aids_;
  }
  const std::map<size_t, std::vector<AnfNodeWeakPtr>> &internal_parameters() const { return internal_parameters_; }
  const mindspore::HashMap<std::string, std::vector<DataArrowPtr>> &batch_output_data_arrows() const {
    return batch_output_data_arrows_;
  }
  const AbstractActor *parent_fusion_actor() const { return parent_fusion_actor_; }
  const mindspore::HashMap<std::string, std::shared_ptr<AbstractActor>> &sub_actors() const { return sub_actors_; }
  const std::unordered_set<std::string> &dependent_actors() const { return dependent_actors_; }
  AbstractActor *memory_alloc_insert_position() const { return memory_alloc_insert_position_; }
  AbstractActor *memory_free_insert_position() const { return memory_free_insert_position_; }

 protected:
  friend class GraphScheduler;
  friend class ControlNodeScheduler;
  friend class SchedulerHelper;

  // Check whether satisfy the actor running condition.
  virtual bool CheckRunningCondition(const OpContext<DeviceTensor> *context) const;
  // The actor run really when satisfy the actor running condition.
  virtual void Run(OpContext<DeviceTensor> *const context) {}

  // Erase input data and input controls when finish actor running.
  virtual void EraseInput(const OpContext<DeviceTensor> *context);

  // Init the member output_data_ and batch_output_data_ by output data arrows.
  void InitOutputData();
  // Update the output data before send output data.
  virtual void UpdateOutputData(OpData<DeviceTensor> *const output_data, const DataArrowPtr &data_arrow,
                                const AnfNodePtr &output_node, OpContext<DeviceTensor> *const context) {}
  // Send output to downstream actors to trigger running.
  virtual void SendOutput(OpContext<DeviceTensor> *const context);
  // Send recorder info to recorder actor.
  virtual void SendRecorderInfo(OpContext<DeviceTensor> *const context) const {}

  // Fetch the sub actor in the fusion actor by the name.
  AbstractActor *FetchSubActorInFusionActor(const std::string &sub_actor_name) const;

  KernelTransformType type_;

  // The device interface.
  std::vector<const DeviceContext *> device_contexts_;

  // The id of recorder actor. Send message to it for recording info.
  const AID *recorder_aid_;

  // The output_data_nodes_ and output_data_ corresponds to the output_data_arrows_ one by one.
  std::vector<AnfNodePtr> output_data_nodes_;
  // The second of pair indicates the output data flag. See constant prefixed with kOutputDataFalg for details.
  std::vector<std::pair<OpDataUniquePtr<DeviceTensor>, size_t>> output_data_;
  // Record the fusion output index for output data arrow.
  mindspore::HashMap<DataArrow *, size_t> data_arrow_to_fusion_actor_indexs_;
  // Used to send batch data in the message which RunBatchOpData needs, the key is the actor name of destination actor.
  mindspore::HashMap<std::string, std::vector<OpData<DeviceTensor> *>> batch_output_data_;
  mindspore::HashMap<std::string, std::vector<DataArrowPtr>> batch_output_data_arrows_;

  // When there is recursion in the graph, the actor will send data to the same stack actor multiple times. Since
  // messages are sent asynchronously between actors, there will be multiple messages that remain unprocessed in
  // the channel. In order to prevent old data from being overwritten, it is necessary to allocate a new op data,
  // and these op data will be uniformly cleared by the scheduler after the step ends.
  std::vector<OpDataUniquePtr<DeviceTensor>> to_stack_data_;

  // The dependent device tensor stores, the dependent expression is pair<index, AnfNode>.
  // Index is the input position, AnfNode is the key of the device tensor store.
  std::vector<std::pair<size_t, AnfNodePtr>> device_tensor_store_keys_;
  // The device tensor stores which have the auto monad attribute.
  std::set<AnfNodePtr> auto_monad_device_tensor_stores_;

  // Map <output_index, internal_parameter> is used to update the shape of internal parameter node for inferring the
  // dynamic shape information of the nodes located at the boundary of the graph partition, such as heterogeneous
  // scenario and so on.
  std::map<size_t, std::vector<AnfNodeWeakPtr>> internal_parameters_;

  // The dependent input actors.
  std::vector<std::pair<AID, DataArrow *>> input_data_arrow_aids_;
  std::vector<std::pair<AID, ControlArrow *>> input_control_arrow_aids_;
  // The dependent inputs number.
  size_t input_datas_num_;
  size_t input_controls_num_;

  // The dependent messages number of actor running.
  int running_dependent_msg_num_;

  // Indicates whether the actor is in fusion actor.
  AbstractActor *parent_fusion_actor_;

  // The sub actors in the fusion actor are not spawned in the ActorMgr, so they do not participate in message
  // interaction, but only internal processing.
  mindspore::HashMap<std::string, std::shared_ptr<AbstractActor>> sub_actors_;

  // All actors that the actor depends on for execution, the dependent actors are expanded by the input data and input
  // controls. For example, ActorA->ActorB->ActorC, the expanded dependent actors of ActorC are ActorA and ActorB.
  std::unordered_set<std::string> dependent_actors_;

  // The information used for integration of dynamic and static memory.
  AbstractActor *memory_alloc_insert_position_;
  AbstractActor *memory_free_insert_position_;
};

using AbstractActorPtr = std::shared_ptr<AbstractActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_ABSTRACT_ACTOR_H_
