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
#include "mindrt/include/actor/op_actor.h"
#include "runtime/framework/actor/actor_common.h"
#include "runtime/framework/device_tensor_store.h"
#include "runtime/framework/device_tensor_copy_store.h"
#include "runtime/hardware/device_context.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceContext;

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
        running_dependent_msg_num_(0) {}
  virtual ~AbstractActor() = default;

  bool IsActive(int msg_num) override { return msg_num >= running_dependent_msg_num_ ? true : false; }

  // The actor run when receive the input data.
  void RunOpData(OpData<DeviceTensor> *const input_data, OpContext<DeviceTensor> *const context) override;
  // The actor run when receive the input control.
  void RunOpControl(AID *const input_control, OpContext<DeviceTensor> *const context) override;

  // Get the position of node in the actor.
  virtual size_t FetchNodePosition(const AnfNodePtr &node) const { return 0; }

  // Get the member.
  KernelTransformType type() const { return type_; }
  const std::vector<const DeviceContext *> &device_contexts() const { return device_contexts_; }
  const std::vector<AnfNodePtr> &output_data_nodes() const { return output_data_nodes_; }
  const std::vector<std::pair<size_t, AnfNodePtr>> &device_tensor_store_keys() const {
    return device_tensor_store_keys_;
  }
  const std::vector<AID> &input_data_arrow_aids() const { return input_data_arrow_aids_; }
  const std::vector<AID> &input_control_arrow_aids() const { return input_control_arrow_aids_; }

 protected:
  friend class GraphScheduler;
  friend class ControlNodeScheduler;

  // Check whether satisfy the actor running condition.
  virtual bool CheckRunningCondition(const OpContext<DeviceTensor> *context) const;
  // The actor run really when satisfy the actor running condition.
  virtual void Run(OpContext<DeviceTensor> *const context) {}

  // Erase input data and input controls when finish actor running.
  virtual void EraseInput(const OpContext<DeviceTensor> *context);

  // Update the output data before send output data.
  virtual void UpdateOutputData(OpData<DeviceTensor> *const output_data, const DataArrowPtr &data_arrow,
                                const AnfNodePtr &output_node, OpContext<DeviceTensor> *const context) {}
  // Send output to downstream actors to trigger running.
  virtual void SendOutput(OpContext<DeviceTensor> *const context);
  // Send recorder info to recorder actor.
  virtual void SendRecorderInfo(OpContext<DeviceTensor> *const context) const {}

  KernelTransformType type_;

  // The device interface.
  std::vector<const DeviceContext *> device_contexts_;

  // The id of recorder actor. Send message to it for recording info.
  const AID *recorder_aid_;

  // The output_data_nodes_ and output_data_ corresponds to the output_data_arrows_ one by one.
  std::vector<AnfNodePtr> output_data_nodes_;
  std::vector<OpDataUniquePtr<DeviceTensor>> output_data_;
  // When there is recursion in the graph, the actor will send data to the same stack actor multiple times. Since
  // messages are sent asynchronously between actors, there will be multiple messages that remain unprocessed in
  // the channel. In order to prevent old data from being overwritten, it is necessary to allocate a new op data,
  // and these op data will be uniformly cleared by the scheduler after the step ends.
  std::vector<OpDataPtr<DeviceTensor>> to_stack_data_;

  // The dependent device tensor stores, the dependent expression is pair<index, AnfNode>.
  // Index is the input position, AnfNode is the key of the device tensor store.
  std::vector<std::pair<size_t, AnfNodePtr>> device_tensor_store_keys_;
  // The device tensor stores which have the auto monad attribute.
  std::set<AnfNodePtr> auto_monad_device_tensor_stores_;

  // The dependent input actors.
  std::vector<AID> input_data_arrow_aids_;
  std::vector<AID> input_control_arrow_aids_;
  // The dependent inputs number.
  size_t input_datas_num_;
  size_t input_controls_num_;

  // The dependent messages number of actor running.
  int running_dependent_msg_num_;
};

using AbstractActorPtr = std::shared_ptr<AbstractActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_ABSTRACT_ACTOR_H_
