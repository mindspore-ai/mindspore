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
#include "mindrt/include/actor/op_actor.h"
#include "runtime/framework/actor/actor_common.h"
#include "runtime/framework/device_tensor_store.h"
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

  // Get the position of node in the actor.
  virtual size_t FetchNodePosition(const AnfNodePtr &node) const { return 0; }

 protected:
  friend class GraphScheduler;

  // Check whether satisfy the actor running condition.
  bool CheckRunningCondition(const OpContext<DeviceTensor> *context) const;
  // Erase input data and input controls when finish actor running.
  void EraseInput(const OpContext<DeviceTensor> *const context);

  KernelTransformType type_;

  // The device interface.
  std::vector<const DeviceContext *> device_contexts_;

  // The id of recorder actor. Send message to it for recording info.
  const AID *recorder_aid_;

  // The output result arrows of graph output.
  std::vector<DataArrowPtr> output_result_arrows_;

  // The dependent device tensor stores,  the dependent expression is pair<index, AnfNode>.
  // Index is the input position, AnfNode is the key of the device tensor store.
  std::vector<std::pair<size_t, AnfNodePtr>> device_tensor_store_keys_;

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
