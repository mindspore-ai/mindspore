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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_LOOP_COUNT_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_LOOP_COUNT_ACTOR_H_

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <map>
#include <utility>
#include "runtime/framework/actor/actor_common.h"
#include "runtime/framework/actor/debug_aware_actor.h"
#include "runtime/framework/device_tensor_store.h"
#include "runtime/framework/control_node_parser.h"

namespace mindspore {
namespace runtime {
// The loop count actor is used to receive the control of tail kernel actor to represent the end of one step
// and decide whether to loop execution by loop count.
class LoopCountActor : public DebugAwareActor {
 public:
  LoopCountActor(std::string name, size_t loop_count, const AID memory_manager_aid, const AID *debug_aid,
                 const AID *recorder_aid)
      : DebugAwareActor(name),
        loop_count_(loop_count),
        current_count_(0),
        total_running_count_(0),
        input_controls_num_(0),
        memory_manager_aid_(memory_manager_aid),
        debug_aid_(debug_aid),
        recorder_aid_(recorder_aid) {}

  ~LoopCountActor() override = default;

  void Init() override;

  // The loop count actor run when receive the input control.
  void RunOpControl(AID *const input_control, OpContext<DeviceTensor> *const context) override;

  // The memory related operation interface.
  void SendMemoryAllocReq(OpContext<DeviceTensor> *const context) override;
  // The callback after memory alloc finished.
  void OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) override;

  // The debug related operation interface.
  void SendDebugReq(OpContext<DeviceTensor> *const context) override;
  // The callback after debug finished.
  void OnDebugFinish(OpContext<DeviceTensor> *const context) override;

 private:
  friend class GraphScheduler;

  void IncreaseLoopCount(OpContext<DeviceTensor> *const context);
  void SendOutput(OpContext<DeviceTensor> *const context);

  bool CheckLoopCountIncreaseCondition(OpContext<DeviceTensor> *const context);
  // The loop count is constant, the current count is increased after each step running finished.
  size_t loop_count_;
  size_t current_count_;
  // The total running count represents the toal step running count.
  size_t total_running_count_;

  // The dependent input controls number.
  // In the multi-branch output scenario of the control flow, the control of each branch needs to be recorded
  // separately with the branch id as the key. When the output has only one branch, the branch id is 0.
  size_t input_controls_num_;

  // The output controls contain the data source actors and the no input kernel actors and output actor.
  std::vector<AID> data_source_aids_;
  std::vector<AID> no_input_kernel_aids_;
  AID output_aid_;

  // The id of memory manager actor. Send message to it for alloc continuous memory before next step running.
  const AID memory_manager_aid_;
  // The id of debug actor. Send message to it for debug before loop count actor exits.
  const AID *debug_aid_;
  // The id of recorder actor. Send message to it for clearing recorder info before loop count actor exits.
  const AID *recorder_aid_;

  // The nodes need continuous memory, which must allocate in the begin of step running. The first bool of pair
  // expresses the inputs of node need continuous memory, the second bool of pair expresses the outputs of node need
  // continuous memory.
  std::map<std::pair<CNodePtr, DeviceContext *>, std::pair<bool, bool>> continuous_memory_nodes_;
  // The members for continuous memory alloc fetched by continuous_memory_nodes_.
  std::vector<std::vector<DeviceTensorPtr>> continuous_memory_alloc_list_list_;
  std::vector<std::vector<size_t>> size_list_list_;
  std::vector<size_t> total_size_list_;
  std::vector<const DeviceContext *> device_contexts_;
};

using LoopCountActorPtr = std::shared_ptr<LoopCountActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_LOOP_COUNT_ACTOR_H_
