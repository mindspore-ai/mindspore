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
  LoopCountActor(const std::string &name, size_t loop_count, const AID &memory_manager_aid, const AID *debug_aid,
                 const AID *recorder_aid)
      : DebugAwareActor(name, KernelTransformType::kLoopCountActor, recorder_aid, memory_manager_aid, debug_aid),
        loop_count_(loop_count),
        current_count_(0),
        total_running_count_(0) {}

  ~LoopCountActor() override = default;

  // The loop count actor run when receive the input control.
  void RunOpControl(AID *const input_control, OpContext<DeviceTensor> *const context) override;

  // The callback waits for the memory manager actor to finish all the message processing.
  void OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) override;

  // The debug related operation interface.
  void SendDebugReq(OpContext<DeviceTensor> *const context) override;
  // The callback after debug finished.
  void OnDebugFinish(OpContext<DeviceTensor> *const context) override;

 private:
  friend class GraphScheduler;

  void IncreaseLoopCount(OpContext<DeviceTensor> *const context);
  void SendOutput(OpContext<DeviceTensor> *const context);

  // The loop count is constant, the current count is increased after each step running finished.
  size_t loop_count_;
  size_t current_count_;
  // The total running count represents the toal step running count.
  size_t total_running_count_;

  // The output controls contain the data prepare actor and output actor.
  AID data_prepare_aid_;
  AID output_aid_;
};

using LoopCountActorPtr = std::shared_ptr<LoopCountActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_LOOP_COUNT_ACTOR_H_
