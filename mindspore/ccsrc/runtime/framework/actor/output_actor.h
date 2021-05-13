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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_OUTPUT_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_OUTPUT_ACTOR_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <algorithm>
#include "runtime/framework/device_tensor_store.h"
#include "runtime/framework/actor/actor_common.h"
#include "runtime/hardware/device_context.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "ir/tensor.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceContext;
using mindspore::session::KernelWithIndex;
using mindspore::tensor::TensorPtr;

// The output actor is used to receive the output result of actor which represents the graph output.
class OutputActor : public ActorBase {
 public:
  OutputActor(std::string name, size_t loop_count, size_t outputs_num)
      : ActorBase(name),
        loop_count_(loop_count),
        current_count_(0),
        outputs_num_(outputs_num),
        current_outputs_num_(0) {
    outputs_.resize(outputs_num);
    output_nodes_.resize(outputs_num);
    device_contexts_.resize(outputs_num);
  }
  ~OutputActor() override = default;

  // The output actor collects loop count when receive the input control of loop count actor.
  void CollectLoopCount(size_t loop_count, OpContext<DeviceTensor> *context);

  // The output actor collects output result when receive the data of actor.
  void CollectOutput(const AnfNodePtr &output_node, size_t output_index, size_t output_position,
                     OpContext<DeviceTensor> *context);

  std::vector<TensorPtr> &outputs() { return outputs_; }

 private:
  friend class GraphScheduler;

  // The loop count is constant, the current count is increased after each step running finished.
  // Collect the output result in the last loop which is represented by "loop_count_ - current_count_ == 1".
  size_t loop_count_;
  size_t current_count_;

  // The outputs.
  std::vector<TensorPtr> outputs_;
  std::vector<KernelWithIndex> output_nodes_;
  std::vector<const DeviceContext *> device_contexts_;
  size_t outputs_num_;
  size_t current_outputs_num_;

  // Pair<index, anfNode> points to the dependent device tensor store, anfNode is the key of the device tensor store.
  std::vector<std::pair<size_t, AnfNodePtr>> device_tensor_store_keys_;
};

using OutputActorPtr = std::shared_ptr<OutputActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_OUTPUT_ACTOR_H_
