/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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
#include <map>

#include "backend/ms_backend/runtime/graph_scheduler/control_flow/control_node_parser.h"
#include "include/runtime/hardware_abstract/kernel_base/device_tensor_store.h"
#include "backend/ms_backend/runtime/actors/base/actor_common.h"
#include "backend/ms_backend/runtime/actors/base/abstract_actor.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "include/utils/stub_tensor.h"
#include "ir/tensor.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceContext;
using mindspore::session::KernelWithIndex;
using mindspore::tensor::TensorPtr;

// The output actor is used to receive the output result of actor which represents the graph output.
class OutputActor : public AbstractActor {
 public:
  OutputActor(const std::string &name, size_t loop_count, size_t outputs_num,
              const std::vector<KernelWithIndex> &summary_nodes)
      : AbstractActor(name, KernelTransformType::kOutputActor, nullptr),
        loop_count_(loop_count),
        current_count_(0),
        summary_nodes_(summary_nodes),
        outputs_num_(outputs_num),
        current_outputs_num_(0) {
    outputs_.resize(outputs_num);
    output_nodes_.resize(outputs_num);
    output_kernel_tensors_.resize(outputs_num);
    device_contexts_.resize(outputs_num);
  }
  ~OutputActor() override = default;

  // The output actor collects loop count when receive the input control of loop count actor.
  void RunOpControl(AID *const input_control, OpContext<KernelTensor> *const context) override;

  // The output actor collects output result when receive the data of actor.
  void RunOpData(OpData<KernelTensor> *const input_data, OpContext<KernelTensor> *const context) override;

  void FetchParameterInput(OpContext<KernelTensor> *const context);

  void HandleOutput();

  // The graph output need be set new device address every step or loop, to avoid that the device address
  // context of tensor be rewritten in the next step or next loop.
  void UpdateOutputDeviceAddress();

  // Summary node will keep the inputs, so if the input(except parameter, weight) size changes in dynamic shape,
  // the input device address will be reuse in a wrong way. So we should free summary node inputs after usage.
  void FreeSummaryNodeMem();

  // Get the member.
  size_t loop_count() const { return loop_count_; }
  size_t outputs_num() const { return outputs_num_; }
  const std::vector<TensorPtr> &outputs() const { return outputs_; }
  const std::vector<KernelTensorPtr> &output_kernel_tensors() const { return output_kernel_tensors_; }
  const std::vector<TypePtr> &output_types() const { return output_types_; }

  void ResetState(OpContext<KernelTensor> *const context) {
    current_outputs_num_ = 0;
    current_count_ = 0;
  }

  // Set the stub output node and flatten them to vector by output position.
  void SetStubOutput(const stub::StubNodePtr &stub_output);

 protected:
  void Init() override;

 private:
  friend class GraphScheduler;
  friend class ControlNodeScheduler;

  TensorPtr CreateOutputTensor(const AnfNodePtr &output_node, size_t output_index, size_t output_position,
                               OpContext<KernelTensor> *const context,
                               const KernelTensorPtr &old_kernel_tensor = nullptr);

  // The output device memory will be taken over by tensor in the last loop, otherwise needs to free the memory.
  // 1.Avoid the memory leak when memory used by dynamic ref count in the control flow scene.
  // 2.Alloc the new memory in the next step using the new shape size in the dynamic shape scene.
  void FreeOutputNodeMem();

  // Clear output nodes and tensors in cache.
  void ClearOutputCache();

  // The loop count is constant, the current count is increased after each step running finished.
  // Collect the output result in the last loop which is represented by "loop_count_ - current_count_ == 1".
  size_t loop_count_;
  size_t current_count_;

  // The outputs.
  std::vector<KernelWithIndex> summary_nodes_;
  std::vector<TensorPtr> outputs_;
  std::vector<KernelWithIndex> output_nodes_;
  std::vector<KernelTensorPtr> output_kernel_tensors_;
  std::vector<TypePtr> output_types_;
  size_t outputs_num_;
  size_t current_outputs_num_;

  std::map<KernelTensor *, KernelTensorPtr> old_to_new_kernel_tensor_;

  // All flatten stub node(TensorNode, ScalarNode, StubString) by output position, used in graph pipeline case.
  std::vector<stub::StubNodePtr> flatten_stub_nodes_{};
};

using OutputActorPtr = std::shared_ptr<OutputActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_OUTPUT_ACTOR_H_
