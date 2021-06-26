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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_GATHER_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_GATHER_ACTOR_H_

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <utility>
#include <algorithm>
#include "runtime/framework/device_tensor_store.h"
#include "runtime/framework/actor/actor_common.h"
#include "runtime/framework/control_node_parser.h"
#include "runtime/hardware/device_context.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/session/kernel_graph.h"
#include "ir/tensor.h"

namespace mindspore {
namespace runtime {

constexpr size_t kReturnInputPos = 1;

// Gather actor is the entrance of sub funcgraph. Graph input is sent to it and sent to other actors by gather actor.
class GatherActor : public OpActor<DeviceTensor> {
 public:
  GatherActor(const std::string &name, const std::vector<AnfNodePtr> &parameters, const AID loop_count_aid,
              const AID output_aid)
      : OpActor(name), data_nodes_(parameters), loop_count_aid_(loop_count_aid), output_aid_(output_aid) {}
  ~GatherActor() override = default;

  // Get the index of the parameter, the data_node needs to be the front node.
  size_t FetchDataNodePosition(const AnfNodePtr &data_node) const;

  // The kernel actor run when receive the input data.
  void RunOpData(OpData<DeviceTensor> *input_data, OpContext<DeviceTensor> *context) override;

  void Init() override;

 private:
  friend class GraphScheduler;

  // Collect the inputs of gather actor.
  void FetchBackendInputNode(const FuncGraphPtr &func_graph, const ControlNodeParserPtr &parser);
  void FetchInputDeviceTensor(OpContext<DeviceTensor> *context);
  // Check whether satisfy the condition for launch.
  bool CheckLaunchCondition(OpContext<DeviceTensor> *context) const;
  void SendOutput(OpContext<DeviceTensor> *context) const;
  // Erase input data and input controls when finish gather launch.
  void EraseInput(OpContext<DeviceTensor> *context);

  // The device tensors for launch.
  std::vector<DeviceTensor *> input_device_tensors_;

  DeviceContext *device_contexts_;

  std::vector<DataArrowPtr> output_result_arrows_;

  // Parameters of sub funcgraph, which is the front node.
  std::vector<AnfNodePtr> data_nodes_;

  // When the output is a parameter of the subgraph, the gather actor needs to send the anfnode to the output actor,
  // so all the nodes that may send the device tensor to gather actor are recorded. When the anfnode needs to be sent
  // to the output actor, the corresponding backend node will be found from the map.
  std::unordered_map<AnfNodePtr, std::vector<KernelWithIndex>> front_to_backend_parameter_;

  // The dependent input data number.
  size_t input_datas_num_{0};
  // The dependent input controls number.
  size_t input_controls_num_{0};

  const AID loop_count_aid_;
  const AID output_aid_;

  // Cache unique output data by output index to modify the output data effectively.
  std::vector<std::vector<OpDataUniquePtr<DeviceTensor>>> output_data_by_output_index_;
  //  The output_data_ corresponds to the output_data_arrows_ one by one.
  std::vector<OpData<DeviceTensor> *> output_data_;

  // When the result of the graph is sent to the output actor, the gather actor of the graph needs
  // to send branch_id to the output actor to determine the corresponding weight.
  int branch_id_{kInvalidBranchID};
};

using GatherActorPtr = std::shared_ptr<GatherActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_GATHER_ACTOR_H_
