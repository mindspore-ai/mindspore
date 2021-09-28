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
#include <stack>
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

// Gather actor is used in three places:
// 1. Entrance of sub funcgraph
// 2. call node which input0 is a funcgraph
// 3. There is some call nodes in the inputs of kernel graph.
// Gather actor will be used in the control flow. When the subgraph is called, the real parameters need to be put
// together and sent to the subgraph. At the same time, the entry of the subgraph needs to accept input data.
// Special in recursion, general inputs and call inputs of the kernel graph are used in stack mode, it needs to be
// collected at the entrance of the kernel graph.
class GatherActor : public OpActor<DeviceTensor> {
 public:
  GatherActor(const std::string &name, const std::vector<KernelWithIndex> &parameters, const bool need_branch_id_input,
              const AID switch_aid, const AID gather_aid, const int branch_id)
      : OpActor(name),
        data_nodes_(parameters),
        need_branch_id_input_(need_branch_id_input),
        switch_aid_(switch_aid),
        gather_aid_(gather_aid),
        local_branch_id_(branch_id) {
    device_contexts_.resize(parameters.size());
  }
  ~GatherActor() override = default;

  // Get the index of the parameter, the data_node needs to be the front node.
  size_t FetchDataNodePosition(const KernelWithIndex &data_node) const;

  // The gather actor run when receive the input data.
  void RunOpData(OpData<DeviceTensor> *input_data, OpContext<DeviceTensor> *context) override;
  // The gather actor run when receive the input control.
  void RunOpControl(AID *input_control, OpContext<DeviceTensor> *context) override;
  // The gather actor run when receive the input branch id.
  void CollectBranchId(const int branch_id, OpContext<DeviceTensor> *const context);
  void Init() override;

 private:
  friend class GraphScheduler;

  // Collect the inputs of gather actor.
  void FetchBackendInputNode(const FuncGraphPtr &func_graph, const ControlNodeParserPtr &parser);
  void FetchInputDeviceTensor(OpContext<DeviceTensor> *const context);
  // Check whether satisfy the condition for launch.
  bool CheckLaunchCondition(OpContext<DeviceTensor> *const context) const;
  void SendOutput(OpContext<DeviceTensor> *const context) const;
  // Erase input data and input controls when finish gather launch.
  void EraseInput(OpContext<DeviceTensor> *const context);

  // The device tensors for launch.
  std::vector<DeviceTensor *> input_device_tensors_;
  // The branch if for current step.
  int input_branch_id_{kInvalidBranchID};

  // Input data.
  std::unordered_map<int, std::unordered_map<size_t, std::stack<DeviceTensor *>>> input_data_;
  // Input branch ids is used to record the id corresponding receive from gather actor.
  // In control flow, sub funcgraph may be called in multiple places, and the output must be return to different
  // places. Therefore, the output of each subgraph will be connected to a switch actor, and the caller will send
  // its branch id to the gather actor of the subgraph. Then branch id will be sent by the gather actor to the
  // switch actor connected to the output.
  std::unordered_map<int, int> input_branch_ids_;

  // Output data.
  // Cache unique output data by output index to modify the output data effectively.
  std::vector<std::vector<OpDataUniquePtr<DeviceTensor>>> output_data_by_output_index_;
  //  The output_data_ corresponds to the output_data_arrows_ one by one.
  std::vector<OpData<DeviceTensor> *> output_data_;

  // Output arrows.
  std::vector<DataArrowPtr> output_result_arrows_;
  std::vector<AID> output_branch_arrows_;

  // Parameters of sub funcgraph, which is the front node.
  std::vector<KernelWithIndex> data_nodes_;
  std::vector<DeviceContext *> device_contexts_;
  // Pair<index, anfNode> points to the dependent device tensor store, anfNode is the key of the device tensor store.
  std::vector<std::pair<size_t, AnfNode *>> device_tensor_store_keys_;

  // When the output is a parameter of the subgraph, the gather actor needs to send the anfnode to the output actor,
  // so all the nodes that may send the device tensor to gather actor are recorded. When the anfnode needs to be sent
  // to the output actor, the corresponding backend node will be found from the map.
  std::unordered_map<AnfNodePtr, std::vector<KernelWithIndex>> front_to_backend_parameter_;

  // The dependent input data number.
  size_t input_datas_num_{0};
  // The dependent input controls number.
  size_t input_controls_num_{0};
  // Whether it needs to accept the branch id. When the gather actor is the input of the subgraph, it needs to receive
  // branch id sent by the subgraph caller, which will be true at this time.
  bool need_branch_id_input_;

  // Actor id that needs to send the branch id to it.
  // When the actor is corresponding to call node, the branch id needs to be sent to the input gather actor and output
  // switch actor of the called funcgraph. When the actor is the entrance of the funcgraph, the gather actor id is
  // empty, just need to send branch id to its output switch actor.
  const AID switch_aid_;
  const AID gather_aid_;

  // The branch id corresponding to the funcgraph to which the gather actor belongs.
  int local_branch_id_;
};

using GatherActorPtr = std::shared_ptr<GatherActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_GATHER_ACTOR_H_
