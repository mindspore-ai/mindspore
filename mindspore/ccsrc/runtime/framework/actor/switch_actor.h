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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_SWITCH_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_SWITCH_ACTOR_H_

#include <vector>
#include <string>
#include <set>
#include <memory>
#include <utility>
#include <stack>
#include <unordered_map>
#include "runtime/framework/actor/actor_common.h"
#include "runtime/framework/device_tensor_store.h"
#include "runtime/framework/control_node_parser.h"
#include "mindrt/include/actor/switch_actor.h"
#include "runtime/hardware/device_context.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceContext;
using mindspore::session::KernelWithIndex;

constexpr size_t kSwitchInputNum = 4;
constexpr size_t kSwitchCondPos = 1;
constexpr size_t kSwitchPartialNum = 2;
constexpr size_t kSwitchLayerCondPos = 1;
constexpr size_t kSwitchLayerBranchPos = 2;
constexpr size_t kSwitchLayerInputNum = 3;
constexpr size_t kMaxSwitchCondSize = 8;
constexpr size_t kSwitchTrueBranchPos = 2;
constexpr size_t kSwitchFalseBranchPos = 3;
constexpr size_t kPartialFuncGraphPos = 1;
constexpr size_t kPartialInputStartPos = 2;
constexpr size_t kCallInputStartPos = 1;
constexpr size_t kMakeTupleInputStartPos = 1;

// Switch actor is used to execute the branch according to the input condition.
// Switch and SwitchLayer node will be converted to switch actor.
// The execution process is divided into:
// 1. Put input into the vector.
// 2. Check whether the input condition has been received.
// 3. Check whether all input from the branch corresponding to the index has been received.
// 4. Send the data to the corresponding branch.
// 5. Free Memory
class SwitchActor : public SwitchActorBase<DeviceTensor> {
 public:
  SwitchActor(const std::string &name, DeviceContext *device_context, const CNodePtr &node, const int branch_id,
              const bool need_branch_id_input)
      : SwitchActorBase(name),
        device_context_(device_context),
        node_(node),
        local_branch_id_(branch_id),
        need_branch_id_input_(need_branch_id_input) {}
  ~SwitchActor() override = default;

  void Init() override;

  // The switch actor run when receive the input data.
  void RunOpData(OpData<DeviceTensor> *input_data, OpContext<DeviceTensor> *const context);
  // The switch actor run when receive the input control.
  void RunOpControl(AID *input_control, OpContext<DeviceTensor> *context);
  // The switch actor run when receive the input branch id.
  void CollectBranchId(const int branch_id, OpContext<DeviceTensor> *const context);
  // Parse the input node information of the switch actor according to node_.
  void ParseInput(const ControlNodeParserPtr &parser);
  // Add input for all branches.
  void AddCommonInput(const AnfNodePtr &node);
  void AddSingleInput(const AnfNodePtr &node, size_t branch) { AddInput(node, branch); }
  // Fetch the input position of the data node.
  size_t FetchDataNodePosition(const AnfNodePtr &data_node) const;

 private:
  friend class GraphScheduler;

  void ParsePartialInput(const AnfNodePtr &node, const size_t branch_id);
  void ParseSwitchInput();
  void ParseSwitchLayerInput();
  // In control flow, the output of each subgraph is connected to a switch actor, and the switch actor is
  // initialized with the return node of the subgraph.
  void ParseReturnInput(const ControlNodeParserPtr &parser);
  // Initialize the size of the vector members.
  void InitVectorSize(const size_t num);
  // Get index from DeviceTensor.
  size_t GetIndex(const OpContext<DeviceTensor> *const context);
  // Add input for the branch.
  void AddInput(const AnfNodePtr &node, size_t branch);
  void AddInput(const KernelWithIndex node_with_index, const size_t branch);

  // Check whether satisfy the condition for send outputs.
  bool CheckLaunchCondition(OpContext<DeviceTensor> *const context) const;
  // Fetch the args of switch branch.
  void FetchInputDeviceTensor(OpContext<DeviceTensor> *const context);
  void SendOutput(OpContext<DeviceTensor> *const context);
  // Erase input data and input controls when finish switch launch.
  void EraseInput(OpContext<DeviceTensor> *const context);
  void SendMemoryFreeReq(OpContext<DeviceTensor> *const context);

  // Collect all the backend inputs of switch actor.
  void FetchInputNode(const ControlNodeParserPtr &parser);
  // All inputs of the switch actor, include weight and tensor.
  // Used to receive input data, the first input is the condition of switch.
  std::vector<KernelWithIndex> input_nodes_;
  // The position of the branch output in the input_nodes_.
  std::vector<std::vector<size_t>> branch_inputs_pos_;

  std::unordered_map<int, std::unordered_map<size_t, std::stack<DeviceTensor *>>> input_data_;

  std::unordered_map<int, std::unordered_map<AID *, size_t>> input_controls_;

  // Branch ids is used to record the id corresponding to the switch output branch.
  // In control flow, sub funcgraph may be called in multiple places, and the output must be return to different
  // places. Therefore, the output of each subgraph will be connected to a switch actor, and the caller will send
  // its branch id to the gather of the subgraph. Then branch id will be sent by the gather actor to the switch
  // actor connected to the output.
  // In a recursive scenario, the switch will sequentially receive the branch ids sent by the caller, and the switch
  // actor needs to store the branch ids in the stack, and pop up in turn when returning.
  std::unordered_map<int, std::stack<int>> input_branch_ids_;

  // Control arrows of different branches.
  std::vector<std::vector<AID>> output_branch_control_arrows_;
  // Branch id arrows of different branches.
  std::vector<std::vector<AID>> output_branch_branch_arrows_;
  // Result arrows of different branches.
  std::vector<std::vector<DataArrowPtr>> output_branch_result_arrows_;

  // When the output is a value node from switch actor, the actor needs to send the anfnode to the output actor,
  // so all the nodes that may send the device tensor to switch actor are recorded.
  std::vector<std::set<KernelWithIndex>> backend_parameters_;
  std::vector<std::vector<AnfNodePtr>> branch_total_inputs_;

  std::vector<FuncGraphPtr> branch_func_graph_;

  std::unordered_map<int, size_t> branch_id_to_index_;

  // Pair<index, anfNode> points to the dependent device tensor store, anfNode is the key of the device tensor store.
  std::vector<std::pair<size_t, AnfNode *>> device_tensor_store_keys_;

  std::vector<DeviceTensor *> input_device_tensors_;

  // Save the DeviceContext of input_nodes_, which is used to release the DeviceTensor.
  const DeviceContext *device_context_;

  // The id of memory manager actor. Send message to it for alloc and free memory.
  const AID memory_manager_aid_;
  // The dependent input data number.
  size_t input_datas_num_{0};
  // The dependent input controls number.
  size_t input_controls_num_{0};
  CNodePtr node_;

  // The branch id corresponding to the funcgraph to which the gather actor belongs.
  int local_branch_id_;
  // Whether it needs to accept the branch id. When the switch actor is the output of the subgraph, it needs to receive
  // branch id sent by the gather actor of subgraph, which will be true at this time.
  bool need_branch_id_input_;

  //  The output_data_ corresponds to the output_data_arrows_ one by one.
  std::vector<std::vector<OpDataUniquePtr<DeviceTensor>>> output_data_;
};

using SwitchActorPtr = std::shared_ptr<SwitchActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_SWITCH_ACTOR_H_
