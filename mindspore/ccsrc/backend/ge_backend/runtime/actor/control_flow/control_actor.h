/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_CONTROLFLOW_CONTROL_ACTOR_H_
#define MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_CONTROLFLOW_CONTROL_ACTOR_H_

#include <vector>
#include <string>
#include <memory>
#include <map>
#include <set>
#include <unordered_map>
#include <stack>
#include <queue>
#include <utility>
#include <algorithm>
#include "backend/ge_backend/runtime/actor/actor_common.h"
#include "backend/ge_backend/runtime/actor/abstract_actor.h"
#include "backend/ge_backend/runtime/actor/memory_aware_actor.h"
#include "backend/ge_backend/runtime/actor/memory_manager_actor.h"

namespace mindspore {
namespace ge_backend {
namespace runtime {
// Op partial represents the partial structure, including a funcgraph and its real parameters, maybe device tensors
// or partials.
struct OpPartial;
using OpPartialPtr = std::shared_ptr<OpPartial>;
struct OpPartial {
  FuncGraph *func_graph_{nullptr};
  std::vector<std::pair<size_t, KernelTensorPtr>> kernel_tensors_;
  std::vector<std::pair<size_t, OpPartialPtr>> partials_;
};

// Op real parameters with branch ID represents the data sent by gather actor to entrance actor, including all real
// parameters and the id of the caller.
struct OpRealParameterWithBranchID {
  std::vector<std::pair<size_t, KernelTensorPtr>> kernel_tensors_;
  std::vector<std::pair<size_t, OpPartialPtr>> partials_;
  int branch_id_;
};
// The control actor is the base class of control flow actor.
class ControlActor : public MemoryAwareActor {
 public:
  ControlActor(const std::string &name, KernelTransformType type, const AID &memory_manager_aid,
               const std::vector<KernelWithIndex> &parameters, const AnfNodePtr &node);
  ~ControlActor() override = default;

  // Receive partial.
  virtual void RunOpPartial(const OpPartialPtr &partial, size_t position, OpContext<KernelTensor> *const context);

  // Receive branch id.
  virtual void RunBranchID(int branch_id, OpContext<KernelTensor> *const context);

  const std::vector<DataArrowPtr> &output_partial_arrows() const { return output_partial_arrows_; }
  const std::vector<AID> &output_branch_id_arrows() const { return output_branch_id_arrows_; }
  const std::unordered_map<size_t, OpPartialPtr> &local_partials() const { return local_partials_; }
  const std::unordered_map<size_t, std::pair<KernelTensorPtr, AnfNodePtr>> &local_kernel_tensors() const {
    return local_kernel_tensors_;
  }
  const std::vector<KernelWithIndex> &formal_parameters() const { return formal_parameters_; }
  const std::vector<std::pair<AID, DataArrow *>> &input_partial_arrow_aids() const { return input_partial_arrow_aids_; }
  const std::vector<AID> &input_branch_id_arrow_aids() const { return input_branch_id_arrow_aids_; }
  const std::map<size_t, std::set<KernelTensorPtr>> &ref_formal_parameter_kernel_tensors() const {
    return ref_formal_parameter_kernel_tensors_;
  }
  const std::map<size_t, std::set<KernelTensorPtr>> &ref_node_formal_parameter_kernel_tensors() const {
    return ref_node_formal_parameter_kernel_tensors_;
  }
  int branch_id() const { return output_branch_id_; }
  // Free memory by the dynamic ref count decremented. It corresponds to the EraseInput.
  void SendMemoryFreeReq(OpContext<KernelTensor> *const context) override;

  void set_start_time(double start_time) { start_time_ = start_time; }
  const AnfNodePtr &node() const { return node_; }

 protected:
  friend class ControlNodeScheduler;
  friend class SchedulerHelper;

  void Init() override;

  // The basic interfaces for op partial and op real parameter.
  void GetAllKernelTensors(const OpPartialPtr &op_partial, std::vector<KernelTensorPtr> *kernel_tensors);
  void GetAllKernelTensors(const OpRealParameterWithBranchID &op_real_parameter,
                           std::vector<KernelTensorPtr> *kernel_tensors);
  void IncreaseDynamicRefCount(const OpData<KernelTensor> *op_data) const;
  void IncreaseDynamicRefCount(const OpPartialPtr &op_partial);
  void IncreaseDynamicRefCount(const OpRealParameterWithBranchID &op_real_parameter);

  // Get the position of node in the input.
  size_t FetchNodePosition(const KernelWithIndex &node) const;

  // Get all input, including data, partial, branchid.
  virtual void FetchInput(OpContext<KernelTensor> *const context);
  void Run(OpContext<KernelTensor> *const context) override;
  bool CheckRunningCondition(const OpContext<KernelTensor> *context) const override;
  void UpdateOutputData(OpData<KernelTensor> *const output_data, const DataArrowPtr &data_arrow,
                        const AnfNodePtr &output_node, OpContext<KernelTensor> *const context) override;
  void CreateHeterDeviceTensor(KernelTensor *const node_kernel_tensor, KernelTensor *const input_kernel_tensor,
                               size_t index, OpContext<KernelTensor> *const context, const AnfNodePtr &node);
  void SendOutput(OpContext<KernelTensor> *const context) override;
  void EraseInput(const OpContext<KernelTensor> *context) override;

  // Increase the dynamic ref count by the outputs. It corresponds to the SendOutput.
  virtual void IncreaseDynamicRefCounts(OpContext<KernelTensor> *const context);

  // Input data.
  // 1.Input partial.
  // Record the partial received by each step, the key of the pair indicates the location of the partial.
  std::unordered_map<int, std::vector<std::pair<size_t, OpPartialPtr>>> input_op_partials_;
  // 2. Branch ids is used to record the id corresponding to the output branch.
  // In control flow, sub funcgraph may be called in multiple places, and the output must be return to different
  // places. Therefore, the output of each subgraph will be connected to a exit actor, and the caller will send
  // its branch id to the entrance actor of the subgraph. Then branch id will be sent by the entrance actor to
  // the exit actor connected to the output.
  // In a recursive scenario, the exit will sequentially receive the branch ids sent by the caller, and the exit
  // actor needs to store the branch ids in the stack, and pop up in turn when returning.
  std::unordered_map<int, std::stack<int>> input_branch_ids_;

  // Fetch data. After fetch input, all the input collected is saved here.
  std::vector<OpPartialPtr> input_partials_;
  std::vector<KernelTensorPtr> input_kernel_tensors_;

  // The lists of device tensors which need free by dynamic ref count, will be cleared at the end of step.
  std::queue<std::vector<KernelTensorPtr>> memory_free_lists_;

  // The exit actor needs to create a new device address and take out the ptr from the device tensor come from
  // the kernel actor. These new created device tensors are stored in the created device tensors.
  std::vector<KernelTensorPtr> created_kernel_tensors_;
  std::map<std::pair<int, device::DeviceType>, KernelTensorPtr> created_heter_kernel_tensors_;
  std::vector<KernelTensorPtr> last_step_created_kernel_tensors_;
  // In control flow, when the argument is not a dynamic len tuple but the parameter is, need create a new
  // real make tuple node for it.
  std::vector<FuncGraphPtr> created_new_graphs_;
  std::vector<AnfNodePtr> created_new_nodes_;
  // Input num.
  size_t input_partials_num_{0};
  size_t input_branch_ids_num_{0};

  // The dependent input actors.
  std::vector<std::pair<AID, DataArrow *>> input_partial_arrow_aids_;
  std::vector<AID> input_branch_id_arrow_aids_;

  // Output Arrows.
  std::vector<DataArrowPtr> output_partial_arrows_;

  std::set<int> input_need_disable_dynamic_ref_counts_;
  std::vector<bool> output_need_disable_dynamic_ref_counts_;

  std::vector<AID> output_branch_id_arrows_;
  // The branch id is the unique identifier of the control actor. In the control flow, there are multiple control
  // actors calling the same subgraph at the same time. At this time, the output of the subgraph needs to be returned
  // to the calling place according to the branch id.
  int output_branch_id_{0};

  // Partial data in local. When partial is only funcgraph without real parameter, it is stored inside the actor.
  std::unordered_map<size_t, OpPartialPtr> local_partials_;
  // Device tensor in control node, but not in kernel graph.
  std::unordered_map<size_t, std::pair<KernelTensorPtr, AnfNodePtr>> local_kernel_tensors_;

  // Cache output data by output index to modify the output data effectively.
  std::vector<std::vector<OpData<KernelTensor> *>> output_data_by_output_index_;

  // Formal parameters for control actor.
  std::vector<KernelWithIndex> formal_parameters_;
  // The device tensors of backend input nodes corresponding to ref formal parameters, the key is the position index of
  // formal parameter. Used to update the ptr of device tensors when receive the real parameters for ref nodes.
  std::map<size_t, std::set<KernelTensorPtr>> ref_formal_parameter_kernel_tensors_;
  std::map<size_t, std::set<KernelTensorPtr>> ref_node_formal_parameter_kernel_tensors_;

  // Count the time cost bewtween this actor to the end actors, when this actor is executed, set current time to the
  // start_time_ of the end actors and then when the end actors are executed, it will count the time cost between its
  // start_time_ and its current time, for example, set exit actor of kernel graph to its entrance actor to count the
  // execution time of the kernel graph.
  std::set<ControlActor *> end_actors_;
  double start_time_{0};

  // local node for control actor, such as return node for exit actor, switch node for switch actor.
  AnfNodePtr node_;
};

using ControlActorPtr = std::shared_ptr<ControlActor>;
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_CONTROLFLOW_CONTROL_ACTOR_H_
