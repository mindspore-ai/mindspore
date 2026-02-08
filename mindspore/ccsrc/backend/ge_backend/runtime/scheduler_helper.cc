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

#include "backend/ge_backend/runtime/scheduler_helper.h"
#include "primitive/framework_ops.h"
#include "primitive/array_ops.h"
#include "backend/ge_backend/runtime/actor/actor_dump.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "utils/log_adapter.h"
#include "utils/convert_utils.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"

namespace mindspore {
namespace ge_backend {
namespace runtime {
namespace {
void CollectControlActors(const ActorSet *actor_set, std::vector<AbstractActorPtr> *actors) {
  MS_EXCEPTION_IF_NULL(actor_set);
  MS_EXCEPTION_IF_NULL(actors);
  if (actor_set->control_actors_ != nullptr) {
    const auto &control_actor_set = actor_set->control_actors_;
    for (auto &switch_actor : control_actor_set->switch_actors_) {
      MS_EXCEPTION_IF_NULL(switch_actor);
      (void)actors->emplace_back(static_cast<AbstractActorPtr>(switch_actor));
    }
    for (auto &gather_actor : control_actor_set->gather_actors_) {
      MS_EXCEPTION_IF_NULL(gather_actor);
      (void)actors->emplace_back(static_cast<AbstractActorPtr>(gather_actor));
    }
    for (auto &entrance_actor : control_actor_set->entrance_actors_) {
      MS_EXCEPTION_IF_NULL(entrance_actor);
      (void)actors->emplace_back(static_cast<AbstractActorPtr>(entrance_actor));
    }
    for (auto &exit_actor : control_actor_set->exit_actors_) {
      MS_EXCEPTION_IF_NULL(exit_actor);
      (void)actors->emplace_back(static_cast<AbstractActorPtr>(exit_actor));
    }
    for (auto &stack_actor : control_actor_set->stack_actors_) {
      MS_EXCEPTION_IF_NULL(stack_actor);
      (void)actors->emplace_back(static_cast<AbstractActorPtr>(stack_actor));
    }
  }
}
}  // namespace

std::vector<AbstractActorPtr> SchedulerHelper::CollectActors(const ActorSet *actor_set) {
  MS_EXCEPTION_IF_NULL(actor_set);
  std::vector<AbstractActorPtr> actors;

  if (actor_set->data_prepare_actor_ != nullptr) {
    (void)actors.emplace_back(static_cast<AbstractActorPtr>(actor_set->data_prepare_actor_));
  }
  for (auto &data_source_actor : actor_set->data_source_actors_) {
    MS_EXCEPTION_IF_NULL(data_source_actor);
    (void)actors.emplace_back(static_cast<AbstractActorPtr>(data_source_actor));
  }
  for (auto &super_kernel_actor : actor_set->super_kernel_actors_) {
    MS_EXCEPTION_IF_NULL(super_kernel_actor);
    (void)actors.emplace_back(static_cast<AbstractActorPtr>(super_kernel_actor));
  }
  if (actor_set->loop_count_actor_ != nullptr) {
    (void)actors.emplace_back(static_cast<AbstractActorPtr>(actor_set->loop_count_actor_));
  }
  if (actor_set->output_actor_ != nullptr) {
    (void)actors.emplace_back(static_cast<AbstractActorPtr>(actor_set->output_actor_));
  }
  CollectControlActors(actor_set, &actors);
  return actors;
}

bool SchedulerHelper::HasMonadControl(const AnfNodePtr &input_node, const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(input_node);
  MS_EXCEPTION_IF_NULL(graph);
  const mindspore::HashSet<PrimitivePtr, PrimitiveHasher, PrimitiveEqual> auto_monad_prims = {
    prim::kPrimDepend, prim::kPrimUpdateState, prim::kPrimLoad};
  if (IsOneOfPrimitiveCNode(input_node, auto_monad_prims) || HasAbstractMonad(input_node)) {
    return true;
  }

  // The subgraph input.
  if (IsInternalParameter(input_node, graph)) {
    auto front_output_with_index = graph->GetOriginFrontNodeByInternalParameter(input_node);
    auto front_output_node = front_output_with_index.first;
    MS_EXCEPTION_IF_NULL(front_output_node);
    if (IsOneOfPrimitiveCNode(front_output_node, auto_monad_prims) || HasAbstractMonad(front_output_node)) {
      MS_LOG(INFO) << "The graph " << graph->graph_id()
                   << " has monad control from internal parameter: " << input_node->DebugString()
                   << ", front output node: " << front_output_node->fullname_with_scope();
      return true;
    }
  }

  return false;
}

void SchedulerHelper::AddDeviceTensorStore(const AnfNodePtr &anf_node, const KernelTensorPtr &kernel_tensor) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  MS_EXCEPTION_IF_NULL(kernel_tensor->device_address());
  MS_LOG(DEBUG) << "Add device tensor store:" << kernel_tensor << " for node:" << anf_node.get()->DebugString()
                << " node addr:" << anf_node.get() << " device type:" << kernel_tensor->GetDeviceType();
  DeviceTensorStore::GetInstance().Insert(const_cast<AnfNode *>(anf_node.get()), kernel_tensor);
  kernel_tensor->ClearFlag(device::kDeviceAddressFlagNotUsed);
  UpdateRefCount(kernel_tensor, true);
}

void SchedulerHelper::AddMonadDeviceTensorStore(AbstractActor *const to_actor, const CNodePtr &kernel,
                                                const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(graph);
  // Ref node monad device tensor store.
  if (common::AnfAlgo::HasNodeAttr(kAttrRefNodeMonadOutputIdx, kernel)) {
    auto output_idx = common::AnfAlgo::GetNodeAttr<size_t>(kernel, kAttrRefNodeMonadOutputIdx);
    const auto &origin_pair = graph->GetRefNodeRecursive({kernel, output_idx});
    auto front_node = AnfAlgo::FetchFrontNodeByBackendNode(origin_pair.first, *graph);
    MS_EXCEPTION_IF_NULL(front_node);
    if (IsPersistentDeviceTensor(front_node)) {
      MS_LOG(INFO) << to_actor->GetAID().Name() << ", kernel:" << kernel->fullname_with_scope()
                   << " add ref node monad device tensor store:" << front_node->fullname_with_scope();
      (void)to_actor->auto_monad_device_tensor_stores_.insert(front_node);
    }
  }

  // set in GEGraphOp in graph_compiler, for copy parameter in heterogeneous
  if (common::AnfAlgo::HasNodeAttr(kAttrRefNodeMonadInputIdx, kernel)) {
    auto input_idxes = common::AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(kernel, kAttrRefNodeMonadInputIdx);
    for (auto idx : input_idxes) {
      KernelWithIndex from_kernel_with_output_idx = common::AnfAlgo::GetPrevNodeOutput(kernel, idx, false);
      auto front_node = AnfAlgo::FetchFrontNodeByBackendNode(from_kernel_with_output_idx.first, *graph);
      MS_EXCEPTION_IF_NULL(front_node);
      if (IsPersistentDeviceTensor(front_node)) {
        MS_LOG(INFO) << to_actor->GetAID().Name() << ", kernel:" << kernel->fullname_with_scope()
                     << " add input node monad device tensor store:" << front_node->fullname_with_scope();
        (void)to_actor->auto_monad_device_tensor_stores_.insert(front_node);
      }
    }
  }

  // Input node monad device tensor store.
  if (!common::AnfAlgo::HasMonadInput(kernel)) {
    return;
  }

  // Super kernel actor need fetch by the input device tensor store.
  if (to_actor->type_ == KernelTransformType::kSuperKernelActor) {
    for (size_t i = 0; i < common::AnfAlgo::GetInputTensorNum(kernel); ++i) {
      KernelWithIndex from_kernel_with_output_idx = common::AnfAlgo::GetPrevNodeOutput(kernel, i, false);
      auto front_node = AnfAlgo::FetchFrontNodeByBackendNode(from_kernel_with_output_idx.first, *graph);
      MS_EXCEPTION_IF_NULL(front_node);
      if (IsPersistentDeviceTensor(front_node)) {
        MS_LOG(INFO) << to_actor->GetAID().Name() << ", kernel:" << kernel->fullname_with_scope()
                     << " add input node monad device tensor store:" << front_node->fullname_with_scope();
        (void)to_actor->auto_monad_device_tensor_stores_.insert(front_node);
      }
    }
  } else {
    // Kernel actor can fetch by the device tensor store key directly.
    const auto &device_tensor_store_keys = to_actor->device_tensor_store_keys_;
    (void)std::for_each(device_tensor_store_keys.begin(), device_tensor_store_keys.end(), [&](const auto &store_key) {
      MS_EXCEPTION_IF_NULL(store_key.second);
      MS_LOG(INFO) << to_actor->GetAID().Name() << ", kernel:" << kernel->fullname_with_scope()
                   << " add input node monad device tensor store:" << store_key.second->fullname_with_scope();
      (void)to_actor->auto_monad_device_tensor_stores_.insert(store_key.second);
    });
  }
}

void SchedulerHelper::AddDataArrow(AbstractActor *const from_actor, AbstractActor *const to_actor,
                                   size_t from_output_index, size_t to_input_index, const AnfNodePtr &from_kernel) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_LOG(DEBUG) << "Add data arrow from actor:" << from_actor->GetAID() << " index:" << from_output_index
                << " to actor:" << to_actor->GetAID() << " to index:" << to_input_index
                << " from kernel:" << (from_kernel == nullptr ? "null" : from_kernel->fullname_with_scope());

  auto data_arrow = std::make_shared<DataArrow>(from_output_index, to_actor->GetAID(), to_input_index);
  (void)from_actor->output_data_arrows_.emplace_back(data_arrow);
  (void)from_actor->output_data_nodes_.emplace_back(from_kernel);
  to_actor->input_datas_num_++;
  (void)to_actor->input_data_arrow_aids_.emplace_back(std::make_pair(from_actor->GetAID(), data_arrow.get()));

  if (from_kernel == nullptr) {
    return;
  }
  // Update the reference count of from_kernel.
  auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(from_kernel, from_output_index, false);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto device_tensor = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);
  // The superkernel actor is linked by input parameter, maybe the not used parameter.
  if (to_actor->type() != KernelTransformType::kSuperKernelActor) {
    kernel_tensor->ClearFlag(device::kDeviceAddressFlagNotUsed);
  }
  // The device address of super kernel actor can't be changed, so set the max reference count.
  if (IsControlFlowActor(to_actor->type()) || (from_actor->type_ == KernelTransformType::kSuperKernelActor) ||
      (to_actor->type_ == KernelTransformType::kSuperKernelActor)) {
    UpdateRefCount(kernel_tensor, true);
  }

  if (IsControlFlowActor(to_actor->type())) {
    device_tensor->SetNodeIndex(from_kernel, from_output_index);
  }
}

void SchedulerHelper::AddResultArrow(AbstractActor *const from_actor, OutputActor *const to_actor,
                                     const AnfNodePtr &from_kernel, size_t from_output_index, size_t output_position) {
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_EXCEPTION_IF_NULL(from_kernel);

  if (from_actor == nullptr) {
    (void)to_actor->device_tensor_store_keys_.emplace_back(output_position, from_kernel);
  } else {
    auto result_arrow = std::make_shared<DataArrow>(from_output_index, to_actor->GetAID(), output_position);
    (void)from_actor->output_data_arrows_.insert(from_actor->output_data_arrows_.begin(), result_arrow);
    (void)from_actor->output_data_nodes_.insert(from_actor->output_data_nodes_.begin(), from_kernel);
    to_actor->input_datas_num_++;
    (void)to_actor->input_data_arrow_aids_.emplace_back(std::make_pair(from_actor->GetAID(), result_arrow.get()));
  }

  if (!AnfAlgo::OutputAddrExist(from_kernel, from_output_index, false)) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, from_kernel)
      << "#dmsg#Runtime error info:#dmsg#" << from_kernel->DebugString() << " index:" << from_output_index
      << " device address does not exist";
  }
  auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(from_kernel, from_output_index, false);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto device_tensor = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);
  kernel_tensor->ClearFlag(device::kDeviceAddressFlagNotUsed);
  // The output actor need use the relevant information of node to create output tensor.
  device_tensor->SetNodeIndex(from_kernel, from_output_index);
  // The device tensor of graph out need be taken over by host tensor, so set the max reference count.
  UpdateRefCount(kernel_tensor, true);

  MS_LOG(DEBUG) << "Add result arrow from actor:" << (from_actor != nullptr ? from_actor->GetAID().Name() : "null")
                << " to actor:" << to_actor->GetAID() << " from kernel"
                << (from_kernel == nullptr ? "null" : from_kernel->DebugString()) << " device address:" << device_tensor
                << " original ref count:" << kernel_tensor->original_ref_count()
                << " ref count:" << kernel_tensor->ref_count()
                << " dynamic ref count:" << kernel_tensor->dynamic_ref_count();
}

void SchedulerHelper::AddControlArrow(AbstractActor *const from_actor, AbstractActor *const to_actor) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);

  // Check the control arrow whether exists.
  auto iter = std::find_if(from_actor->output_control_arrows_.begin(), from_actor->output_control_arrows_.end(),
                           [&to_actor](const auto &output_control_arrow) {
                             return output_control_arrow->to_op_id_.Name() == to_actor->GetAID().Name();
                           });
  if (iter != from_actor->output_control_arrows_.end()) {
    // The stack actor can only link the single control arrow.
    if (to_actor->type_ == KernelTransformType::kStackActor) {
      MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#The control arrow between "
                                 << from_actor->GetAID().Name() << " and " << to_actor->GetAID().Name()
                                 << " is repeated.";
    }
    return;
  }

  auto control_arrow = std::make_shared<ControlArrow>(to_actor->GetAID());
  (void)from_actor->output_control_arrows_.emplace_back(control_arrow);
  to_actor->input_controls_num_++;
  (void)to_actor->input_control_arrow_aids_.emplace_back(std::make_pair(from_actor->GetAID(), control_arrow.get()));
  MS_LOG(DEBUG) << "Add control arrow from actor:" << from_actor->GetAID() << " to actor:" << to_actor->GetAID();
}

void SchedulerHelper::AddPartialArrow(ControlActor *const from_actor, ControlActor *const to_actor, size_t from_index,
                                      size_t to_index) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);
  auto op_arrow = std::make_shared<DataArrow>(from_index, to_actor->GetAID(), to_index);
  (void)from_actor->output_partial_arrows_.emplace_back(op_arrow);
  to_actor->input_partials_num_++;
  (void)to_actor->input_partial_arrow_aids_.emplace_back(from_actor->GetAID(), op_arrow.get());
}

void SchedulerHelper::AddBranchIDArrow(ControlActor *const from_actor, ControlActor *const to_actor) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);
  (void)from_actor->output_branch_id_arrows_.emplace_back(to_actor->GetAID());
  (void)to_actor->input_branch_id_arrow_aids_.emplace_back(from_actor->GetAID());
  to_actor->input_branch_ids_num_++;
}

void SchedulerHelper::AddLoopBodyControlArrow(AbstractActor *from_actor, EntranceActor *to_actor) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_LOG(DEBUG) << "Link loop body control arrow from:" << from_actor->GetAID() << " to actor:" << to_actor->GetAID();
  auto control_arrow = std::make_shared<ControlArrow>(to_actor->GetAID());
  (void)from_actor->output_control_arrows_.emplace_back(control_arrow);
  to_actor->loop_body_input_controls_nums_++;
  (void)to_actor->loop_body_input_control_arrow_aids_.emplace_back(from_actor->GetAID());
}

void SchedulerHelper::AddDataWithBranchIDArrow(GatherActor *const gather_actor, const EntranceActor *entrance_actor,
                                               const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(gather_actor);
  MS_EXCEPTION_IF_NULL(entrance_actor);
  (void)gather_actor->output_data_with_branch_id_arrows_[func_graph.get()].emplace_back(entrance_actor->GetAID());
}

void SchedulerHelper::AddDataArrowForExitActor(ExitActor *const exit_actor, AbstractActor *const to_actor,
                                               size_t from_index, size_t to_index, int branch_id) {
  MS_EXCEPTION_IF_NULL(exit_actor);
  MS_EXCEPTION_IF_NULL(to_actor);

  MS_LOG(DEBUG) << "Link data arrow from actor:" << exit_actor->GetAID() << " from index:" << from_index
                << " to actor:" << to_actor->GetAID() << " to index:" << to_index;
  auto data_arrow = std::make_shared<DataArrow>(from_index, to_actor->GetAID(), to_index);
  (void)exit_actor->output_branch_data_arrows_[branch_id].emplace_back(data_arrow);
  (void)to_actor->input_data_arrow_aids_.emplace_back(std::make_pair(exit_actor->GetAID(), data_arrow.get()));
}

void SchedulerHelper::AddPartialArrowForExitActor(ExitActor *const exit_actor, ControlActor *const to_actor,
                                                  size_t from_index, size_t to_index, int branch_id) {
  MS_EXCEPTION_IF_NULL(exit_actor);
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_LOG(DEBUG) << "Link partial arrow from actor:" << exit_actor->GetAID() << " from index:" << from_index
                << " to actor:" << to_actor->GetAID() << " to index:" << to_index;
  auto partial_arrow = std::make_shared<DataArrow>(from_index, to_actor->GetAID(), to_index);
  (void)exit_actor->output_branch_partial_arrows_[branch_id].emplace_back(partial_arrow);
  (void)to_actor->input_partial_arrow_aids_.emplace_back(exit_actor->GetAID(), partial_arrow.get());
}

void SchedulerHelper::AddControlArrowForExitActor(ExitActor *from_actor, AbstractActor *to_actor, int branch_id) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);

  MS_LOG(DEBUG) << "Link control arrow from:" << from_actor->GetAID() << " to:" << to_actor->GetAID();
  (void)from_actor->output_branch_control_arrows_[branch_id].emplace_back(to_actor->GetAID());
  to_actor->input_controls_num_++;
  (void)to_actor->input_control_arrow_aids_.emplace_back(std::make_pair(from_actor->GetAID(), nullptr));
}

void SchedulerHelper::DumpActorSet(const ActorSet *actor_set, std::ofstream &ofs) {
  MS_EXCEPTION_IF_NULL(actor_set);
  DumpDataPrepareActor(actor_set->data_prepare_actor_, ofs);
  DumpDSActors(actor_set->data_source_actors_, ofs);
  DumpSuperKernelActors(actor_set->super_kernel_actors_, ofs);
  if (actor_set->control_actors_ == nullptr) {
    DumpNoInputKernelActors(actor_set->no_input_kernel_actors_, ofs);
  }
  DumpLoopCountActor(actor_set->loop_count_actor_, ofs);
  DumpOutputActor(actor_set->output_actor_, ofs);
  DumpControlActors(actor_set->control_actors_, ofs);
}

void SchedulerHelper::DumpFormatActorSet(const ActorSet *actor_set, std::ofstream &ofs) {
  MS_EXCEPTION_IF_NULL(actor_set);
  try {
    MS_LOG(DEBUG) << "Start dump format actor set:" << actor_set->name_;
    if (actor_set->control_actors_ != nullptr) {
      for (const auto &exit_actor : actor_set->control_actors_->exit_actors_) {
        if (exit_actor->node() != nullptr) {
          continue;
        }
        auto actors = TopoSortForActor(exit_actor.get());
        ActorInfoMap actor_info;
        ofs << "\n\nBase Block : "
            << exit_actor->GetAID().Name().substr(0, exit_actor->GetAID().Name().find(kExitActorNameSuffix)) << "\n\n";
        for (size_t i = 0; i < actors.size(); ++i) {
          DumpActorInfo(actors[i], i, &actor_info, ofs);
        }
      }
      return;
    }

    auto actors = TopoSortForActor(actor_set->output_actor_.get());
    ActorInfoMap actor_info;
    for (size_t i = 0; i < actors.size(); ++i) {
      DumpActorInfo(actors[i], i, &actor_info, ofs);
    }
    MS_LOG(DEBUG) << "End dump format actor set:" << actor_set->name_;
  } catch (const std::exception &e) {
    MS_LOG(INFO) << "Failed to dump actor set:" << actor_set->name_ << ", msg: " << e.what();
  }
}
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore
