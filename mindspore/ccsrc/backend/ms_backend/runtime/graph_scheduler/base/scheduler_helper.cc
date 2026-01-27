/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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

#include "backend/ms_backend/runtime/graph_scheduler/base/scheduler_helper.h"
#include <memory>
#include <string>
#include <vector>
#include <set>
#include <utility>
#include "primitive/framework_ops.h"
#include "primitive/array_ops.h"
#include "backend/ms_backend/runtime/graph_scheduler/dump/actor_dump.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "utils/log_adapter.h"
#include "utils/convert_utils.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"
#include "backend/ms_backend/runtime/graph_scheduler/base/parameter_store.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_u.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/backend/common/device_address_utils.h"

namespace mindspore {
namespace runtime {
size_t SchedulerHelper::fusion_actor_index_ = 0;

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

void UpdateDataArrowRefCount(AbstractActor *const to_actor, size_t to_input_index,
                             const KernelTensorPtr &kernel_tensor) {
  MS_LOG(DEBUG) << "Process shape depend attribute for actor : " << to_actor->GetAID().Name();
  bool need_increase_ref_count = true;
  auto to_kernel_actor = dynamic_cast<KernelActor *>(to_actor);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  static const bool enable_infer_boost = ms_context->IsEnableInferBoost();
  if (to_kernel_actor != nullptr && !enable_infer_boost) {
    auto to_kernel = to_kernel_actor->kernel();
    auto cnode = to_kernel->cast<CNodePtr>();
    if (cnode != nullptr) {
      MS_LOG(DEBUG) << "Process shape depend attribute for cnode : " << cnode->fullname_with_scope();
      const auto &only_depend_shape_attr = common::AnfAlgo::GetCNodePrimitiveAttr(cnode, kAttrOnlyDependShape);
      if (only_depend_shape_attr != nullptr) {
        auto only_depend_shape = GetValue<std::vector<bool>>(only_depend_shape_attr);
        if (only_depend_shape.size() <= to_input_index) {
          MS_LOG(DEBUG) << "to_input_index : " << to_input_index
                        << " is out of range, only_depend_shape size : " << only_depend_shape.size();
        } else {
          auto is_shape_depend = only_depend_shape[to_input_index];
          MS_LOG(DEBUG) << "only_depend_shape[" << to_input_index << "] : " << is_shape_depend;
          if (is_shape_depend) {
            need_increase_ref_count = false;
          }
        }
      }
    }
  }
  if (!need_increase_ref_count) {
    kernel_tensor->UpdateFlag(device::kDeviceAddressFlagNullptr);
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
  for (auto &kernel_actor : actor_set->kernel_actors_) {
    MS_EXCEPTION_IF_NULL(kernel_actor);
    (void)actors.emplace_back(static_cast<AbstractActorPtr>(kernel_actor));
  }
  for (auto &super_kernel_actor : actor_set->super_kernel_actors_) {
    MS_EXCEPTION_IF_NULL(super_kernel_actor);
    (void)actors.emplace_back(static_cast<AbstractActorPtr>(super_kernel_actor));
  }
  for (auto &any_type_kernel_actor : actor_set->any_type_kernel_actors_) {
    MS_EXCEPTION_IF_NULL(any_type_kernel_actor);
    (void)actors.emplace_back(static_cast<AbstractActorPtr>(any_type_kernel_actor));
  }
  for (auto &memory_actor : actor_set->memory_actors_) {
    MS_EXCEPTION_IF_NULL(memory_actor);
    (void)actors.emplace_back(static_cast<AbstractActorPtr>(memory_actor));
  }
  for (auto &copy_actor : actor_set->copy_actors_) {
    MS_EXCEPTION_IF_NULL(copy_actor);
    (void)actors.emplace_back(static_cast<AbstractActorPtr>(copy_actor));
  }
  for (auto &fusion_actor : actor_set->fusion_actors_) {
    MS_EXCEPTION_IF_NULL(fusion_actor);
    (void)actors.emplace_back(static_cast<AbstractActorPtr>(fusion_actor));
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
  const auto &device_tensor = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);
  if (EnableInputOptimize()) {
    auto real_node = common::AnfAlgo::FetchRealNodeSkipMonadControl({anf_node, 0}).first;
    MS_EXCEPTION_IF_NULL(real_node);
    if (real_node->isa<Parameter>() && common::AnfAlgo::IsParameterWeight(real_node->cast<ParameterPtr>())) {
      // Push kernel tensor of weight into parameter store.
      // Push the kernel tensor if store of the position has no one.
      // If there are heterogeneous kernel tensors, push non cpu device address into store.
      auto graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
      MS_EXCEPTION_IF_NULL(graph_parameter_store);
      auto outer_idx = graph_parameter_store->GetFrontNodeToIndex(anf_node.get());
      auto store_kernel_tensor = graph_parameter_store->Fetch(outer_idx, 0);
      if (store_kernel_tensor == nullptr || store_kernel_tensor->device_address() == nullptr) {
        graph_parameter_store->Push(outer_idx, 0, kernel_tensor, SIZE_MAX);
        const auto &parameter_device = AnfAlgo::GetParameterDeviceStr(anf_node);
        if (!parameter_device.empty()) {
          if (parameter_device != kToCpu && parameter_device != kToRemote) {
            MS_LOG(EXCEPTION) << "Device of parameter is supposed to be \"CPU\" or \"Remote\" if it is set, but got "
                              << parameter_device;
          }
          if (parameter_device == kToCpu) {
            graph_parameter_store->SetOffloaded(outer_idx, 0, true);
            MS_LOG(INFO) << "Offloaded parameter:" << real_node->fullname_with_scope();
          }
        }
      } else if (store_kernel_tensor->device_address()->GetDeviceType() != device_tensor->GetDeviceType() &&
                 device_tensor->GetDeviceType() != device::DeviceType::kCPU) {
        graph_parameter_store->Push(outer_idx, 0, kernel_tensor, SIZE_MAX);
      } else {
        return;
      }

      MS_LOG(DEBUG) << "Add graph parameter store:" << kernel_tensor << " for node:" << anf_node.get()->DebugString()
                    << " node addr:" << anf_node.get() << " device type:" << kernel_tensor->GetDeviceType()
                    << ", outer idx:" << outer_idx;
      kernel_tensor->ClearFlag(device::kDeviceAddressFlagNotUsed);
      kernel_tensor->set_new_ref_count(SIZE_MAX);
      return;
    }
  }
  MS_LOG(DEBUG) << "Add device tensor store:" << kernel_tensor << " for node:" << anf_node.get()->DebugString()
                << " node addr:" << anf_node.get() << " device type:" << kernel_tensor->GetDeviceType();
  DeviceTensorStore::GetInstance().Insert(const_cast<AnfNode *>(anf_node.get()), kernel_tensor);
  kernel_tensor->ClearFlag(device::kDeviceAddressFlagNotUsed);
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

bool SchedulerHelper::IsSkipLaunchShapeRelatedOp(KernelActor *kernel_actor) {
  MS_EXCEPTION_IF_NULL(kernel_actor);
  if (kernel_actor->skip_launch_shape_related_op()) {
    return true;
  }

  auto &kernel = kernel_actor->kernel();
  MS_EXCEPTION_IF_NULL(kernel);

  // RealMakeTuple --> ShapeCalc pattern:
  // If ShapeCalc is not value depend for one input RealMakeTuple op, we can skip launch this RealMakeTuple.
  if (IsPrimitiveCNode(kernel, prim::kPrimRealMakeTuple)) {
    auto func_graph = kernel->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    auto manager = func_graph->manager();
    if (manager == nullptr) {
      manager = Manage(func_graph, true);
      func_graph->set_manager(manager);
    }

    const auto &users_set = manager->node_users()[kernel];
    bool can_skip_launch_real_make_tuple = true;
    for (const auto &item : users_set) {
      const auto &user_node = item.first;
      if (!user_node->isa<CNode>()) {
        can_skip_launch_real_make_tuple = false;
        break;
      }
      auto user_cnode = user_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(user_cnode);
      if (!IsPrimitiveCNode(user_cnode, prim::kPrimShapeCalc)) {
        can_skip_launch_real_make_tuple = false;
        break;
      }

      if (!common::AnfAlgo::HasNodeAttr(kAttrOnlyDependShape, user_cnode)) {
        can_skip_launch_real_make_tuple = false;
        break;
      }
      const auto &only_depend_shape = common::AnfAlgo::GetNodeAttr<std::vector<bool>>(user_cnode, kAttrOnlyDependShape);
      auto user_input_index = item.second;
      if (user_input_index < 1) {
        MS_LOG(EXCEPTION) << "The input index should start from 1, but got: " << user_input_index;
      }
      if (IntToSize(user_input_index) > only_depend_shape.size()) {
        MS_LOG(EXCEPTION) << "The input index[" << user_input_index
                          << "] is out of range, input size: " << only_depend_shape.size();
      }
      if (!only_depend_shape[user_input_index - 1]) {
        can_skip_launch_real_make_tuple = false;
        break;
      }
    }

    if (can_skip_launch_real_make_tuple) {
      return true;
    }
  }

  return false;
}

bool SchedulerHelper::IsSkipLaunchShapeRelatedOpV2(KernelRunner *kernel_actor) {
  MS_EXCEPTION_IF_NULL(kernel_actor);
  if (kernel_actor->skip_launch_shape_related_op()) {
    return true;
  }

  auto &kernel = kernel_actor->kernel();
  MS_EXCEPTION_IF_NULL(kernel);

  // RealMakeTuple --> ShapeCalc pattern:
  // If ShapeCalc is not value depend for one input RealMakeTuple op, we can skip launch this RealMakeTuple.
  if (IsPrimitiveCNode(kernel, prim::kPrimRealMakeTuple)) {
    auto func_graph = kernel->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    auto manager = func_graph->manager();
    if (manager == nullptr) {
      manager = Manage(func_graph, true);
      func_graph->set_manager(manager);
    }

    const auto &users_set = manager->node_users()[kernel];
    bool can_skip_launch_real_make_tuple = true;
    for (const auto &item : users_set) {
      const auto &user_node = item.first;
      if (!user_node->isa<CNode>()) {
        can_skip_launch_real_make_tuple = false;
        break;
      }
      auto user_cnode = user_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(user_cnode);
      if (!IsPrimitiveCNode(user_cnode, prim::kPrimShapeCalc)) {
        can_skip_launch_real_make_tuple = false;
        break;
      }

      if (!common::AnfAlgo::HasNodeAttr(kAttrOnlyDependShape, user_cnode)) {
        can_skip_launch_real_make_tuple = false;
        break;
      }
      const auto &only_depend_shape = common::AnfAlgo::GetNodeAttr<std::vector<bool>>(user_cnode, kAttrOnlyDependShape);
      auto user_input_index = item.second;
      if (user_input_index < 1) {
        MS_LOG(EXCEPTION) << "The input index should start from 1, but got: " << user_input_index;
      }
      if (IntToSize(user_input_index) > only_depend_shape.size()) {
        MS_LOG(EXCEPTION) << "The input index[" << user_input_index
                          << "] is out of range, input size: " << only_depend_shape.size();
      }
      if (!only_depend_shape[user_input_index - 1]) {
        can_skip_launch_real_make_tuple = false;
        break;
      }
    }

    if (can_skip_launch_real_make_tuple) {
      return true;
    }
  }

  return false;
}

bool SchedulerHelper::IsIgnoredInputAddress(AbstractActor *const to_actor, size_t to_input_index) {
  MS_EXCEPTION_IF_NULL(to_actor);
  if (to_actor->type() != KernelTransformType::kKernelActor) {
    return false;
  }

  auto kernel_actor = dynamic_cast<KernelActor *>(to_actor);
  auto &to_kernel = kernel_actor->kernel();
  MS_EXCEPTION_IF_NULL(to_kernel);

  if (IsSkipLaunchShapeRelatedOp(kernel_actor)) {
    kernel_actor->set_skip_launch_shape_related_op(true);
    return true;
  }

  MS_EXCEPTION_IF_NULL(to_actor->device_contexts_[0]);
  auto kernel_executor = to_actor->device_contexts_[0]->GetKernelExecutor();
  MS_EXCEPTION_IF_NULL(kernel_executor);
  if (kernel_executor->IsLaunchIgnoredInputAddressIdx(to_kernel, to_input_index)) {
    MS_LOG(INFO) << "Ignore the input address for kernel: " << to_kernel->fullname_with_scope()
                 << " with input index: " << to_input_index;
    return true;
  }

  return false;
}

bool SchedulerHelper::IsIgnoredInputAddressV2(KernelRunner *const to_actor, size_t to_input_index) {
  MS_EXCEPTION_IF_NULL(to_actor);
  if (to_actor->type() != KernelTransformType::kKernelActor) {
    return false;
  }

  auto kernel_actor = to_actor;
  auto &to_kernel = kernel_actor->kernel();
  MS_EXCEPTION_IF_NULL(to_kernel);

  if (IsSkipLaunchShapeRelatedOpV2(kernel_actor)) {
    kernel_actor->set_skip_launch_shape_related_op(true);
    return true;
  }

  MS_EXCEPTION_IF_NULL(to_actor->device_contexts_[0]);
  auto kernel_executor = to_actor->device_contexts_[0]->GetKernelExecutor();
  MS_EXCEPTION_IF_NULL(kernel_executor);
  if (kernel_executor->IsLaunchIgnoredInputAddressIdx(to_kernel, to_input_index)) {
    MS_LOG(INFO) << "Ignore the input address for kernel: " << to_kernel->fullname_with_scope()
                 << " with input index: " << to_input_index;
    return true;
  }

  return false;
}

size_t SchedulerHelper::GetIgnoredInputAddressCount(const AnfNodePtr &node, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(device_context);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(node);

  auto kernel_executor = device_context->GetKernelExecutor();
  MS_EXCEPTION_IF_NULL(kernel_executor);
  std::vector<size_t> ignored_input_addresses = kernel_executor->GetLaunchIgnoredInputAddressIdx(node);
  if (ignored_input_addresses.empty()) {
    return 0;
  }

  auto count = std::count_if(ignored_input_addresses.begin(), ignored_input_addresses.end(),
                             [input_num](size_t index) { return index < input_num; });
  return static_cast<size_t>(count);
}

void SchedulerHelper::AddDataArrow(AbstractActor *const from_actor, AbstractActor *const to_actor,
                                   size_t from_output_index, size_t to_input_index, const AnfNodePtr &from_kernel) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_LOG(DEBUG) << "Add data arrow from actor:" << from_actor->GetAID() << " index:" << from_output_index
                << " to actor:" << to_actor->GetAID() << " to index:" << to_input_index
                << " from kernel:" << (from_kernel == nullptr ? "null" : from_kernel->fullname_with_scope());
  // Check the data arrow legitimacy.
  if (IsControlFlowActor(to_actor->type()) && (from_actor->type() == KernelTransformType::kKernelActor) &&
      (to_actor->type() != KernelTransformType::kExitActor)) {
    MS_LOG(WARNING) << "Kernel actor:" << from_actor->GetAID().Name()
                    << " link data arrow to actor:" << to_actor->GetAID().Name() << " is not an exit actor.";
  }

  if (from_actor->type() == KernelTransformType::kKernelActor &&
      to_actor->type() == KernelTransformType::kKernelActor) {
    auto from_kernel_actor = dynamic_cast<KernelActor *>(from_actor);
    MS_EXCEPTION_IF_NULL(from_kernel_actor);
    if (IsSkipLaunchShapeRelatedOp(from_kernel_actor)) {
      from_kernel_actor->set_skip_launch_shape_related_op(true);
    }
  }

  // The continuous memory inpus need allocate memory in advance, so must be from the inside subgraph.
  if (to_actor->type() == KernelTransformType::kKernelActor) {
    auto to_kernel_actor = dynamic_cast<KernelActor *>(to_actor);
    MS_EXCEPTION_IF_NULL(to_kernel_actor);
    if (to_kernel_actor->inputs_continuous_memory() && (from_actor->type() != KernelTransformType::kKernelActor)) {
      MS_LOG(INTERNAL_EXCEPTION)
        << "#dmsg#Runtime error info:#dmsg#The continuous memory input is not from the inside subgraph, to actor: "
        << to_actor->GetAID().Name() << ", to input index: " << to_input_index
        << ", from actor: " << from_actor->GetAID().Name() << ", from output index: " << from_output_index;
    }
  }

  AddMemorySign(from_actor, to_actor);

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
  if (!(IsControlFlowActor(to_actor->type()) || (((from_actor->type_ == KernelTransformType::kSuperKernelActor) ||
                                                  (to_actor->type_ == KernelTransformType::kSuperKernelActor)) &&
                                                 !EnableKbkSubGraphExecute()))) {
    UpdateDataArrowRefCount(to_actor, to_input_index, kernel_tensor);
    GetUnusedRefCount(from_actor, to_actor, from_output_index, to_input_index, kernel_tensor);
  }

  if (IsControlFlowActor(to_actor->type())) {
    device_tensor->SetNodeIndex(from_kernel, from_output_index);
  }
}

void SchedulerHelper::GetUnusedRefCount(AbstractActor *const from_actor, AbstractActor *const to_actor,
                                        size_t from_input_index, size_t to_input_index,
                                        const KernelTensorPtr &kernel_tensor) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto device_tensor = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);
  if (from_actor->type() != KernelTransformType::kKernelActor ||
      to_actor->type() != KernelTransformType::kKernelActor) {
    return;
  }
  auto from_kernel_actor = dynamic_cast<KernelActor *>(from_actor);
  MS_EXCEPTION_IF_NULL(from_kernel_actor);
  auto to_kernel_actor = dynamic_cast<KernelActor *>(to_actor);
  MS_EXCEPTION_IF_NULL(to_kernel_actor);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  static const bool enable_infer_boost = ms_context->IsEnableInferBoost();
  if (enable_infer_boost) {
    return;
  }
  auto to_kernel = to_kernel_actor->kernel();
  auto cnode = to_kernel->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &only_depend_shape_attr = common::AnfAlgo::GetCNodePrimitiveAttr(cnode, kAttrOnlyDependShape);
  if (only_depend_shape_attr == nullptr) {
    return;
  }
  auto only_depend_shape = GetValue<std::vector<bool>>(only_depend_shape_attr);
  if (to_input_index >= only_depend_shape.size()) {
    MS_LOG(DEBUG) << "to_input_index : " << to_input_index
                  << " is out of range, only_depend_shape size : " << only_depend_shape.size();
    return;
  }
  if (only_depend_shape[to_input_index]) {
    kernel_tensor->UpdateFlag(device::kDeviceAddressFlagNullptr);
    from_kernel_actor->output_free_index_.emplace_back(from_input_index);
    MS_LOG(DEBUG) << "Add output free index:" << from_input_index
                  << " and null flag to device address:" << device_tensor
                  << " by only shape depend flag for actor:" << from_actor->GetAID();
  }
}

bool IsOnlyShapeDepend(AbstractActor *const to_actor, size_t to_index) {
  MS_EXCEPTION_IF_NULL(to_actor);
  if (to_actor->type() != KernelTransformType::kKernelActor) {
    return false;
  }
  auto to_kernel_actor = dynamic_cast<KernelActor *>(to_actor);
  MS_EXCEPTION_IF_NULL(to_kernel_actor);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  static const bool enable_infer_boost = ms_context->IsEnableInferBoost();
  if (enable_infer_boost) {
    return false;
  }
  auto to_kernel = to_kernel_actor->kernel();
  auto cnode = to_kernel->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &only_depend_shape_attr = common::AnfAlgo::GetCNodePrimitiveAttr(cnode, kAttrOnlyDependShape);
  if (only_depend_shape_attr == nullptr) {
    return false;
  }
  auto only_depend_shape = GetValue<std::vector<bool>>(only_depend_shape_attr);
  if (to_index >= only_depend_shape.size()) {
    MS_LOG(DEBUG) << "To index : " << to_index
                  << " is out of range, only_depend_shape size : " << only_depend_shape.size();
    return false;
  }
  return only_depend_shape[to_index];
}

void SchedulerHelper::InsertParameterIndexsForActor(AbstractActor *const to_actor,
                                                    const KernelWithIndex &front_node_with_idx,
                                                    const KernelWithIndex &from_kernel_with_output_idx,
                                                    const KernelWithIndex &to_kernel_with_input_idx,
                                                    const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_EXCEPTION_IF_NULL(front_node_with_idx.first);
  MS_LOG(DEBUG) << "Insert parameter index to actor:" << to_actor->GetAID()
                << " front node:" << front_node_with_idx.first->DebugString()
                << " index:" << front_node_with_idx.second;

  // Obtain the corresponding front node from back node.
  ParameterStore &parameterStore = ParameterStore::GetInstance();
  auto cur_graph_parameter_store = parameterStore.GetGraphParameterStore();
  size_t real_outer_idx = cur_graph_parameter_store->GetFrontNodeToIndex(front_node_with_idx.first.get());
  // The index of the font node is flattened
  size_t real_inner_idx = front_node_with_idx.second;
  auto cur_kernel_tensor =
    AnfAlgo::GetOutputKernelTensor(from_kernel_with_output_idx.first, from_kernel_with_output_idx.second, false);
  MS_EXCEPTION_IF_NULL(cur_kernel_tensor);
  auto cur_device_tensor = cur_kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(cur_device_tensor);
  // The superkernel actor is linked by input parameter, maybe the not used parameter.
  if (to_actor->type() != KernelTransformType::kSuperKernelActor) {
    cur_kernel_tensor->ClearFlag(device::kDeviceAddressFlagNotUsed);
  }
  // Cal ref count
  auto real_node = common::AnfAlgo::FetchRealNodeSkipMonadControl(from_kernel_with_output_idx).first;
  MS_EXCEPTION_IF_NULL(real_node);
  if (real_node->isa<Parameter>() && common::AnfAlgo::IsParameterWeight(real_node->cast<ParameterPtr>())) {
    cur_graph_parameter_store->SetUserCnt(real_outer_idx, real_inner_idx, SIZE_MAX);
  } else if (graph->IsRefOutputMapValue(from_kernel_with_output_idx)) {
    MS_LOG(INFO) << "Ref input: " << from_kernel_with_output_idx.first->DebugString()
                 << ", index: " << from_kernel_with_output_idx.second;
    cur_graph_parameter_store->SetUserCnt(real_outer_idx, real_inner_idx, SIZE_MAX);
  } else if (IsOnlyShapeDepend(to_actor, to_kernel_with_input_idx.second)) {
    MS_LOG(DEBUG) << "Is only shape depend to actor:" << to_actor->GetAID()
                  << " and skip increase user count for outer index:" << real_outer_idx
                  << " and inner index:" << real_inner_idx;
  } else {
    MS_LOG(DEBUG) << "Insert parameter store user count to actor:" << to_actor->GetAID()
                  << " front node:" << front_node_with_idx.first->DebugString() << " out index:" << real_outer_idx
                  << " inner index:" << real_inner_idx << " device address:" << cur_device_tensor->ToString();
    cur_graph_parameter_store->IncreaseUserCnt(real_outer_idx, real_inner_idx);
    cur_kernel_tensor->ClearFlag(device::kDeviceAddressFlagNotUsed);
  }
  if (IsControlFlowActor(to_actor->type())) {
    cur_device_tensor->SetNodeIndex(from_kernel_with_output_idx.first, from_kernel_with_output_idx.second);
  }
  // Save to_actor info into parameter_index
  ParameterInfo cur_param_info{front_node_with_idx, real_outer_idx};
  to_actor->InsertParameterIndexs(to_kernel_with_input_idx.second, cur_param_info);
  UpdateDataArrowRefCount(to_actor, to_kernel_with_input_idx.second, cur_kernel_tensor);
}

void SchedulerHelper::AddResultParameter(AbstractActor *const from_actor, OutputActor *const to_actor,
                                         const KernelWithIndex &kernel_with_index, DeviceContext *device_context,
                                         size_t output_position) {
  if (!EnableInputOptimize()) {
    return;
  }

  auto front_node_with_index = kernel_with_index;
  auto from_kernel = kernel_with_index.first;
  MS_EXCEPTION_IF_NULL(from_kernel);
  MS_EXCEPTION_IF_NULL(device_context);
  auto graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
  MS_EXCEPTION_IF_NULL(graph_parameter_store);
  if (!graph_parameter_store->IsFrontNodeInStore(from_kernel.get())) {
    front_node_with_index = graph_parameter_store->GetRealFrontNode(kernel_with_index);
    from_kernel = front_node_with_index.first;
  }
  auto outer_idx = graph_parameter_store->GetFrontNodeToIndex(from_kernel.get());
  ParameterInfo parameter_info{front_node_with_index, outer_idx};
  to_actor->InsertParameterIndexs(output_position, parameter_info);
  MS_LOG(DEBUG) << "Set parameter user count to max for outer index:" << outer_idx
                << " inner index:" << front_node_with_index.second;
  graph_parameter_store->SetUserCnt(outer_idx, front_node_with_index.second, SIZE_MAX);

  const auto &kernel_tensor = graph_parameter_store->Fetch(outer_idx, front_node_with_index.second);
  if (kernel_tensor != nullptr && kernel_tensor->device_address() != nullptr) {
    kernel_tensor->ClearFlag(device::kDeviceAddressFlagNotUsed);
    MS_LOG(DEBUG) << "Add result arrow from actor:" << (from_actor != nullptr ? from_actor->GetAID().Name() : "null")
                  << " to actor:" << to_actor->GetAID() << " from kernel"
                  << (from_kernel == nullptr ? "null" : from_kernel->DebugString())
                  << " kernel tensor:" << kernel_tensor;
  }

  // Set the device contexts of to_actor.
  if (output_position >= to_actor->device_contexts_.size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#The output position is out of range.";
  }
  to_actor->device_contexts_[output_position] = device_context;
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
  MS_LOG(DEBUG) << "Add result arrow from actor:" << (from_actor != nullptr ? from_actor->GetAID().Name() : "null")
                << " to actor:" << to_actor->GetAID() << " from kernel"
                << (from_kernel == nullptr ? "null" : from_kernel->DebugString())
                << " device address:" << device_tensor;

  // Set the device contexts of to_actor.
  if (output_position >= to_actor->device_contexts_.size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#The output position is out of range.";
  }
  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {device_tensor->GetDeviceType(), device_tensor->device_id()});
  to_actor->device_contexts_[output_position] = device_context;
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

  // No need add control arrow if already exists data arrow in from and to actor.
  if (from_actor->type() == KernelTransformType::kKernelActor &&
      to_actor->type() == KernelTransformType::kKernelActor) {
    const auto &input_data_arrows = to_actor->input_data_arrow_aids();
    if (std::any_of(input_data_arrows.begin(), input_data_arrows.end(),
                    [&from_actor](const std::pair<AID, DataArrow *> &input_data_arrow_pair) {
                      return input_data_arrow_pair.first.Name() == from_actor->GetAID().Name();
                    })) {
      MS_LOG(INFO) << "No need add control arrow, because already exists data arrow in from actor: "
                   << from_actor->GetAID().Name() << " and to actor: " << to_actor->GetAID().Name();
      return;
    }
  }

  auto control_arrow = std::make_shared<ControlArrow>(to_actor->GetAID());
  (void)from_actor->output_control_arrows_.emplace_back(control_arrow);
  to_actor->input_controls_num_++;
  (void)to_actor->input_control_arrow_aids_.emplace_back(std::make_pair(from_actor->GetAID(), control_arrow.get()));
  MS_LOG(DEBUG) << "Add control arrow from actor:" << from_actor->GetAID() << " to actor:" << to_actor->GetAID();
  AddMemorySign(from_actor, to_actor);
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

void SchedulerHelper::AddFormalParameterDeviceTensor(ControlActor *const from_actor, size_t from_index,
                                                     const AnfNodePtr &input_node, const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(input_node);
  MS_EXCEPTION_IF_NULL(graph);
  // Graph mode does not support dynamic shape and ref node.
  if ((graph->is_graph_run_mode() && !EnableKbkSubGraphExecute()) || graph->is_any_type_input()) {
    return;
  }

  if (!common::AnfAlgo::HasAbstractRef(input_node)) {
    return;
  }

  auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(input_node, 0, false);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto device_tensor = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);
  (void)from_actor->ref_formal_parameter_kernel_tensors_[from_index].insert(kernel_tensor);
  if (graph->IsRefOutputMapValue({input_node, 0})) {
    MS_LOG(DEBUG) << "Add device address:" << device_tensor << " from index:" << from_index
                  << " parameter:" << input_node->DebugString() << " for actor:" << from_actor->GetAID();
    (void)from_actor->ref_node_formal_parameter_kernel_tensors_[from_index].insert(kernel_tensor);
  }

  kernel_tensor->ClearFlag(device::kDeviceAddressFlagNotUsed);
  device_tensor->SetNodeIndex(input_node, 0);
}

void SchedulerHelper::ConvertDataArrowToControlArrow(AbstractActor *const from_actor, AbstractActor *const to_actor,
                                                     const DataArrowPtr &data_arrow, size_t data_arrow_index) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_EXCEPTION_IF_NULL(data_arrow);
  MS_EXCEPTION_IF_CHECK_FAIL((data_arrow_index < from_actor->output_data_nodes_.size()), "Index out of range.");
  auto &need_converted_node = from_actor->output_data_nodes_[data_arrow_index];
  MS_EXCEPTION_IF_NULL(need_converted_node);

  // Skip the ref node because its reference count cann‘t be recalculated correctly.
  auto kernel_tensor =
    AnfAlgo::GetOutputKernelTensor(need_converted_node, IntToSize(data_arrow->from_output_index_), false);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  if (TEST_FLAG(kernel_tensor->flag(), device::kDeviceAddressFlagRefNode)) {
    MS_LOG(INFO) << "Skip the invalid data arrow of ref node, from actor:" << from_actor->GetAID().Name()
                 << ", from index:" << data_arrow->from_output_index_ << ", to actor:" << to_actor->GetAID().Name()
                 << ", to index:" << data_arrow->to_input_index_;
    return;
  }

  auto kernel_info = dynamic_cast<KernelInfo *>(need_converted_node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  const auto &somas_outputs = kernel_info->somas_output_result();
  if (kernel_info->IsTensorEnableSomas(somas_outputs, data_arrow->from_output_index_)) {
    MS_LOG(INFO) << "Skip the invalid data arrow of somas inner address, from actor:" << from_actor->GetAID().Name()
                 << ", from index:" << data_arrow->from_output_index_ << ", to actor:" << to_actor->GetAID().Name()
                 << ", to index:" << data_arrow->to_input_index_;
    return;
  }

  // Erase the output data arrow in from actor.
  const auto &arrow_addr = (*(from_actor->output_data_arrows_.begin() + SizeToLong(data_arrow_index))).get();
  (void)from_actor->output_data_arrows_.erase(from_actor->output_data_arrows_.begin() + SizeToLong(data_arrow_index));
  (void)from_actor->output_data_nodes_.erase(from_actor->output_data_nodes_.begin() + SizeToLong(data_arrow_index));

  // Erase the input data arrow aid in to actor.
  bool to_actor_erase = false;
  for (auto iter = to_actor->input_data_arrow_aids_.begin(); iter != to_actor->input_data_arrow_aids_.end(); ++iter) {
    if ((*iter).first == from_actor->GetAID() && (*iter).second == arrow_addr) {
      (void)to_actor->input_data_arrow_aids_.erase(iter);
      to_actor_erase = true;
      to_actor->input_datas_num_--;
      break;
    }
  }
  if (to_actor_erase == false) {
    MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Erase no input data arrow, from actor:"
                               << from_actor->GetAID().Name() << ", to actor:" << to_actor->GetAID().Name()
                               << ", data arrow index:" << data_arrow_index;
  }

  for (auto &output_data_arrow : from_actor->output_data_arrows_) {
    MS_EXCEPTION_IF_NULL(output_data_arrow);
    if (output_data_arrow->from_output_index_ != data_arrow->from_output_index_) {
      continue;
    }
    if ((output_data_arrow->to_op_id_.Name().find(kExitActorNameSuffix) != std::string::npos) ||
        (output_data_arrow->to_op_id_.Name().find(kOutputActorNameSuffix) != std::string::npos)) {
      break;
    }
  }
  MS_LOG(INFO) << "Erase the invalid data arrow, from actor:" << from_actor->GetAID().Name()
               << ", from index:" << data_arrow->from_output_index_ << ", to actor:" << to_actor->GetAID().Name()
               << ", to index:" << data_arrow->to_input_index_;

  // Add the control arrow.
  SchedulerHelper::AddControlArrow(from_actor, to_actor);
}

void SchedulerHelper::FuseDataArrowsToBatchDataArrow(AbstractActor *const actor) {
  MS_EXCEPTION_IF_NULL(actor);
  // Count the number of the same destination actor.
  mindspore::HashMap<std::string, size_t> to_actor_count;
  for (const auto &data_arrow : actor->output_data_arrows()) {
    MS_EXCEPTION_IF_NULL(data_arrow);
    ++(to_actor_count[data_arrow->to_op_id_.Name()]);
  }

  // Sign and add the batch data arrow.
  for (auto &data_arrow : actor->output_data_arrows()) {
    MS_EXCEPTION_IF_NULL(data_arrow);
    auto &to_op_name = data_arrow->to_op_id_.Name();
    // The output data cannot be reused whose destination is stack actor, and cannot to be fused.
    if ((to_actor_count[to_op_name] > 1) && (to_op_name.find(kStackActorNameSuffix) == std::string::npos)) {
      SET_FLAG(data_arrow->flag_, kOutputDataFlagBatch);
      (void)actor->batch_output_data_arrows_[to_op_name].emplace_back(data_arrow);
    }
  }
}

void SchedulerHelper::AddDependency(AbstractActor *const actor, const AbstractActor *dependent_actor) {
  MS_EXCEPTION_IF_NULL(actor);
  MS_EXCEPTION_IF_NULL(dependent_actor);
  // For example, ActorA->dependent_actor->actor, the expanded dependent actors of actor are dependent_actor and ActorA.
  (void)actor->dependent_actors_.insert(dependent_actor->GetAID().Name());
  actor->dependent_actors_.insert(dependent_actor->dependent_actors_.begin(), dependent_actor->dependent_actors_.end());
}

bool SchedulerHelper::CheckDependency(const std::vector<AbstractActorPtr> &output_actors) {
  if (output_actors.size() <= 1) {
    return true;
  }

  for (size_t i = 1; i < output_actors.size(); ++i) {
    auto &pre_actor = output_actors[i - 1];
    auto &actor = output_actors[i];
    MS_EXCEPTION_IF_NULL(pre_actor);
    MS_EXCEPTION_IF_NULL(actor);
    // The outputs have no dependencies.
    if ((actor->dependent_actors_.count(pre_actor->GetAID().Name()) == 0) &&
        (pre_actor->dependent_actors_.count(actor->GetAID().Name()) == 0)) {
      return false;
    }
  }

  return true;
}

FusionActorPtr SchedulerHelper::BuildFusionActor(const std::vector<AbstractActorPtr> &actors) {
  if (actors.size() <= 1) {
    MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#The fusion actor size must be greater than 1.";
  }

  std::string fusion_actor_name = std::to_string(++fusion_actor_index_) + kFusionActorNameSuffix;
  auto fusion_actor = std::make_shared<FusionActor>(fusion_actor_name);
  InsertActor(fusion_actor.get());
  for (auto &actor : actors) {
    MS_EXCEPTION_IF_NULL(actor);
    actor->parent_fusion_actor_ = fusion_actor.get();
    MS_LOG(DEBUG) << "Set fusion actor:" << fusion_actor->GetAID() << " to actor:" << actor->GetAID();
    fusion_actor->sub_actors_[actor->GetAID().Name()] = actor;
  }
  return fusion_actor;
}

void SchedulerHelper::AddArrowForFusionActor(FusionActor *fusion_actor) {
  MS_EXCEPTION_IF_NULL(fusion_actor);
  for (auto &actor_iter : fusion_actor->sub_actors_) {
    auto &actor = actor_iter.second;
    MS_EXCEPTION_IF_NULL(actor);

    // Link data arrow of fusion actor by the input data arrow of real actor.
    for (auto &input_data_arrow_aid : actor->input_data_arrow_aids_) {
      auto input_data_arrow = input_data_arrow_aid.second;
      MS_EXCEPTION_IF_NULL(input_data_arrow);
      // Mark the kOutputDataFlagBetweenFusion flag when the input data arrow is the Internal actor in fusion actor.
      if (fusion_actor->sub_actors_.count(input_data_arrow_aid.first.Name()) > 0) {
        SET_FLAG(input_data_arrow->flag_, kOutputDataFlagBetweenFusion);
        continue;
      }

      SET_FLAG(input_data_arrow->flag_, kOutputDataFlagToFusion);
      // The ActorB is in fusion actor and the input ActorA is on the outside of fusion actor, then change
      // 'ActorA->ActorB' to 'ActorA->FusionActor'.
      auto from_actor = FetchActor(input_data_arrow_aid.first.Name());
      MS_EXCEPTION_IF_NULL(from_actor);
      // Record the input index of real actor and fusion actor.
      (void)fusion_actor->real_input_data_.emplace_back(std::make_pair(actor.get(), input_data_arrow->to_input_index_));
      from_actor->data_arrow_to_fusion_actor_indexs_[input_data_arrow] = fusion_actor->input_data_arrow_aids_.size();
      input_data_arrow->to_input_index_ = SizeToInt(fusion_actor->input_data_arrow_aids_.size());

      input_data_arrow->to_op_id_ = fusion_actor->GetAID();
      ++fusion_actor->input_datas_num_;
      (void)fusion_actor->input_data_arrow_aids_.emplace_back(
        std::make_pair(input_data_arrow_aid.first, input_data_arrow));
    }

    // Link control arrow of fusion actor by the input control arrow of real actor.
    for (auto &input_control_arrow_aid : actor->input_control_arrow_aids_) {
      auto input_control_arrow = input_control_arrow_aid.second;
      MS_EXCEPTION_IF_NULL(input_control_arrow);
      // Mark the kOutputDataFlagBetweenFusion flag when the input control arrow is the Internal actor in fusion
      // actor.
      if (fusion_actor->sub_actors_.count(input_control_arrow_aid.first.Name()) > 0) {
        SET_FLAG(input_control_arrow->flag_, kOutputDataFlagBetweenFusion);
        continue;
      }

      SET_FLAG(input_control_arrow->flag_, kOutputDataFlagToFusion);
      // The ActorB is in fusion actor and the input ActorA is on the outside of fusion actor, then change
      // 'ActorA->ActorB' to 'ActorA->FusionActor'.
      (void)fusion_actor->real_input_controls_[input_control_arrow_aid.first.Name()].emplace_back(actor.get());
      input_control_arrow->to_op_id_ = fusion_actor->GetAID();
      ++fusion_actor->input_controls_num_;
      (void)fusion_actor->input_control_arrow_aids_.emplace_back(
        std::make_pair(input_control_arrow_aid.first, input_control_arrow));
    }
  }
}

void SchedulerHelper::AddMemorySign(AbstractActor *const from_actor, AbstractActor *const to_actor) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (runtime::RuntimeConf::GetInstance()->mem_optimize_level() == kOptimizeO0) {
    return;
  }

  if (EnableKbkSubGraphExecute()) {
    return;
  }

  // The link of memory actor no need add the memory sign.
  if (IsMemoryActor(from_actor->type()) || IsMemoryActor(to_actor->type())) {
    return;
  }

  // Add the somas info.
  AddSomasInfo(from_actor);
  AddSomasInfo(to_actor);

  auto from_graph = FetchKernelGraphByActor(from_actor);
  auto to_graph = FetchKernelGraphByActor(to_actor);
  // Add the memory alloc and free sign at the boundary of the graph.
  if ((from_graph != nullptr) && (to_graph != nullptr)) {
    // The same graph no need insert the memory actor.
    if (from_graph->graph_id() == to_graph->graph_id()) {
      return;
    }
    AddMemoryFreeSign(from_actor, to_actor, from_graph);
    AddMemoryAllocSign(from_actor, to_actor, to_graph);
  } else if (from_graph != nullptr) {
    AddMemoryFreeSign(from_actor, to_actor, from_graph);
  } else if (to_graph != nullptr) {
    AddMemoryAllocSign(from_actor, to_actor, to_graph);
  }
}

KernelGraphPtr SchedulerHelper::FetchKernelGraphByActor(AbstractActor *const actor) {
  MS_EXCEPTION_IF_NULL(actor);
  AnfNode *from_kernel = nullptr;
  if (actor->type() == KernelTransformType::kKernelActor ||
      actor->type() == KernelTransformType::kConditionGatherActor ||
      actor->type() == KernelTransformType::kConditionSwitchActor) {
    auto kernel_actor = dynamic_cast<KernelActor *>(actor);
    MS_EXCEPTION_IF_NULL(kernel_actor);
    from_kernel = kernel_actor->kernel().get();
    MS_EXCEPTION_IF_NULL(from_kernel);
  }

  // Only the copy actor from device tensor store need to fetch the kernel graph, because the copy actor is not a
  // boundary of the graph and is equivalent to the kernel actor when inserted the memory actor.
  if ((actor->type() == KernelTransformType::kCopyActor) &&
      (actor->GetAID().Name().find(kCopyActorNameSignFromStore) != std::string::npos)) {
    auto copy_actor = dynamic_cast<CopyActor *>(actor);
    MS_EXCEPTION_IF_NULL(copy_actor);
    from_kernel = copy_actor->from_kernel_;
  }

  if (from_kernel == nullptr) {
    return nullptr;
  }
  auto graph = AnfAlgo::FetchKernelGraph(from_kernel);
  if (graph == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#No associated graph for node: "
                               << from_kernel->fullname_with_scope();
  }

  return graph;
}

void SchedulerHelper::AddMemoryAllocSign(AbstractActor *const from_actor, AbstractActor *const to_actor,
                                         const KernelGraphPtr &to_graph) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_EXCEPTION_IF_NULL(to_graph);
  // Somas is not work for this graph.
  if (to_graph->somas_whole_block_size() == 0) {
    return;
  }

  // Set the memory alloc info.
  to_actor->memory_alloc_insert_position_ = from_actor;
}

void SchedulerHelper::AddMemoryFreeSign(AbstractActor *const from_actor, AbstractActor *const to_actor,
                                        const KernelGraphPtr &from_graph) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_EXCEPTION_IF_NULL(from_graph);
  // Somas is not work for this graph.
  if (from_graph->somas_whole_block_size() == 0) {
    return;
  }

  // Set the memory free info.
  from_actor->memory_free_insert_position_ = to_actor;
}

void SchedulerHelper::AddSomasInfo(AbstractActor *const actor) {
  MS_EXCEPTION_IF_NULL(actor);
  // Only the kernel actor supports somas.
  if (actor->type() != KernelTransformType::kKernelActor &&
      actor->type() != KernelTransformType::kConditionGatherActor &&
      actor->type() != KernelTransformType::kConditionSwitchActor) {
    return;
  }
  auto kernel_actor = dynamic_cast<KernelActor *>(actor);
  MS_EXCEPTION_IF_NULL(kernel_actor);
  if (kernel_actor->somas_info_ != nullptr) {
    return;
  }

  MS_EXCEPTION_IF_NULL(kernel_actor->kernel());
  auto graph = AnfAlgo::FetchKernelGraph(kernel_actor->kernel().get());
  if (graph == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#No associated graph for node: "
                               << kernel_actor->kernel()->fullname_with_scope();
  }
  // Somas is not work for this graph.
  if (graph->somas_whole_block_size() == 0) {
    return;
  }

  // Set the somas info.
  auto somas_info = graph->MutableSomasInfo();
  MS_EXCEPTION_IF_NULL(somas_info);
  somas_info->graph_id_ = graph->graph_id();
  kernel_actor->somas_info_ = somas_info;
}

void SchedulerHelper::AddSomasInfoV2(KernelRunner *const actor) {
  MS_EXCEPTION_IF_NULL(actor);
  // Only the kernel actor supports somas.
  if (actor->type() != KernelTransformType::kKernelActor &&
      actor->type() != KernelTransformType::kConditionGatherActor &&
      actor->type() != KernelTransformType::kConditionSwitchActor) {
    return;
  }
  auto kernel_actor = actor;
  MS_EXCEPTION_IF_NULL(kernel_actor);
  if (kernel_actor->somas_info_ != nullptr) {
    return;
  }

  MS_EXCEPTION_IF_NULL(kernel_actor->kernel());
  auto graph = AnfAlgo::FetchKernelGraph(kernel_actor->kernel().get());
  if (graph == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#No associated graph for node: "
                               << kernel_actor->kernel()->fullname_with_scope();
  }
  // Somas is not work for this graph.
  if (graph->somas_whole_block_size() == 0) {
    return;
  }

  // Set the somas info.
  auto somas_info = graph->MutableSomasInfo();
  MS_EXCEPTION_IF_NULL(somas_info);
  somas_info->graph_id_ = graph->graph_id();
  kernel_actor->somas_info_ = somas_info;
}

void SchedulerHelper::AddSomasInfoForGraphOutput(AbstractActor *const output_actor, size_t output_index,
                                                 size_t graph_id) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (runtime::RuntimeConf::GetInstance()->mem_optimize_level() == kOptimizeO0) {
    return;
  }
  if ((output_actor == nullptr) || (output_actor->type() != KernelTransformType::kKernelActor &&
                                    output_actor->type() != KernelTransformType::kConditionSwitchActor &&
                                    output_actor->type() != KernelTransformType::kConditionGatherActor)) {
    return;
  }

  auto kernel_actor = dynamic_cast<KernelActor *>(output_actor);
  MS_EXCEPTION_IF_NULL(kernel_actor);
  const auto &kernel = kernel_actor->kernel();
  MS_EXCEPTION_IF_NULL(kernel);
  auto kernel_info = dynamic_cast<KernelInfo *>(kernel->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  const auto &somas_outputs = kernel_info->somas_output_result();
  auto is_somas = kernel_info->IsTensorEnableSomas(somas_outputs, output_index);
  MS_LOG(INFO) << "The graph " << graph_id << " output node:" << kernel->fullname_with_scope()
               << " with index: " << output_index << " somas enable or not: " << is_somas
               << ", somas offset: " << kernel_info->GetTensorSomasOffset(somas_outputs, output_index)
               << ", aligned size: " << kernel_info->GetTensorSomasAlignedSize(somas_outputs, output_index);
  if (is_somas) {
    kernel_actor->somas_graph_output_indexes_.insert(output_index);
  }
}

void SchedulerHelper::AddSomasInfoForGraphOutputV2(KernelRunner *const output_actor, size_t output_index,
                                                   size_t graph_id) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (runtime::RuntimeConf::GetInstance()->mem_optimize_level() == kOptimizeO0) {
    return;
  }
  if ((output_actor == nullptr) || (output_actor->type() != KernelTransformType::kKernelActor &&
                                    output_actor->type() != KernelTransformType::kConditionSwitchActor &&
                                    output_actor->type() != KernelTransformType::kConditionGatherActor)) {
    return;
  }

  auto kernel_actor = output_actor;
  MS_EXCEPTION_IF_NULL(kernel_actor);
  const auto &kernel = kernel_actor->kernel();
  MS_EXCEPTION_IF_NULL(kernel);
  auto kernel_info = dynamic_cast<KernelInfo *>(kernel->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  const auto &somas_outputs = kernel_info->somas_output_result();
  auto is_somas = kernel_info->IsTensorEnableSomas(somas_outputs, output_index);
  MS_LOG(INFO) << "The graph " << graph_id << " output node:" << kernel->fullname_with_scope()
               << " with index: " << output_index << " somas enable or not: " << is_somas
               << ", somas offset: " << kernel_info->GetTensorSomasOffset(somas_outputs, output_index)
               << ", aligned size: " << kernel_info->GetTensorSomasAlignedSize(somas_outputs, output_index);
  if (is_somas) {
    kernel_actor->somas_graph_output_indexes_.insert(output_index);
  }
}

namespace {
void CheckKernelActorValid(const std::vector<KernelActorPtr> &kernel_actors) {
  for (const auto &kernel_actor : kernel_actors) {
    MS_EXCEPTION_IF_NULL(kernel_actor);
    std::string exit_actor_name = "";

    for (const auto &arrow : kernel_actor->output_data_arrows()) {
      MS_EXCEPTION_IF_NULL(arrow);
      if (arrow->to_op_id_.Name().find(kExitActorNameSuffix) == std::string::npos) {
        continue;
      }
      if (exit_actor_name == "") {
        exit_actor_name = arrow->to_op_id_.Name();
        continue;
      }
      if (exit_actor_name != arrow->to_op_id_.Name()) {
        MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Kernel actor:" << kernel_actor->GetAID()
                                   << " link to two exit actor:" << exit_actor_name
                                   << " and:" << arrow->to_op_id_.Name();
      }
    }
  }
}

bool CheckExitActorInvalid(const ExitActorPtr &exit_actor) {
  MS_EXCEPTION_IF_NULL(exit_actor);

  return exit_actor->output_data_arrows().empty() && exit_actor->output_partial_arrows().empty() &&
         exit_actor->output_control_arrows().empty() && exit_actor->output_branch_control_arrows().empty() &&
         exit_actor->output_branch_data_arrows().empty() && exit_actor->output_branch_partial_arrows().empty() &&
         !exit_actor->input_data_arrow_aids().empty();
}

// Convert the control actors vector by the control actor set.
std::vector<ControlActorPtr> CollectControlActors(const ControlActorSetPtr &control_actor_set) {
  MS_EXCEPTION_IF_NULL(control_actor_set);
  std::vector<ControlActorPtr> actors;

  for (auto &switch_actor : control_actor_set->switch_actors_) {
    MS_EXCEPTION_IF_NULL(switch_actor);
    (void)actors.emplace_back(static_cast<ControlActorPtr>(switch_actor));
  }
  for (auto &gather_actor : control_actor_set->gather_actors_) {
    MS_EXCEPTION_IF_NULL(gather_actor);
    (void)actors.emplace_back(static_cast<ControlActorPtr>(gather_actor));
  }
  for (auto &entrance_actor : control_actor_set->entrance_actors_) {
    MS_EXCEPTION_IF_NULL(entrance_actor);
    (void)actors.emplace_back(static_cast<ControlActorPtr>(entrance_actor));
  }
  for (auto &exit_actor : control_actor_set->exit_actors_) {
    MS_EXCEPTION_IF_NULL(exit_actor);
    (void)actors.emplace_back(static_cast<ControlActorPtr>(exit_actor));
  }
  for (auto &stack_actor : control_actor_set->stack_actors_) {
    MS_EXCEPTION_IF_NULL(stack_actor);
    (void)actors.emplace_back(static_cast<ControlActorPtr>(stack_actor));
  }

  return actors;
}

void CheckControlActorValid(const ActorSet *actor_set) {
  MS_EXCEPTION_IF_NULL(actor_set);
  if (actor_set->control_actors_ == nullptr) {
    return;
  }

  CheckKernelActorValid(actor_set->kernel_actors_);

  auto control_actors = CollectControlActors(actor_set->control_actors_);
  for (const auto &control_actor : control_actors) {
    MS_EXCEPTION_IF_NULL(control_actor);
    for (auto &ref_node_formal_parameter_kernel_tensor : control_actor->ref_node_formal_parameter_kernel_tensors()) {
      auto &kernel_tensors = ref_node_formal_parameter_kernel_tensor.second;
      for (auto iter = kernel_tensors.begin(); iter != kernel_tensors.end(); ++iter) {
        MS_EXCEPTION_IF_NULL((*kernel_tensors.begin())->device_address());
        MS_EXCEPTION_IF_NULL((*iter)->device_address());
        if (((*kernel_tensors.begin())->format() != (*iter)->format()) ||
            ((*kernel_tensors.begin())->dtype_id() != (*iter)->dtype_id())) {
          MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#" << control_actor->GetAID().Name()
                                     << " does not support the ref node formal parameters with different format.";
        }
      }
    }

    for (auto &ref_formal_parameter_kernel_tensor : control_actor->ref_formal_parameter_kernel_tensors()) {
      auto &kernel_tensors = ref_formal_parameter_kernel_tensor.second;
      for (auto iter = kernel_tensors.begin(); iter != kernel_tensors.end(); ++iter) {
        MS_EXCEPTION_IF_NULL((*kernel_tensors.begin())->device_address());
        MS_EXCEPTION_IF_NULL((*iter)->device_address());
        if ((*kernel_tensors.begin())->dtype_id() != (*iter)->dtype_id()) {
          MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#" << control_actor->GetAID().Name()
                                     << " does not support the ref formal parameters with different type.";
        }
      }
    }
  }

  for (const auto &exit_actor : actor_set->control_actors_->exit_actors_) {
    MS_EXCEPTION_IF_NULL(exit_actor);
    if (CheckExitActorInvalid(exit_actor)) {
      MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Invalid exit actor:" << exit_actor->GetAID();
    }
  }

  // Since some control arrows of stack actors need to be counted according to aid, the input control arrow cannot
  // be repeated, otherwise the count will be inaccurate. But there are exceptions, if the control arrow does not
  // need to be counted according to aid, it can be repeated.
  for (const auto &stack_actor : actor_set->control_actors_->stack_actors_) {
    MS_EXCEPTION_IF_NULL(stack_actor);
    const auto &input_control_aids = stack_actor->input_control_arrow_aids();
    std::set<AID> aid_set;
    (void)std::for_each(input_control_aids.begin(), input_control_aids.end(),
                        [&aid_set](const auto &input_control_aid) { (void)aid_set.emplace(input_control_aid.first); });
    if (aid_set.size() != input_control_aids.size()) {
      MS_LOG(WARNING) << "Stack actor:" << stack_actor->GetAID() << " has duplicate control arrows.";
    }
  }
}
}  // namespace

void SchedulerHelper::CheckActorValid(const ActorSet *actor_set) {
  MS_EXCEPTION_IF_NULL(actor_set);
  auto actors = SchedulerHelper::CollectActors(actor_set);
  for (auto &actor : actors) {
    MS_EXCEPTION_IF_NULL(actor);
    if (actor->type_ >= KernelTransformType::kSwitchActor) {
      continue;
    }

    if ((actor->input_datas_num_ != actor->input_data_arrow_aids_.size()) ||
        (actor->input_controls_num_ != actor->input_control_arrow_aids_.size())) {
      MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#The input num of " << actor->GetAID().Name()
                                 << " is wrong, expect data num: " << actor->input_datas_num_
                                 << ", actual data num: " << actor->input_data_arrow_aids_.size()
                                 << ", expect control num: " << actor->input_controls_num_
                                 << ", actual control num: " << actor->input_control_arrow_aids_.size();
    }

    if ((actor->type_ != KernelTransformType::kOutputActor) && (actor->output_data_arrows_.size() == 0) &&
        (actor->output_control_arrows_.size() == 0)) {
      MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#" << actor->GetAID().Name() << " has no user.";
    }
    if ((actor->type_ != KernelTransformType::kDataPrepareActor) &&
        (actor->input_datas_num_ == 0 && actor->parameter_indexs_.size() == 0) && (actor->input_controls_num_ == 0)) {
      MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#" << actor->GetAID().Name() << " has no source.";
    }

    // Check the input of kernel actors and copy actors.
    if ((actor->type_ == KernelTransformType::kKernelActor) || (actor->type_ == KernelTransformType::kCopyActor)) {
      size_t expect_input_num = 1;
      if (actor->type_ == KernelTransformType::kKernelActor) {
        auto kernel_actor = dynamic_cast<KernelActor *>(actor.get());
        MS_EXCEPTION_IF_NULL(kernel_actor);
        auto &kernel = kernel_actor->kernel();
        MS_EXCEPTION_IF_NULL(kernel);
        auto kernel_info = dynamic_cast<device::KernelInfo *>(kernel->kernel_info());
        MS_EXCEPTION_IF_NULL(kernel_info);
        auto build_info = kernel_info->select_kernel_build_info();
        MS_EXCEPTION_IF_NULL(build_info);
        expect_input_num = build_info->GetInputNumWithoutMonad();
      }
      auto input_data_num = actor->input_datas_num_;
      auto device_tensor_store_num = actor->device_tensor_store_keys_.size();
      auto parameter_index_num = actor->parameter_indexs_.size();
      if (input_data_num + device_tensor_store_num + parameter_index_num != expect_input_num) {
        MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#The input building of " << actor->GetAID().Name()
                                   << " is wrong, input data num: " << input_data_num
                                   << ", device tensor store num: " << device_tensor_store_num
                                   << ", parameter index num: " << parameter_index_num
                                   << ", total input num: " << expect_input_num;
      }
    }
  }

  // Check the output actor.
  auto output_actor = actor_set->output_actor_;
  MS_EXCEPTION_IF_NULL(output_actor);
  if (output_actor->input_datas_num_ + output_actor->device_tensor_store_keys_.size() +
        output_actor->parameter_indexs_.size() !=
      output_actor->outputs_num()) {
    MS_LOG(INTERNAL_EXCEPTION)
      << "#dmsg#Runtime error info:#dmsg#The outputs num of output actor is wrong, the total outputs num: "
      << output_actor->outputs_num() << ", the input data arrows num: " << output_actor->input_datas_num_
      << ", the device tensor store num: " << output_actor->device_tensor_store_keys_.size()
      << ", parameter indexes size: " << output_actor->parameter_indexs_.size();
  }

  CheckControlActorValid(actor_set);
}

void SchedulerHelper::DumpActorSet(const ActorSet *actor_set, std::ofstream &ofs) {
  MS_EXCEPTION_IF_NULL(actor_set);
  DumpParameterStore(ofs);
  DumpContinuousMemoryNodes(actor_set, ofs);
  DumpDataPrepareActor(actor_set->data_prepare_actor_, ofs);
  DumpDSActors(actor_set->data_source_actors_, ofs);
  DumpKernelActors(actor_set->kernel_actors_, ofs);
  DumpSuperKernelActors(actor_set->super_kernel_actors_, ofs);
  DumpAnyTypeKernelActors(actor_set->any_type_kernel_actors_, ofs);
  // The on input kernel actors are taken over by control actor in the control flow scene.
  if (actor_set->control_actors_ == nullptr) {
    DumpNoInputKernelActors(actor_set->no_input_kernel_actors_, ofs);
  }
  DumpMemoryActors(actor_set->memory_actors_, ofs);
  DumpCopyActors(actor_set->copy_actors_, ofs);
  DumpLoopCountActor(actor_set->loop_count_actor_, ofs);
  DumpOutputActor(actor_set->output_actor_, ofs);
  DumpFusionActors(actor_set->fusion_actors_, ofs);
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

void SchedulerHelper::ProcessStreamSendRecvEventPair(
  mindspore::HashMap<uint32_t, std::pair<KernelActorPtr, KernelActorPtr>> *send_recv_nodes, const CNodePtr &kernel,
  const KernelActorPtr &kernel_actor, bool is_send_node) {
  auto primitive = common::AnfAlgo::GetCNodePrimitive(kernel);
  MS_EXCEPTION_IF_NULL(primitive);
  auto record_event_stream_pair_attr = primitive->GetAttr(kAttrRecordWaitEventStreamPairId);
  if (record_event_stream_pair_attr != nullptr) {
    auto event_pair_id = GetValue<uint32_t>(record_event_stream_pair_attr);
    MS_LOG(DEBUG) << "Process event pair id : " << event_pair_id << ".";
    auto &send_recv_actor = (*send_recv_nodes)[event_pair_id];
    if (is_send_node) {
      MS_EXCEPTION_IF_CHECK_FAIL(send_recv_actor.first == nullptr, "Stream send pair id is already set.");
      send_recv_actor.first = kernel_actor;
    } else {
      MS_EXCEPTION_IF_CHECK_FAIL(send_recv_actor.second == nullptr, "Stream recv pair id is already set.");
      send_recv_actor.second = kernel_actor;
    }
  } else {
    MS_LOG(INFO) << "Stream send/recv kernel : " << kernel->DebugString() << " has no event stream pair id.";
  }
}

void SchedulerHelper::ProcessStreamSendRecvEventPairV2(
  mindspore::HashMap<uint32_t, std::pair<KernelRunnerPtr, KernelRunnerPtr>> *send_recv_nodes, const CNodePtr &kernel,
  const KernelRunnerPtr &kernel_actor, bool is_send_node) {
  auto primitive = common::AnfAlgo::GetCNodePrimitive(kernel);
  MS_EXCEPTION_IF_NULL(primitive);
  auto record_event_stream_pair_attr = primitive->GetAttr(kAttrRecordWaitEventStreamPairId);
  if (record_event_stream_pair_attr != nullptr) {
    auto event_pair_id = GetValue<uint32_t>(record_event_stream_pair_attr);
    MS_LOG(DEBUG) << "Process event pair id : " << event_pair_id << ".";
    auto &send_recv_actor = (*send_recv_nodes)[event_pair_id];
    if (is_send_node) {
      MS_EXCEPTION_IF_CHECK_FAIL(send_recv_actor.first == nullptr, "Stream send pair id is already set.");
      send_recv_actor.first = kernel_actor;
    } else {
      MS_EXCEPTION_IF_CHECK_FAIL(send_recv_actor.second == nullptr, "Stream recv pair id is already set.");
      send_recv_actor.second = kernel_actor;
    }
  } else {
    MS_LOG(INFO) << "Stream send/recv kernel : " << kernel->DebugString() << " has no event stream pair id.";
  }
}

KernelTensorPtr SchedulerHelper::CloneKernelTensorWithDeviceInfo(const KernelTensorPtr &kernel_tensor,
                                                                 const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  auto device_address = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_address);
  auto new_device_address =
    std::make_shared<device::DeviceAddress>(device_address->device_pointer()->ptr(), device_address->size(),
                                            device_context->GetDeviceType(), device_address->stream_id());
  auto new_kernel_tensor = kernel_tensor->CloneKernelTensor();
  new_kernel_tensor->set_user_data(kernel_tensor->user_data());
  new_kernel_tensor->set_need_sync_user_data(kernel_tensor->need_sync_user_data());
  new_kernel_tensor->set_device_address(new_device_address);
  new_kernel_tensor->set_tensor_storage_info(nullptr);
  return new_kernel_tensor;
}
}  // namespace runtime
}  // namespace mindspore
