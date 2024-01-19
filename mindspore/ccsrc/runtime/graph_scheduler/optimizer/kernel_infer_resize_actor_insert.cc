/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "runtime/graph_scheduler/optimizer/kernel_infer_resize_actor_insert.h"
#include <set>
#include <vector>
#include <utility>
#include "runtime/graph_scheduler/scheduler_helper.h"

namespace mindspore {
namespace runtime {
namespace {
// Create KernelInferActor and insert to actor set.
KernelInferActorPtr CreateKernelInferActor(ActorSet *const actor_set, const KernelActor *kernel_actor) {
  MS_EXCEPTION_IF_NULL(actor_set);
  MS_EXCEPTION_IF_NULL(kernel_actor);

  const auto name = FetchActorName(KernelTransformType::kKernelInferActor, actor_set->name_, kernel_actor->kernel());
  if (kernel_actor->device_contexts().empty()) {
    MS_LOG(EXCEPTION) << "The device contexts of kernel actor[ " << kernel_actor->GetAID().Name() << " ] is empty.";
  }

  auto kernel_infer_actor = std::make_shared<KernelInferActor>(
    name, kernel_actor->kernel(), kernel_actor->device_contexts()[0], kernel_actor->memory_manager_aid());
  InsertActor(kernel_infer_actor.get());
  (void)actor_set->kernel_infer_actors_.emplace_back(kernel_infer_actor);
  return kernel_infer_actor;
}

// Create KernelResizeActor and insert to actor set.
KernelResizeActorPtr CreateKernelResizeActor(ActorSet *const actor_set, const KernelActor *kernel_actor) {
  MS_EXCEPTION_IF_NULL(actor_set);
  MS_EXCEPTION_IF_NULL(kernel_actor);

  const auto name = FetchActorName(KernelTransformType::kKernelResizeActor, actor_set->name_, kernel_actor->kernel());
  if (kernel_actor->device_contexts().empty()) {
    MS_LOG(EXCEPTION) << "The device contexts of kernel actor[ " << kernel_actor->GetAID().Name() << " ] is empty.";
  }

  auto kernel_resize_actor = std::make_shared<KernelResizeActor>(
    name, kernel_actor->kernel(), kernel_actor->device_contexts()[0], kernel_actor->memory_manager_aid());
  InsertActor(kernel_resize_actor.get());
  (void)actor_set->kernel_resize_actors_.emplace_back(kernel_resize_actor);
  return kernel_resize_actor;
}

// Fetch the from actor to insert data arrow between it and current kernel infer/resize actor.
AbstractActor *FetchRealFromActor(const DataArrow *data_arrow, const ActorSet *actor_set, const CNodePtr &kernel,
                                  KernelActor *from_kernel_actor) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(data_arrow);
  MS_EXCEPTION_IF_NULL(actor_set);
  MS_EXCEPTION_IF_NULL(from_kernel_actor);

  std::set<int64_t> depend_list = abstract::GetValueDependArgIndices(kernel);
  AnfNodePtr from_kernel = from_kernel_actor->kernel();
  MS_EXCEPTION_IF_NULL(from_kernel);
  auto from_kernel_mod = AnfAlgo::GetKernelMod(from_kernel);
  MS_EXCEPTION_IF_NULL(from_kernel_mod);

  // The following situations require inserting data arrow between from kernel actor(not from kernel infer actor) and
  // current kernel infer/resize actor:
  // 1. Value Depend case: this input index is is value depend list of current kernel.
  // 2. Compute Depend case: the output shape of from kernel depends on computation result.
  // 3. Not Dynamic Shape case: the from kernel is not dynamic shape which has no kernel infer/resize actor.
  // 4. Special Operator: 'GetNext' has no kernel infer/resize actor.
  if ((depend_list.find(data_arrow->to_input_index_) != depend_list.end()) ||
      from_kernel_mod->IsNeedUpdateOutputShapeAndSize() || !from_kernel_actor->is_dynamic_shape() ||
      common::AnfAlgo::IsGetNextNode(from_kernel)) {
    return from_kernel_actor;
  } else {
    // Other case require inserting data arrow between from kernel infer actor and current kernel infer/resize actor.
    return FetchActor(KernelTransformType::kKernelInferActor, actor_set->name_, from_kernel);
  }
}

// Add data arrows for KernelInferActor, KernelResizeActor, KernelActor.
void AddDataArrows(const ActorSet *actor_set, const KernelActor *kernel_actor, KernelInferActor *kernel_infer_actor,
                   KernelResizeActor *kernel_resize_actor) {
  MS_EXCEPTION_IF_NULL(kernel_actor);
  MS_EXCEPTION_IF_NULL(kernel_infer_actor);
  MS_EXCEPTION_IF_NULL(kernel_resize_actor);
  const CNodePtr &kernel = kernel_actor->kernel();
  MS_EXCEPTION_IF_NULL(kernel);
  const std::vector<std::pair<AID, DataArrow *>> &input_data_arrow_aids = kernel_actor->input_data_arrow_aids();

  // Clone all input data arrows of KernelActor to KernelInferActor and KernelResizeActor.
  for (const auto &input_pair : input_data_arrow_aids) {
    const AID &from_actor_aid = input_pair.first;
    const DataArrow *data_arrow = input_pair.second;
    MS_EXCEPTION_IF_NULL(data_arrow);
    AbstractActor *from_actor = FetchActor(from_actor_aid.Name());
    MS_EXCEPTION_IF_NULL(from_actor);

    if (from_actor->type() == KernelTransformType::kExitActor ||
        from_actor->type() == KernelTransformType::kAnyTypeKernelActor) {
      MS_LOG(EXCEPTION) << "KernelActor can not receive data from ExitActor or AnyTypeKernelActor.";
    }

    if (from_actor->type() == KernelTransformType::kKernelActor) {
      auto from_kernel_actor = dynamic_cast<KernelActor *>(from_actor);
      MS_EXCEPTION_IF_NULL(from_kernel_actor);
      auto real_from_actor = FetchRealFromActor(data_arrow, actor_set, kernel, from_kernel_actor);
      MS_EXCEPTION_IF_NULL(real_from_actor);

      AnfNodePtr from_kernel = from_kernel_actor->kernel();
      MS_EXCEPTION_IF_NULL(from_kernel);
      SchedulerHelper::AddDataArrow(real_from_actor, kernel_infer_actor, data_arrow->from_output_index_,
                                    data_arrow->to_input_index_, from_kernel);
      SchedulerHelper::AddDataArrow(real_from_actor, kernel_resize_actor, data_arrow->from_output_index_,
                                    data_arrow->to_input_index_, from_kernel);
    } else if (from_actor->type() == KernelTransformType::kHostDataSourceActor) {
      const AnfNodePtr &from_node = common::AnfAlgo::GetInputNode(kernel, data_arrow->to_input_index_);
      const auto &real_from_node = common::AnfAlgo::VisitKernelWithReturnType(from_node, 0, false).first;
      MS_EXCEPTION_IF_NULL(real_from_node);
      if (!real_from_node->isa<Parameter>()) {
        MS_LOG(EXCEPTION) << "The input[" << data_arrow->to_input_index_
                          << "] of kernel: " << kernel->fullname_with_scope()
                          << " expect a Parameter, but got: " << real_from_node->DebugString();
      }
      SchedulerHelper::AddDataArrow(from_actor, kernel_infer_actor, data_arrow->from_output_index_,
                                    data_arrow->to_input_index_, real_from_node);
      SchedulerHelper::AddDataArrow(from_actor, kernel_resize_actor, data_arrow->from_output_index_,
                                    data_arrow->to_input_index_, real_from_node);
    } else if (from_actor->type() == KernelTransformType::kDeviceDataSourceActor) {
      // Note: Remove this branch in future after change DevieDataSourceActor to a KernelActor.
      auto device_data_source_actor = dynamic_cast<DeviceQueueDataSourceActor *>(from_actor);
      MS_EXCEPTION_IF_NULL(device_data_source_actor);
      SchedulerHelper::AddDataArrow(from_actor, kernel_infer_actor, data_arrow->from_output_index_,
                                    data_arrow->to_input_index_, device_data_source_actor->data_kernel());
      SchedulerHelper::AddDataArrow(from_actor, kernel_resize_actor, data_arrow->from_output_index_,
                                    data_arrow->to_input_index_, device_data_source_actor->data_kernel());
    } else {
      SchedulerHelper::AddDataArrow(from_actor, kernel_infer_actor, data_arrow->from_output_index_,
                                    data_arrow->to_input_index_);
      SchedulerHelper::AddDataArrow(from_actor, kernel_resize_actor, data_arrow->from_output_index_,
                                    data_arrow->to_input_index_);
    }
  }
}

// Add control arrows for KernelInferActor, KernelResizeActor, KernelActor.
void AddControlArrows(KernelActor *kernel_actor, KernelInferActor *kernel_infer_actor,
                      KernelResizeActor *kernel_resize_actor) {
  MS_EXCEPTION_IF_NULL(kernel_actor);
  MS_EXCEPTION_IF_NULL(kernel_infer_actor);
  MS_EXCEPTION_IF_NULL(kernel_resize_actor);

  const CNodePtr &kernel = kernel_actor->kernel();
  MS_EXCEPTION_IF_NULL(kernel);
  const std::vector<std::pair<AID, DataArrow *>> &input_data_arrow_aids = kernel_actor->input_data_arrow_aids();
  const std::vector<std::pair<AID, ControlArrow *>> &input_control_arrow_aids =
    kernel_actor->input_control_arrow_aids();

  if (input_data_arrow_aids.empty() && input_control_arrow_aids.empty()) {
    MS_LOG(EXCEPTION) << "No input node should has input control arrow.";
  }

  // The following situations require inserting control arrow between from actor and kernel infer actor:
  // 1. no input data arrows
  // 2. this kernel has value depend input and input control arrow, may exists umonad in this case.
  bool need_add_control_arrow_for_infer_actor =
    input_data_arrow_aids.empty() ||
    (!input_control_arrow_aids.empty() && !abstract::GetValueDependArgIndices(kernel).empty());

  // Clone all control arrow of this kernel to kernel infer actor
  if (need_add_control_arrow_for_infer_actor) {
    for (const auto &input_pair : input_control_arrow_aids) {
      const AID &from_actor_aid = input_pair.first;
      AbstractActor *from_actor = FetchActor(from_actor_aid.Name());
      MS_EXCEPTION_IF_NULL(from_actor);
      SchedulerHelper::AddControlArrow(from_actor, kernel_infer_actor);
    }
  }

  // For a dynamic kernel, the control always like this: KernelInferActor --> KernelResizeActor --> KernelActor.
  SchedulerHelper::AddControlArrow(kernel_infer_actor, kernel_resize_actor);
  SchedulerHelper::AddControlArrow(kernel_resize_actor, kernel_actor);
}
}  // namespace

bool KernelInferResizeActorInsert::MatchPattern(const AbstractActor *actor) const {
  MS_EXCEPTION_IF_NULL(actor);
  if (actor->type() == KernelTransformType::kKernelActor) {
    auto kernel_actor = dynamic_cast<const KernelActor *>(actor);
    MS_EXCEPTION_IF_NULL(kernel_actor);
    const auto &kernel = kernel_actor->kernel();
    MS_EXCEPTION_IF_NULL(kernel);
    return kernel_actor->is_dynamic_shape() && !common::AnfAlgo::IsGetNextNode(kernel);
  }
  return false;
}

void KernelInferResizeActorInsert::Process(ActorSet *const actor_set, AbstractActor *const actor) {
  MS_EXCEPTION_IF_NULL(actor_set);
  MS_EXCEPTION_IF_NULL(actor);
  auto kernel_actor = dynamic_cast<KernelActor *>(actor);
  MS_EXCEPTION_IF_NULL(kernel_actor);

  const auto &kernel_infer_actor = CreateKernelInferActor(actor_set, kernel_actor);
  const auto &kernel_resize_actor = CreateKernelResizeActor(actor_set, kernel_actor);
  MS_EXCEPTION_IF_NULL(kernel_infer_actor);
  MS_EXCEPTION_IF_NULL(kernel_resize_actor);
  kernel_infer_actor->set_device_tensor_store_keys(kernel_actor->device_tensor_store_keys());
  kernel_resize_actor->set_device_tensor_store_keys(kernel_actor->device_tensor_store_keys());

  AddDataArrows(actor_set, kernel_actor, kernel_infer_actor.get(), kernel_resize_actor.get());
  AddControlArrows(kernel_actor, kernel_infer_actor.get(), kernel_resize_actor.get());

  kernel_actor->set_enable_async_infer(true);
}
}  // namespace runtime
}  // namespace mindspore
