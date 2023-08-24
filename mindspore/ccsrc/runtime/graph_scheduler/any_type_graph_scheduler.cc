/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include <memory>
#include "runtime/graph_scheduler/any_type_graph_scheduler.h"
#include "runtime/graph_scheduler/graph_scheduler.h"

namespace mindspore {
namespace runtime {
std::vector<AnyTypeKernelActorPtr> AnyTypeGraphScheduler::Build(const GraphCompilerInfo &graph_compiler_info,
                                                                const AID &memory_manager_aid, const AID *debug_id) {
  std::vector<AnyTypeKernelActorPtr> any_type_kernel_actors;
  for (size_t i = 0; i < graph_compiler_info.graphs_.size(); ++i) {
    const auto &graph = graph_compiler_info.graphs_[i];
    const auto &device_context = graph_compiler_info.device_contexts_[i];
    MS_EXCEPTION_IF_NULL(graph);
    if (!graph->is_any_type_input()) {
      continue;
    }
    if (graph->execution_order().empty()) {
      MS_LOG(INFO) << "The graph " << graph->graph_id() << " is an empty graph and skips building.";
      continue;
    }

    auto actor_name = graph->ToString() + kAnyTypeKernelActorNameSuffix;
    auto any_type_kernel_actor =
      std::make_shared<AnyTypeKernelActor>(actor_name, graph, device_context, memory_manager_aid, debug_id, nullptr);
    any_type_kernel_actor->compile_func_ = graph_compiler_info.compile_func_;
    any_type_kernel_actor->transform_func_ = [this, &graph_compiler_info](const KernelGraphPtr &model_graph,
                                                                          const KernelGraphPtr &real_graph,
                                                                          const DeviceContext *device_context) {
      return Transform(model_graph, real_graph, device_context, graph_compiler_info.origin_parameters_order_);
    };
    any_type_kernel_actor->schedule_func_ = [this](const std::vector<AbstractActorPtr> &actors) {
      auto actor_manager = ActorMgr::GetActorMgrRef();
      MS_EXCEPTION_IF_NULL(actor_manager);
      for (auto actor : actors) {
        MS_EXCEPTION_IF_NULL(actor);
        // The sub actors in the fusion actor do not participate in message interaction.
        if (actor->parent_fusion_actor_ == nullptr) {
          (void)actor_manager->Spawn(actor);
        } else {
          actor->Init();
        }
      }
    };
    MS_EXCEPTION_IF_NULL(any_type_kernel_actor);
    InsertActor(any_type_kernel_actor.get());
    (void)any_type_kernel_actors.emplace_back(any_type_kernel_actor);
  }
  return any_type_kernel_actors;
}

void AnyTypeGraphScheduler::TransArrowInDSActorToAnyTypeKernelActor(AnyTypeKernelActor *const any_type_kernel_actor,
                                                                    const DataSourceActorPtr &data_source_actor,
                                                                    const KernelGraphPtr &model_graph,
                                                                    const KernelGraphPtr &real_graph) {}

void AnyTypeGraphScheduler::TransArrowInActorSetToAnyTypeKernelActor(const ActorSet *const actor_set,
                                                                     const KernelGraphPtr &model_graph,
                                                                     const KernelGraphPtr &real_graph) {}

std::vector<AbstractActorPtr> AnyTypeGraphScheduler::Transform(const KernelGraphPtr &model_graph,
                                                               const KernelGraphPtr &real_graph,
                                                               const DeviceContext *device_context,
                                                               const std::vector<AnfNodePtr> &front_parameters) {
  return {};
}
}  // namespace runtime
}  // namespace mindspore
