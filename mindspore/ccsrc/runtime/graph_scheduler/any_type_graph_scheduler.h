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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ANY_TYPE_GRAPH_SCHEDULER_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ANY_TYPE_GRAPH_SCHEDULER_H_

#include <vector>
#include "utils/ms_utils.h"
#include "runtime/graph_scheduler/actor/actor_set.h"

namespace mindspore {
namespace runtime {
class AnyTypeGraphScheduler {
 public:
  AnyTypeGraphScheduler() = default;
  ~AnyTypeGraphScheduler() = default;
  DISABLE_COPY_AND_ASSIGN(AnyTypeGraphScheduler);

  // Transform the control nodes to control actors.
  std::vector<AnyTypeKernelActorPtr> Build(const GraphCompilerInfo &graph_compiler_info, const AID &memory_manager_aid,
                                           const AID *debug_id);

  // Transform any type input graph to actor DAG, Generate actor set according to real graph, eliminate data source
  // actor, loop count actor, output actor, and link arrows to any type kernel actor of model graph.
  std::vector<AbstractActorPtr> Transform(const KernelGraphPtr &model_graph, const KernelGraphPtr &real_graph,
                                          const DeviceContext *device_context,
                                          const std::vector<AnfNodePtr> &front_parameters);

 private:
  void TransArrowInActorSetToAnyTypeKernelActor(const ActorSet *const actor_set, const KernelGraphPtr &model_graph,
                                                const KernelGraphPtr &real_graph);

  void TransArrowInDataSourceActorToAnyTypeKernelActor(AnyTypeKernelActor *const any_type_kernel_actor,
                                                       const DataSourceActorPtr &data_source_actor,
                                                       const KernelGraphPtr &model_graph,
                                                       const KernelGraphPtr &real_graph);

  void TransArrowInDataPrepareActorToAnyTypeKernelActor(AnyTypeKernelActor *const any_type_kernel_actor,
                                                        const DataPrepareActorPtr &data_prepare_actor);

  void TransArrowInLoopCountActorToAnyTypeKernelActor(AnyTypeKernelActor *const any_type_kernel_actor,
                                                      const LoopCountActorPtr &loop_count_actor);

  void TransArrowInOutputActorToAnyTypeKernelActor(AnyTypeKernelActor *const any_type_kernel_actor,
                                                   const OutputActorPtr &output_actor);
  void CollectBackendParameterForDynamicShape(AnyTypeKernelActor *const any_type_kernel_actor,
                                              const KernelGraphPtr &model_graph, const KernelGraphPtr &real_graph);
  void FixDeviceTensorStoreKeyInActor(const std::vector<AbstractActorPtr> &actors,
                                      AnyTypeKernelActor *const any_type_kernel_actor,
                                      const KernelGraphPtr &model_graph, const GraphCompilerInfo &graph_compiler_info,
                                      const std::vector<AnfNodePtr> &front_parameters);
};
}  // namespace runtime
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ANY_TYPE_GRAPH_SCHEDULER_H_
