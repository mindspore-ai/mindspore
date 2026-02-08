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

#ifndef MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_GRAPH_SCHEDULER_H_
#define MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_GRAPH_SCHEDULER_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <map>
#include <set>
#include <algorithm>
#include <fstream>

#include "utils/hash_map.h"
#include "backend/ge_backend/runtime/control_node_scheduler.h"
#include "backend/ge_backend/runtime/actor/actor_set.h"
#include "backend/ge_backend/runtime/graph_compiler.h"
#include "backend/ge_backend/runtime/actor/actor_dump.h"
#include "thread/actor_threadpool.h"

namespace mindspore {
namespace ge_backend {
namespace runtime {
using mindspore::session::KernelGraph;
using mindspore::session::KernelWithIndex;

// The second element of pair represents the output node and output index of abstract actor corresponding to the graph
// output node.
using GraphOutputPair = std::pair<AbstractActor *, KernelWithIndex>;

class GraphScheduler {
 public:
  static GraphScheduler &GetInstance() noexcept;

  // 1. Thread pool creating.
  // 2. The global actors creating and scheduling.
  void Initialize();

  // Clear the members.
  void Clear();
  void Clear(const ActorInfo &actor_info, const std::vector<KernelGraphPtr> &graphs,
             const std::vector<AnfNodePtr> &root_graph_parameters,
             const ControlNodeParserPtr &parser = nullptr) noexcept;
  // The control flow actors will generate some data in the loop body execution, so need clear on the end of execution.
  void ClearActorData(const ActorSet *actor_set);

  // Transform graph to actor DAG, contains build and link.
  ActorSet *Transform(const GraphCompilerInfo &graph_compiler_info);

  // Schedule actors in the actor runtime. Single machine scheduling is supported currently, and distributed scheduling
  // will be supported in the future.
  void Schedule(const ActorSet *actor_set);

  // The processing entry of actors running. The fourth parameter is used only in the step execution strategy.
  void Run(ActorSet *constactor_set, const std::vector<std::vector<TensorPtr>> &input_tensors, const VectorRef &args,
           GraphExecutionStrategy strategy = GraphExecutionStrategy::kPipeline);

  // Fetch the actor set by actor info.
  ActorSet *Fetch(const ActorInfo &actor_info) const;
  // Fetch the actor set by actor_id.
  ActorSet *Fetch(uint32_t actor_id) const;

  // Whether graph scheduler is initialized.
  bool initialized() const { return init_; }

  // The callback function after process fork finish to reinitialize multi pipeline actors.
  void ChildAfterFork();

 private:
  GraphScheduler() = default;
  ~GraphScheduler() = default;
  DISABLE_COPY_AND_ASSIGN(GraphScheduler);

  // The Global actors contain memory manager actor, recorder actor and debug actor.
  void BuildAndScheduleGlobalActor();

  // Transform the nodes of graph to actors.
  ActorSetPtr Build(const GraphCompilerInfo &graph_compiler_info);
  // Link actors to DAG through the edge connection of graph and graph execution strategy.
  void Link(ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info);

  std::vector<AnfNodePtr> GatherAllParams(const GraphCompilerInfo &graph_compiler_info);

  // The processing of actors build.
  std::vector<DataSourceActorPtr> BuildDataSourceActor(const GraphCompilerInfo &graph_compiler_info,
                                                       const HostTensorQueuePtr &host_queue);
  std::vector<SuperKernelActorPtr> BuildSuperKernelActor(const GraphCompilerInfo &graph_compiler_info);
  LoopCountActorPtr BuildLoopCountActor(const GraphCompilerInfo &graph_compiler_info);
  OutputActorPtr BuildOutputActor(const GraphCompilerInfo &graph_compiler_info) const;
  DataPrepareActorPtr BuildDataPrepareActor(const GraphCompilerInfo &graph_compiler_info,
                                            const std::vector<DataSourceActorPtr> &data_source_actors,
                                            const HostTensorQueuePtr &host_queue);
  std::vector<AbstractActorPtr> BuildNoInputKernelActor(const ActorSet *actor_set,
                                                        GraphExecutionStrategy strategy) const;

  // Cache the information of graph output node to actor between “build” and “link”, for linking between the tail of
  // previous graph and the head of next graph.
  void CacheGraphOutputToActor(const GraphCompilerInfo &graph_compiler_info);

  // The processing of actors linking.
  // 1. The processing of linking data arrows.
  void LinkDataArrowInSinkMode(const KernelGraphPtr &graph, const GraphCompilerInfo &graph_compiler_info,
                               std::vector<AbstractActor *> *const auto_monad_actors);

  // The gather of linking data arrows of kernel, it will call following functions by the different from actor type.
  void LinkDataArrow(AbstractActor *const to_actor, const GraphCompilerInfo &graph_compiler_info,
                     const KernelGraphPtr &graph, const KernelWithIndex &from_kernel_with_output_idx,
                     const KernelWithIndex &to_kernel_with_input_idx);
  void LinkDataArrowForBaseActor(AbstractActor *const from_actor, AbstractActor *const to_actor,
                                 const KernelWithIndex &from_kernel_with_output_idx,
                                 const KernelWithIndex &to_kernel_with_input_idx, const KernelGraphPtr &graph);
  // Link data arrows for internal parameter, convert internal parameter to actor by internal parameter cache to link.
  void LinkDataArrowForInternalParameter(AbstractActor *const from_actor, AbstractActor *const to_actor,
                                         const KernelWithIndex &from_kernel_with_output_idx,
                                         const KernelWithIndex &to_kernel_with_input_idx, const KernelGraphPtr &graph);
  void LinkDataArrowForDeviceTensorStore(AbstractActor *const from_actor, AbstractActor *const to_actor,
                                         const KernelWithIndex &from_kernel_with_output_idx,
                                         const KernelWithIndex &to_kernel_with_input_idx, const KernelGraphPtr &graph);

  void LinkDataArrowForHostDSActor(AbstractActor *const from_actor, AbstractActor *const to_actor,
                                   const KernelWithIndex &from_kernel_with_output_idx,
                                   const KernelWithIndex &to_kernel_with_input_idx, const KernelGraphPtr &graph);

  // 2. The processing of linking control arrows.
  // The parameter cnode_to_monad_inputs contains all the update states that each cnode in the graph depends on. When
  // processing the first input of update state, the map is used to check whether it is necessary to link control arrow
  // for the first input of update state.
  void LinkControlArrowByAutoMonad(
    AbstractActor *to_actor, const AnfNodePtr &from_node, const KernelGraphPtr &graph,
    const ControlNodeParserPtr &parser = nullptr,
    const mindspore::HashMap<AnfNodePtr, std::set<AnfNodePtr>> &cnode_to_monad_inputs = {},
    std::set<AnfNodePtr> *checked_nodes = nullptr);

  // The gather of linking the global control arrows, it will call following functions:
  void LinkGlobalControlArrow(ActorSet *const actor_set, const std::vector<AbstractActor *> &auto_monad_actors,
                              const GraphCompilerInfo &graph_compiler_info);

  void LinkControlArrowForDataPrepareActor(DataPrepareActor *data_prepare_actor, const ActorSet *actor_set,
                                           const ControlNodeParserPtr &parser) const;
  void LinkControlArrowForLoopCountActor(LoopCountActor *loop_count_actor, const ActorSet *actor_set,
                                         const ControlNodeParserPtr &parser);
  void LinkControlArrowForOutputActor(OutputActor *output_actor, const ActorSet *actor_set) const;

  // 3. The processing of linking output result arrows.
  void LinkOutputResultArrowForOutputActor(OutputActor *to_actor, const GraphCompilerInfo &graph_compiler_info) const;

  void CorrectControlArrowForAutoMonadActor(AbstractActor *const auto_monad_actor, const AbstractActorPtr &copy_actor);

  // Persist device tensors of graph's some nodes(such as weights and value nodes).
  void PersistDeviceTensor(const GraphCompilerInfo &graph_compiler_info) const;
  void PersistDeviceTensorForValueNode(const AnfNodePtr &value_node, const KernelGraphPtr &graph) const;
  void PersistDeviceTensorForParameter(const AnfNodePtr &parameter, const KernelGraphPtr &graph,
                                       const GraphCompilerInfo &graph_compiler_info) const;
  // When the parameters of root graph are not in backend kernel graphs, need persist device tensor by this function.
  void PersistDeviceTensorForRootGraphControlNode(const GraphCompilerInfo &graph_compiler_info) const;

  // Display the actor information of corresponding kernel graph.
  void DumpActor(const ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info) const;
  void DumpDeviceTensorStore(const GraphCompilerInfo &graph_compiler_info, std::ofstream &ofs) const;
  void DumpFinalActor(const ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info);

  // The global maps, only be cleared in the deconstruction.
  mindspore::HashMap<ActorInfo, ActorSetPtr> actors_;

  // The local maps and vectors, will be cleared at the end of each graph transform:
  // 1.The second element of pair represents the output index of op actor corresponding to the graph output front node.
  std::map<KernelWithIndex, GraphOutputPair, session::KernelWithIndexCmp> graph_output_to_actor_;

  // In the control flow, used to build and link control actor.
  ControlNodeScheduler control_node_scheduler_;

  // The id of global actor.
  AID memory_manager_aid_;
  const AID *recorder_aid_{nullptr};
  const AID *debug_aid_{nullptr};
  const AID *profiler_aid_{nullptr};

  size_t default_actor_thread_num_{1};

  bool init_{false};
};
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_GRAPH_SCHEDULER_H_
