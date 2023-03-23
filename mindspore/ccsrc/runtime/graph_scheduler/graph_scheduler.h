/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_GRAPH_SCHEDULER_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_GRAPH_SCHEDULER_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <map>
#include <set>
#include <algorithm>
#include <fstream>
#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "runtime/graph_scheduler/control_node_scheduler.h"
#include "runtime/graph_scheduler/memory_swap_node_scheduler.h"
#include "runtime/graph_scheduler/actor/actor_set.h"
#include "runtime/graph_scheduler/graph_compiler.h"
#include "runtime/graph_scheduler/actor/actor_dump.h"
#include "thread/actor_threadpool.h"

#ifdef ENABLE_RPC_ACTOR
#include "runtime/graph_scheduler/rpc_node_scheduler.h"
#endif
#include "include/backend/visible.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceContext;
using mindspore::session::KernelGraph;
using mindspore::session::KernelWithIndex;

// The second element of pair represents the output node and output index of abstract actor corresponding to the graph
// output node.
using GraphOutputPair = std::pair<AbstractActor *, KernelWithIndex>;

class BACKEND_EXPORT GraphScheduler {
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
  void Run(ActorSet *constactor_set, const std::vector<std::vector<TensorPtr>> &input_tensors,
           GraphExecutionStrategy strategy = GraphExecutionStrategy::kPipeline);

  // Fetch the actor set by actor info.
  ActorSet *Fetch(const ActorInfo &actor_info) const;

  // Whether graph scheduler is initialized.
  bool initialized() const { return init_; }

#ifdef ENABLE_RPC_ACTOR
  // Returns pointer of RpcNodeScheduler to distributed module.
  RpcNodeScheduler *rpc_node_scheduler() { return rpc_node_scheduler_.get(); }
#endif

 private:
  GraphScheduler() = default;
  ~GraphScheduler() = default;
  DISABLE_COPY_AND_ASSIGN(GraphScheduler);

  // Set using the multi thread or single thread to execute the actor set by the execution time compared.
  void SetActorExecutionStrategy(ActorSet *const actor_set, GraphExecutionStrategy strategy,
                                 double execution_time) const;

  // Check whether the single thread execution condition is met.
  bool CheckSingleThreadRunningCondition(ActorSet *const actor_set, GraphExecutionStrategy strategy) const;

  // The Global actors contain memory manager actor, recorder actor and debug actor.
  void BuildAndScheduleGlobalActor();

  // Transform the nodes of graph to actors.
  ActorSetPtr Build(const GraphCompilerInfo &graph_compiler_info);
  // Link actors to DAG through the edge connection of graph and graph execution strategy.
  void Link(ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info);
  // Optimize the actor DAG. For example, erase invalid data arrow, etc.
  void Optimize(const ActorSetPtr &actor_set) const;

  // The processing of actors build.
  std::vector<DataSourceActorPtr> BuildDataSourceActor(const GraphCompilerInfo &graph_compiler_info,
                                                       const HostTensorQueuePtr &host_queue);
  std::vector<KernelActorPtr> BuildKernelActor(const GraphCompilerInfo &graph_compiler_info);
  std::vector<CustomActorPtr> BuildCustomActor(const GraphCompilerInfo &graph_compiler_info);
  std::vector<SuperKernelActorPtr> BuildSuperKernelActor(const GraphCompilerInfo &graph_compiler_info);
  LoopCountActorPtr BuildLoopCountActor(const GraphCompilerInfo &graph_compiler_info);
  OutputActorPtr BuildOutputActor(const GraphCompilerInfo &graph_compiler_info) const;
  DataPrepareActorPtr BuildDataPrepareActor(const GraphCompilerInfo &graph_compiler_info,
                                            const std::vector<DataSourceActorPtr> &data_source_actors,
                                            const HostTensorQueuePtr &host_queue);
  std::vector<AbstractActorPtr> BuildNoInputKernelActor(const ActorSet *actor_set,
                                                        GraphExecutionStrategy strategy) const;

  // Generate rpc actor object inherited from kernel actor.
  KernelActorPtr GenerateRpcActor(const CNodePtr &kernel, const DeviceContext *device_context,
                                  GraphExecutionStrategy strategy, const std::set<size_t> &modifiable_ref_input_indexes,
                                  const std::set<size_t> &modifiable_ref_output_indexes);

  // Cache the information of graph output node to actor between “build” and “link”, for linking between the tail of
  // previous graph and the head of next graph.
  void CacheGraphOutputToActor(const GraphCompilerInfo &graph_compiler_info);
  // The input and output of ref node may be in the different subgraphs, so need the global subgraphs info to update the
  // device address of ref node.
  void UpdateDeviceAddressByRefInternalParameter(const GraphCompilerInfo &graph_compiler_info);

  // The processing of actors linking.
  // 1. The processing of linking data arrows.
  void LinkDataArrowInSinkMode(const KernelGraphPtr &graph, const GraphCompilerInfo &graph_compiler_info,
                               std::vector<AbstractActor *> *const auto_monad_actors);
  void LinkDataArrowInNonSinkMode(const KernelGraphPtr &graph, const GraphCompilerInfo &graph_compiler_info,
                                  std::vector<AbstractActor *> *const auto_monad_actors,
                                  std::vector<CNodePtr> *const communication_nodes);
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
  void LinkDataArrowForKernelActor(AbstractActor *const from_actor, AbstractActor *const to_actor,
                                   const KernelWithIndex &from_kernel_with_output_idx,
                                   const KernelWithIndex &to_kernel_with_input_idx, const KernelGraphPtr &graph);
  // Link data arrows in the copy actor scene, insert the copy actor between from_actor and to_actor.
  void LinkDataArrowForCopyActor(AbstractActor *const from_actor, AbstractActor *const to_actor,
                                 const KernelWithIndex &from_kernel_with_output_idx,
                                 const KernelWithIndex &to_kernel_with_input_idx);

  // 2. The processing of linking control arrows.
  // The parameter cnode_to_monad_inputs contains all the update states that each cnode in the graph depends on. When
  // processing the first input of update state, the map is used to check whether it is necessary to link control arrow
  // for the first input of update state.
  void LinkControlArrowByAutoMonad(
    AbstractActor *to_actor, const AnfNodePtr &from_node, const KernelGraphPtr &graph,
    const ControlNodeParserPtr &parser = nullptr,
    const mindspore::HashMap<AnfNodePtr, std::set<AnfNodePtr>> &cnode_to_monad_inputs = {},
    std::set<AnfNodePtr> *checked_nodes = nullptr);
  // The skipped node doesn't run, so need link the control arrow between the inputs and user of skipped node.
  void LinkControlArrowBySkippedNode(AbstractActor *to_actor, const AnfNodePtr &skipped_node,
                                     const KernelGraphPtr &graph) const;
  // Link the control arrows for allreduce kernel by the send/recv nodes in the kernel graph.
  void LinkControlArrowBySendRecvNodes(const KernelGraphPtr &graph) const;

  // The gather of linking the global control arrows, it will call following functions:
  void LinkGlobalControlArrow(ActorSet *const actor_set, const GroupNameToCommuNodes &communication_node_groups,
                              const std::vector<AbstractActor *> &auto_monad_actors,
                              const GraphCompilerInfo &graph_compiler_info);
  void LinkDataArrowForCustomActor(const ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info) const;
  void LinkControlArrowForCustomActor(const ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info);
  void LinkControlArrowByExecutionOrder(const KernelGraphPtr &graph,
                                        const GraphCompilerInfo &graph_compiler_info) const;
  // Link the control arrows by the communication nodes in the kernel graph to ensure communication nodes running order.
  void LinkControlArrowByCommunicationNode(const std::vector<CNodePtr> &communication_nodes,
                                           const std::vector<KernelGraphPtr> &graphs,
                                           const GraphCompilerInfo &graph_compiler_info) const;
  void LinkDeviceTensorStoreForAutoMonadActor(const std::vector<AbstractActor *> &auto_monad_actors);
  void LinkControlArrowForDataPrepareActor(DataPrepareActor *data_prepare_actor, const ActorSet *actor_set,
                                           const ControlNodeParserPtr &parser) const;
  void LinkControlArrowForLoopCountActor(LoopCountActor *loop_count_actor, const ActorSet *actor_set,
                                         const ControlNodeParserPtr &parser);
  void LinkControlArrowForOutputActor(OutputActor *output_actor, const ActorSet *actor_set) const;

  // 3. The processing of linking output result arrows.
  void LinkOutputResultArrowForOutputActor(OutputActor *to_actor, const GraphCompilerInfo &graph_compiler_info) const;

  // Persist device tensors of graph's some nodes(such as weights and value nodes).
  void PersistDeviceTensor(const GraphCompilerInfo &graph_compiler_info) const;
  void PersistDeviceTensorForValueNode(const AnfNodePtr &value_node, const KernelGraphPtr &graph,
                                       const DeviceContext *device_context) const;
  void PersistDeviceTensorForParameter(const AnfNodePtr &parameter, const KernelGraphPtr &graph,
                                       const GraphCompilerInfo &graph_compiler_info,
                                       const DeviceContext *device_context) const;
  // When the parameters of root graph are not in backend kernel graphs, need persist device tensor by this function.
  void PersistDeviceTensorForRootGraphControlNode(const GraphCompilerInfo &graph_compiler_info) const;

  // Display the actor information of corresponding kernel graph.
  void DumpActor(const ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info) const;
  void DumpDeviceTensorStore(const GraphCompilerInfo &graph_compiler_info, std::ofstream &ofs) const;

  // bind thread pool to same numa node
  void BindNumaNode();

  // The global maps, only be cleared in the deconstruction.
  mindspore::HashMap<ActorInfo, ActorSetPtr> actors_;

  // The local maps and vectors, will be cleared at the end of each graph transform:
  // 1.The second element of pair represents the output index of op actor corresponding to the graph output front node.
  std::map<KernelWithIndex, GraphOutputPair, session::KernelWithIndexCmp> graph_output_to_actor_;
  // 2.Beaceuse the copy actors are built in the link, so need record the all copy actors in the link process to push
  // into the actor set after link.
  std::vector<CopyActorPtr> copy_actors_;

  // In the control flow, used to build and link control actor.
  ControlNodeScheduler control_node_scheduler_;

  // Build and link swap actor when memory offload is enabled.
  MemorySwapNodeScheduler swap_node_scheduler_;

#ifdef ENABLE_RPC_ACTOR
  // Return whether the actor set has rpc actors.
  bool HaveRpcActors(const ActorSet *actor_set) const;

  // Used to build and link for rpc actors.
  std::unique_ptr<RpcNodeScheduler> rpc_node_scheduler_{nullptr};
#endif

  // The id of global actor.
  AID memory_manager_aid_;
  const AID *recorder_aid_{nullptr};
  const AID *debug_aid_{nullptr};

  // Whether actor running by the persistent execution order.
  bool execution_order_running_{false};
  // numa library handle
  std::shared_ptr<void> numa_handle_{};

  bool init_{false};
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_GRAPH_SCHEDULER_H_
