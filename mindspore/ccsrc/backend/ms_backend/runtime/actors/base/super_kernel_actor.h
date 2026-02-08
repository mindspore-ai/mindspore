/**
 * Copyright 2021-2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_SUPER_KERNEL_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_SUPER_KERNEL_ACTOR_H_

#include <string>
#include <memory>
#include <map>
#include <utility>
#include <vector>
#include <queue>
#include <set>
#include "backend/ms_backend/runtime/actors/base/debug_aware_actor.h"
#include "backend/ms_backend/runtime/actors/base/actor_common.h"
#include "backend/ms_backend/runtime/actors/base/kernel_runner.h"
#include "backend/ms_backend/runtime/actors/base/kernel_async_launch_actor.h"
#include "backend/ms_backend/runtime/actors/dynamic_shape/kernel_async_infer_actor.h"
#include "backend/ms_backend/runtime/actors/dynamic_shape/kernel_async_resize_actor.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/pipeline/async_rqueue.h"
#include "tools/profiler/profiling.h"
#include "ir/anf.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceAddress;
using mindspore::device::DeviceContext;
using profiler::Profiler;

struct OutputMemoryInfo {
  size_t size;
  std::string node_full_name;
};

// Struct is used to represent the output information of node and is used to mark whether the output has been released
// when get the release position of address ptr.
struct FreeNodeInfo {
  device::DeviceContextKey context_key;
  std::string branch_name;
  bool operator<(const FreeNodeInfo &other) const {
    if (context_key.device_id_ < other.context_key.device_id_) {
      return true;
    }
    if (context_key.device_id_ > other.context_key.device_id_) {
      return false;
    }
    if (context_key.device_type_ < other.context_key.device_type_) {
      return true;
    }
    if (context_key.device_type_ > other.context_key.device_type_) {
      return false;
    }
    return branch_name < other.branch_name;
  }
};

// The Super kernel actor is used to represent the sink executing of graph which is the combination of kernels.
class SuperKernelActor : public DebugAwareActor {
 public:
  SuperKernelActor(const std::string &name, const KernelGraphPtr &graph, const std::string &graph_phase,
                   const DeviceContext *device_context, const AID &memory_manager_aid, const AID *debug_aid,
                   const AID *recorder_aid, KernelTransformType type = KernelTransformType::kSuperKernelActor);
  ~SuperKernelActor() override;

  size_t FetchInputNodePosition(const AnfNodePtr &intput_node);
  virtual void FetchInputDeviceTensor(OpContext<KernelTensor> *const context);

  // The input may come from the control actor, so need free the input memory by the dynamic ref count.
  void SendMemoryFreeReq(OpContext<KernelTensor> *const context) override;

  const KernelGraphPtr &graph() const { return graph_; }

  void BuildAndLinkKernelActors();
  const std::vector<KernelRunnerPtr> &kernel_actors() const { return kernel_actors_; }
  const std::vector<size_t> &input_param_static_use_cnt() const { return input_params_use_cnt_; }
  const std::vector<bool> &is_input_used() const { return is_input_used_; }

  bool LaunchKernel(OpContext<KernelTensor> *const context, const KernelRunnerPtr &kernel_actor, bool hp_mode,
                    bool sync_run = false);

  // Only used for the aclgraph feature, At the capture stage, when dispatching operators.
  bool LaunchKernelForCaptureGraph(OpContext<KernelTensor> *const context, const KernelRunnerPtr &kernel_actor,
                                   size_t index, bool is_graph);

  // Only used for the aclgraph feature, At the replay stage, when dispatching operators.
  bool LaunchKernelForReplayGraph(OpContext<KernelTensor> *const context, const KernelRunnerPtr &kernel_actor,
                                  size_t index);

  bool enable_kbk_sub_graph_execute() const { return enable_kbk_sub_graph_execute_; }

  bool enable_inline_control_flow() const { return enable_inline_control_flow_; }
  bool enable_infer_boost() const { return enable_infer_boost_; }
  const mindspore::HashMap<AnfNode *, std::vector<std::pair<size_t, size_t>>> &kernel_input_to_graph_input_indices()
    const {
    return kernel_input_to_graph_input_indices_;
  }
  const mindspore::HashMap<AnfNode *, std::vector<std::pair<size_t, std::vector<size_t>>>> &
  kernel_input_to_actor_output_indices() const {
    return kernel_input_to_actor_output_indices_;
  }
  const std::set<std::pair<size_t, ParameterInfo>> &input_params_no_user() const { return input_params_no_user_; }

  void IncreaseNewRefCounts(OpContext<KernelTensor> *const context) override;
  // Get the release position of the device address in the graph through static analysis of the input-output
  // relationship in the graph.
  void SetFreePositionForKernelActor();
  void SetInputFreePositionForKernelActor(
    const KernelRunnerPtr &kernel_actor,
    const mindspore::HashMap<AnfNodePtr, device::DeviceContextKey> &kernel_to_context_key,
    const device::DeviceContextKey &graph_device_context_key,
    std::set<std::pair<KernelWithIndex, FreeNodeInfo>> *checked_nodes);
  void SetOutputFreePositionForKernelActor(
    const KernelRunnerPtr &kernel_actor,
    const mindspore::HashMap<AnfNodePtr, device::DeviceContextKey> &kernel_to_context_key,
    const device::DeviceContextKey &graph_device_context_key,
    std::set<std::pair<KernelWithIndex, FreeNodeInfo>> *checked_nodes);
  void GetRefCountForGraphOutput(const std::vector<AnfNodePtr> &output_data_nodes,
                                 const std::vector<DataArrowPtr> &output_data_arrows,
                                 const mindspore::HashMap<AnfNodePtr, KernelRunner *> &kernel_to_actor,
                                 const std::map<uint32_t, std::vector<CNodePtr>> &inplace_groups,
                                 const std::string &actor_name);

  // Collect conditions at compilation and execution phase to judge whether running with high performance mode.
  bool IsHighPerfModeAtComp();
  bool IsHighPerfModeAtExec();
  // Reset state for UCE, ARF.
  void ResetState(OpContext<KernelTensor> *const context) override;

  void UpdateOutputKernelTensors(const std::vector<std::pair<KernelTensorPtr, size_t>> &new_kernel_pair,
                                 const std::vector<KernelTensorPtr> &output_kernel_tensors);

 protected:
  void Init() override;
  void Run(OpContext<KernelTensor> *const context) override;
  void Finalize() override;
  // The input device tensors for launch.
  std::vector<KernelTensorPtr> input_kernel_tensors_;
  // The device tensors of graph input parameter, which used to compare the recv input data.
  std::vector<KernelTensorPtr> node_kernel_tensors_;
  // The device tensors for memory alloc.
  std::vector<KernelTensorPtr> memory_alloc_list_;
  // The lists of device tensors which need free by dynamic ref count, will be cleared at the end of step.
  std::queue<std::vector<KernelTensorPtr>> memory_free_lists_;

 protected:
  // Generate and initialize all kernel actors by execution order of graph_ for kerkel by kernl execute a sub garph
  // mode.
  void BuildKernelActors();
  // Generate KernelRunner for each Kernel if need.
  void GenerateKernelRunners();
  KernelRunnerPtr BuildInnerControlFlowActor(const CNodePtr &kernel, const DeviceContext *device_context,
                                             GraphExecutionStrategy strategy, const std::set<size_t> &ref_input_indexes,
                                             const std::set<size_t> &ref_output_indexes);

  // Parse all nodes dependence of graph_, record device tensor store key of every kernel, calculate original ref
  // count of CNode and Parameter, prepare input and heterogeneous output device address of all kernels.
  void LinkKernelActors();
  // When there is control flow in the graph, in order to control the execution of the kernel actor the relationship
  // between condition actor, as well as the relationship between condition actor and kernel actors, should be set.
  void SetRelationForControlFlow();
  void AnalyseNodesDependence(const HashMap<size_t, AnfNodePtr> &device_tensor_store_keys_map,
                              const HashMap<size_t, ParameterInfo> &parameter_indexs_map,
                              const HashMap<AnfNodePtr, std::vector<size_t>> &output_node_to_actor_output_index,
                              std::vector<std::pair<KernelRunnerPtr, size_t>> *param_first_used_kernel_actors);

  void LinkKernelActor(const CNodePtr &kernel, size_t input_index, const AnfNodePtr &input_node, size_t output_index);
  void LinkKernelActorByDeviceType(const CNodePtr &kernel, size_t input_index, const AnfNodePtr &input_node,
                                   size_t output_index);
  void CheckDeviceTypeValid(const DeviceTensorPtr &input_device_tensor, const DeviceContext *input_device_context,
                            const CNodePtr &kernel, const AnfNodePtr &input_kernel, size_t input_index);

  void RunGraphKernelByKernel(OpContext<KernelTensor> *const context);

  void UpdateMemoryTraceMangerStatus(OpContext<KernelTensor> *const context);
  void SetTraceMemoryForKernel(const KernelRunnerPtr &kernel_actor, bool safe_update = false);
  // Allocate block memory for use trace memory (run by static shape) step.
  void AllocateTraceMemory(OpContext<KernelTensor> *const context) const;
  // Free block memory for use trace memory (run by static shape) step.
  void FreeTraceMemory() const;
  void SetInputTraceMemory(const KernelRunnerPtr &kernel_actor) const;

  // Handle copy output for different device type kernel.
  bool CopyHeterogeneousOutput(OpContext<KernelTensor> *const context, const KernelRunnerPtr &kernel_actor) const;

  void UpdateOutputAddress(const std::vector<std::pair<size_t, std::vector<size_t>>> &kernel_inputs_to_actor_outputs,
                           const KernelRunnerPtr &kernel_actor);

  // Launch all kernels by execution order in kernel graph: graph_.
  bool LaunchAllKernels(OpContext<KernelTensor> *const context, bool hp_mode);

  // Async launch a kernel by debug mode or high performance mode.
  void AsyncLaunchKernelByCondition(OpContext<KernelTensor> *const context, KernelRunner *kernel_actor, bool hp_mode);

  // Sync dispatch a kernel, including infer/resize/launch.
  void SyncDispatchKernel(OpContext<KernelTensor> *const context, KernelRunner *kernel_actor, bool hp_mode);

  void FetchParameterInput(const KernelRunnerPtr &kernel_actor, OpContext<KernelTensor> *const context,
                           size_t stream_id = SIZE_MAX);
  void FreeInputParamWithoutUser(OpContext<KernelTensor> *const context);
  void RecordKernelActorWeight();

  void HandleFirstUserInputMemoryFree(const KernelRunnerPtr &kernel_actor, size_t kernel_input_index);

  // Prepare non top cell input, such as internal parameter msg input, control flow msg input and const value.
  bool FetchMsgInputAndConstValueForKernel(KernelRunner *kernel_actor, OpContext<KernelTensor> *const context);

  void ParallelDispatchKernels(OpContext<KernelTensor> *const context);
  // Dispatch kernel which can parallel launch.
  void DispatchParallelLaunchKernels(size_t index, OpContext<KernelTensor> *const context);
  // Dispatch serial launch kernels: communication ops and the kernel need force resize.
  void DispatchSerialLaunchKernels(OpContext<KernelTensor> *const context);

  void InitParallelDispatchResource();
  void PartitionParallelDispatchKernels();
  // Recreate the communication group for the communication operators, ensuring that the communication group is the
  // same for the communication operators on each concurrent thread.
  void RecreateCommunicationGroup();
  void ClearParallelDispatchResource();

  friend class GraphScheduler;
  KernelGraphPtr graph_;

  // The phase of the root graph this super actor belongs to.
  std::string graph_phase_;
  // Whether the super kernel actor is a infer 'prefill' or 'increment' graph or not.
  bool is_infer_phase_;

  // In the scheduler, check whether the parameters need to be copied after lunch. Only when the parameter has
  // the ref attribute and is directly used by the kernel in the graph, it needs to be copied.
  std::vector<bool> is_parameters_need_copy_;

  // Record the address map of ref node to copy back when running finished.
  std::map<kernel::KernelTensorPtr, kernel::KernelTensorPtr> ref_node_addr_map_;

  // The received input device type and format may be different from the formal parameter in the control flow scenarios,
  // so it needs to be copied from the input data to real data that graph launch needs.
  std::vector<KernelTensorPtr> copy_input_kernel_tensors_;
  // Record the device address to the output node of graph.
  std::map<DeviceAddress *, OutputMemoryInfo> device_address_to_node_;

  // Record the use count of all input nodes(parameter) of graph_, use to correct current ref count in runtime.
  std::vector<size_t> input_params_use_cnt_;

  // Record the graph parameter without user.
  std::set<std::pair<size_t, ParameterInfo>> input_params_no_user_;

  std::vector<KernelTensorPtr> new_memory_free_list_;

  // Record whether the input is used by kernel actor.
  std::vector<bool> is_input_used_;
  // Record which kernel actor should insert event when fetch parameter on non-default stream.
  mindspore::HashSet<KernelRunner *> kernel_actors_insert_event_;

  // Record all parameter nodes of graph_ and their index positions in graph_'s input_nodes.
  mindspore::HashMap<AnfNode *, size_t> param_node_to_input_idx_;

  // Kernel by kernel sub graph execute mode need not send actor message.
  bool enable_kbk_sub_graph_execute_;
  bool already_fetch_persistent_device_tensor_{false};
  mindspore::HashMap<AnfNodePtr, KernelRunner *> cnode_to_kernel_actor_;
  std::vector<KernelRunnerPtr> kernel_actors_;
  // Indices from other actor.
  mindspore::HashMap<AnfNode *, std::vector<std::pair<size_t, size_t>>> kernel_input_to_graph_input_indices_;
  mindspore::HashMap<AnfNode *, std::vector<std::pair<size_t, std::vector<size_t>>>>
    kernel_input_to_actor_output_indices_;
  SomasInfo *somas_info_;

  AID kernel_async_infer_aid_;
  AID kernel_async_resize_aid_;
  AID kernel_async_launch_aid_;

  bool enable_trace_memory_;

  // The variables for parallel dispatch kernel.
  bool enable_parallel_dispatch_{false};
  std::vector<std::vector<KernelRunnerPtr>> parallel_launch_kernels_;
  std::vector<KernelRunnerPtr> serial_launch_kernels_;
  HashMap<KernelRunner *, std::vector<DeviceEventPtr>> serial_launch_kernels_to_events_;

  bool enable_capture_graph_{false};
  bool in_increment_{false};
  static bool already_allocate_trace_memory_;

  static size_t parallel_dispatch_num_;
  static size_t parallel_slice_num_;

  static std::vector<std::pair<size_t, void *>> streams_;
  static std::vector<DeviceEventPtr> events_;
  static std::vector<DeviceEventPtr> events_to_default_stream_;
  static std::vector<AsyncRQueuePtr> queues_;

  bool enable_infer_boost_{false};

  // Whether the actor include a control flow actor.
  bool enable_inline_control_flow_{false};

  std::vector<std::shared_ptr<Profiler>> prof_instances_;
  bool is_high_perf_mode_{true};
};

using SuperKernelActorPtr = std::shared_ptr<SuperKernelActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_KERNEL_ACTOR_H_
