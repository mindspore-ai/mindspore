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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_DEVICE_CONTEXT_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_DEVICE_CONTEXT_H_

#include <vector>
#include <memory>
#include <string>
#include <set>
#include <map>
#include "runtime/hardware/device_context.h"
#include "runtime/hardware/device_context_manager.h"
#include "runtime/device/memory_manager.h"
#include "plugin/device/ascend/hal/device/ascend_kernel_runtime.h"
#include "plugin/device/ascend/hal/device/ascend_device_address.h"

namespace mindspore {
namespace device {
namespace ascend {
class AscendDeviceContext : public DeviceContext {
 public:
  explicit AscendDeviceContext(const DeviceContextKey &device_context_key)
      : DeviceContext(device_context_key), mem_manager_(nullptr), initialized_(false) {}
  ~AscendDeviceContext() override = default;

  // Initialize the device context.
  void Initialize() override;

  // Destroy device context and release device resource.
  void Destroy() override;

  // Get rank id for distributed training.
  uint32_t GetRankID() const override { return rank_id_; }

  // Partition the function graph through the device capability and return the partition segments.
  // The second parameter is the default partition segments which are provided by the framework.
  // Device can reprocess the default partition segments to new segments, also can partition the function graph again.
  // If Device can launch the whole graph and not expect partitioning the function graph, then return the empty
  // segments. The default behavior is return the default partition segments.
  std::vector<GraphSegmentPtr> PartitionGraph(const FuncGraphPtr &func_graph,
                                              const std::vector<GraphSegmentPtr> &default_partition_segments) override;

  // Optimize the kernel graph for graph mode.
  void OptimizeGraph(const KernelGraphPtr &graph) const override;

  // Optimize the single operator graph for PyNative mode.
  void OptimizeSingleOpGraph(const KernelGraphPtr &graph) const override;

  // Generate 'KernelMod' for all kernels and set 'KernelMod' into kernel,
  // 'KernelMod' is real executive object of kernel.
  void CreateKernel(const std::vector<CNodePtr> &nodes) const override;

  // Adjust kernel graph before run graph, used in Graph Mode.
  void PreprocessBeforeRunGraph(const KernelGraphPtr &graph) const override;
  // Adjust single op kernel graph before run graph, used in PyNative Mode.
  void PreprocessBeforeRunSingleOpGraph(const KernelGraphPtr &graph) const override;

  // Relevant function to allocate and free device memory of raw ptr.
  void *AllocateMemory(size_t size) const override;
  void FreeMemory(void *ptr) const override;

  // Allocate continuous device memory according to size list.
  // Communication operators may need continuous memory for input and output
  // to optimize the communication performance.
  std::vector<void *> AllocateContinuousMemory(const std::vector<size_t> &size_list) const override;

  // Create concrete device address according different device type.
  DeviceAddressPtr CreateDeviceAddress(void *const device_ptr, size_t device_size, const string &format, TypeId type_id,
                                       const ShapeVector &shape = ShapeVector()) const override;

  // Launch graph, device such as Ascend support the whole graph sink to the device executing.
  bool LaunchGraph(const KernelGraphPtr &graph) const override;

  // Launch a kernel via 'KernelMod' of the kernel.
  bool LaunchKernel(const CNodePtr &kernel, const std::vector<AddressPtr> &inputs,
                    const std::vector<AddressPtr> &workspace, const std::vector<AddressPtr> &outputs) const override;

  // Synchronize stream, device such as GPU and Ascend need stream to launch kernel asynchronously,
  // using 'SyncStream' to block thread and wait for completing all tasks in stream.
  // Devices that do not need stream could ignore the implementation of this function.
  bool SyncStream(size_t stream_id = 0) const override;

  // Create and initialize bucket for every allreduce operator. Bucket is used in PyNative distributed training mode,
  // one bucket handles all resource to launch and sync allreduce operator.
  std::shared_ptr<Bucket> CreateBucket(uint32_t bucket_id, uint32_t bucket_size) const override;

  // Unify the MindIR, the default behavior uses the common unified MindIR.
  void UnifyMindIR(const KernelGraphPtr &graph) const override;

  // Whether the graph sink executing through the device capability, the default behavior is not sink and return false.
  bool IsExecutingSink(const KernelGraphPtr &graph) const override;
  // Whether the graph loop sink executing through the device capability, the default behavior is not loop sink and
  // return false.
  bool IsLoopCountSink(const KernelGraphPtr &graph) const override;

  // set rt_context_ to this thread to control device
  bool BindDeviceToCurrentThread() const override;

  // Launch device aicpu library
  void LaunchDeviceLibrary() const;

 private:
  // Graph loader interface
  void AllocateGraphMemory(const NotNull<KernelGraphPtr> &root_graph) const;
  void AssignInputMemory(const NotNull<KernelGraphPtr> &graph, NotNull<std::set<KernelGraphPtr> *> memo) const;
  void LoadModel(const NotNull<KernelGraphPtr> &root_graph) const;
  void UpdateExecOrder(const KernelGraphPtr &graph) const;
  static bool IsGraphMode();
  bool PySyncRuning() const;
  bool MemoryCopyAsync(const CNodePtr &node, const vector<AddressPtr> &inputs, const vector<AddressPtr> &outputs) const;
  void InsertEventBeforeRunTask(const KernelGraphPtr &graph) const;
  void SetAtomicCleanToNodes(const KernelGraphPtr &graph,
                             const std::map<CNodePtr, std::vector<CNodePtr>> &atomics_node) const;

  void ReportErrorMessage() const;
  void ReportWarningMessage() const;
  void SetErrorManagerContext() const;

  // Really create an ascend stream.
  bool CreateStream(void **stream) const override;

  // Really destroy an ascend stream.
  bool DestroyStream(void *stream) const override;

  // Kernel Runtime  --- only for task sink
  AscendKernelRuntime *runtime_instance_{nullptr};
  std::shared_ptr<MemoryManager> mem_manager_{nullptr};
  // rank id of physical device
  uint32_t rank_id_{0};
  bool initialized_{false};

  // LaunchGraph interface
  bool ExecuteGraph(const KernelGraphPtr &graph) const;
  // The ExecuteGraph is not thread safety specifically, it is not recommended that multiple threads access the same
  // func at the same time, so need the launch mutex when multiple threads launch the graph.
  mutable std::mutex launch_mutex_;
  // Using node to get it's atomics
  mutable std::map<CNodePtr, std::vector<CNodePtr>> node_atomics_;
  // Persistent cache for single op execution.
  // node_atomics_ will be cleaned up in CompileGraph.
  mutable std::map<CNodePtr, std::vector<CNodePtr>> node_atomics_persistent_cache_;
  mutable std::set<CNodePtr> nop_op_to_memcpy_;
  // Event for multi-stream
  mutable std::map<uint32_t, std::shared_ptr<DeviceEvent>> graph_event_;
  // Some NOP nodes have be hide in execution order, it doesn't have output device address, this function creates
  // output device address for these nodes, and the output device address is the same with input device address.
  void AssignOutputNopNodeDeviceAddress(const KernelGraphPtr &graph) const;
  bool LaunchAtomicClean(const CNodePtr &node, const std::vector<AddressPtr> &workspace,
                         const std::vector<AddressPtr> &outputs) const;
  void *compute_stream_;
  void *communication_stream_;
  void *GetKernelStream(const CNodePtr &node) const;
  bool GetKernelRealInputs(const CNodePtr &kernel, const vector<AddressPtr> &inputs,
                           std::vector<AddressPtr> *real_inputs) const;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_DEVICE_CONTEXT_H_
