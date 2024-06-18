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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_DEVICE_CONTEXT_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_DEVICE_CONTEXT_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "include/backend/device_type.h"
#include "include/backend/device_address.h"
#include "runtime/device/gsm/swap_manager.h"
#include "runtime/collective/collective_communication_lib.h"
#include "runtime/collective/collective_comm_lib_loader.h"
#include "include/backend/kernel_graph.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "runtime/hardware/deprecated_interface.h"
#include "runtime/device/auto_mem_offload.h"
#include "runtime/device/memory_manager.h"
#include "include/backend/optimizer/graph_optimizer.h"
#include "runtime/pipeline/task/task.h"
#include "ir/device_event.h"
#include "utils/ms_context.h"
#include "ir/tensor.h"
#ifdef __APPLE__
#include "mindrt/include/async/spinlock.h"
#endif

namespace mindspore {
namespace device {
using mindspore::kernel::AddressPtr;
using mindspore::kernel::KernelMod;
using mindspore::kernel::KernelTensor;

const size_t kDeviceContextsNumOne = 1;
const size_t kDeviceContextsNumTwo = 2;

struct DeviceContextKey {
  // device type name, such as 'GPU' 'Ascend' 'CPU'.
  std::string device_name_;
  uint32_t device_id_{0};

  // Use the result of ToString() as key to look up DeviceContext
  // in cache map which maintains created DeviceContext objects.
  std::string ToString() const { return device_name_ + "_" + std::to_string(device_id_); }
};

class DeviceResManager;
class GraphExecutor;
class KernelExecutor;

// DeviceContext is unified interface of interaction with device.
class DeviceContext {
 public:
  explicit DeviceContext(const DeviceContextKey &device_context_key)
      : device_context_key_(device_context_key), initialized_(false) {}
  virtual ~DeviceContext() = default;

  // Initialize the device context.
  virtual void Initialize() = 0;

  // Destroy device context and release device resource.
  virtual void Destroy() {}

  // Analysis the function graph to check whether all nodes are supported, if yes, return true, if no, return false and
  // mark the unsupported node as "NotSupport" through SetCNodeNotSupported()
  // For further usage, each device can add a attribute kAttrGraphSplitGroup to the node, and give different
  // group_name (the type must be a std::string, default is 'DefaultGroup') to the attribute, which means the
  // continuous nodes with the same group_name will be split into one subgraph.
  virtual bool PartitionGraph(const FuncGraphPtr &func_graph) const { return false; }

  // Analysis the function graph and select the appropriate run mode for the graph
  virtual RunMode GetRunMode(const FuncGraphPtr &func_graph) const = 0;

  // Get device_context_key_ to obtain device name and device id.
  const DeviceContextKey &device_context_key() const { return device_context_key_; }

  // Get device address type according different device type, such GPU, Ascend.
  DeviceType GetDeviceType() const { return GetDeviceTypeByName(device_context_key_.device_name_); }

  // Get kernel executor by is dynamic shape
  std::shared_ptr<KernelExecutor> GetKernelExecutor(bool is_dynamic_shape) const {
    if (is_dynamic_shape) {
      return dyn_kernel_executor_;
    } else {
      return kernel_executor_;
    }
  }

  void SetKernelExecutor(const std::shared_ptr<KernelExecutor> &kernel_executor) { kernel_executor_ = kernel_executor; }

  void SetDynKernelExecutor(const std::shared_ptr<KernelExecutor> &kernel_executor) {
    dyn_kernel_executor_ = kernel_executor;
  }

  // todo: delete
  virtual DeprecatedInterface *GetDeprecatedInterface() { return nullptr; }

  // Return whether this device context is initialized.
  bool initialized() const {
#ifdef __APPLE__
    std::lock_guard<SpinLock> spin_lock(init_lock_);
#else
    std::lock_guard<std::mutex> lock(init_mutex_);
#endif
    return initialized_;
  }

  DeviceContextKey device_context_key_;
  std::unique_ptr<DeviceResManager> device_res_manager_;
  std::unique_ptr<GraphExecutor> graph_executor_;

 protected:
#ifdef __APPLE__
  // There are some problems with using mutex on Mac, use spinlocks instead.
  inline static SpinLock init_lock_;
#else
  inline static std::mutex init_mutex_;
#endif
  bool initialized_;

 private:
  std::shared_ptr<KernelExecutor> kernel_executor_;
  std::shared_ptr<KernelExecutor> dyn_kernel_executor_;
};
using DeviceContextPtr = std::shared_ptr<DeviceContext>;

class BACKEND_EXPORT DeviceResManager {
 public:
  DeviceResManager() : collective_comm_lib_(nullptr), device_context_(nullptr) {
    offloaded_mem_pool_ = std::make_shared<device::OffloadedMemPool>();
  }
  virtual ~DeviceResManager() = default;

  // Initialize the device resource manager.
  virtual void Initialize() {}

  // Destroy device resource manager and release device resource.
  virtual void Destroy() {}

  // Bind device to current thread to gain device control privileges
  // If force_bind is true, bind context to current thread every time;
  // Otherwise, only bind context to current thread for the first time.
  virtual bool BindDeviceToCurrentThread(bool force_bind) const { return true; }
  virtual void ResetStreamAndCtx() {}

  // Relevant function to allocate and free device memory of raw ptr.
  virtual void *AllocateMemory(size_t size, uint32_t stream_id = kDefaultStreamIndex) const = 0;
  virtual void FreeMemory(void *ptr) const = 0;
  virtual void FreePartMemorys(const std::vector<void *> &free_addrs, const std::vector<void *> &keep_addrs,
                               const std::vector<size_t> &keep_addr_sizes) const = 0;

  virtual void SwapIn(const void *host_ptr, void *device_ptr, size_t mem_size, void *stream) {
    MS_LOG(EXCEPTION) << "Unimplemented interface.";
    return;
  }
  virtual void SwapOut(const void *device_ptr, void *host_ptr, size_t mem_size, void *stream) {
    MS_LOG(EXCEPTION) << "Unimplemented interface.";
    return;
  }

  // Relevant function to allocate and free device memory of DeviceAddress.
  virtual bool AllocateMemory(DeviceAddress *const &address, uint32_t stream_id = UINT32_MAX) const;
  virtual void FreeMemory(DeviceAddress *const &address) const;
  virtual size_t GetMaxUsedMemorySize() const { return 0; }

  // Relevant function to manage memory statistics
  virtual size_t GetTotalMemStatistics() const { return 0; }
  virtual size_t GetTotalUsedMemStatistics() const { return 0; }
  virtual size_t GetTotalIdleMemStatistics() const { return 0; }
  virtual size_t GetTotalEagerFreeMemStatistics() const { return 0; }
  virtual size_t GetUsedMemPeakStatistics() const { return 0; }
  virtual size_t GetReservedMemPeakStatistics() const { return 0; }
  virtual std::unordered_map<std::string, std::size_t> GetBlockCountsStatistics() const { return {}; }
  virtual std::unordered_map<std::string, std::size_t> GetBlockUnitSizeStatistics() const { return {}; }
  virtual std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
  GetCommonMemBlocksInfoStatistics() const {
    return {};
  }
  virtual std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
  GetPersistentMemBlocksInfoStatistics() const {
    return {};
  }
  virtual void ResetMaxMemoryReserved() const {};
  virtual void ResetMaxMemoryAllocated() const {};

  // Allocate host memory with raii and ref count
  virtual std::shared_ptr<void> AllocateHostMemory(size_t size) const {
    return std::shared_ptr<void>(::malloc(size), ::free);
  }
  // Allocate host memory for offload device memory.
  virtual void *AllocateOffloadMemory(size_t size) const;
  // Release host memory which was allocated by AllocateOffloadMemory to pool.
  // It will not be free to os.
  virtual void FreeOffloadMemory(void *ptr) const;

  virtual size_t GetAvailableMemSize() const { return 0; }

  // Allocate continuous device memory according to size list.
  // Communication operators may need continuous memory for input and output
  // to optimize the communication performance.
  virtual std::vector<void *> AllocateContinuousMemory(const std::vector<size_t> &size_list,
                                                       uint32_t stream_id = kDefaultStreamIndex) const {
    MS_LOG(EXCEPTION) << "Unimplemented interface.";
  }

  // Create concrete device address according different device type using KernelTensor.
  virtual DeviceAddressPtr CreateDeviceAddress(const KernelTensorPtr &kernel_tensor) const {
    MS_LOG(EXCEPTION) << "Unimplemented interface.";
  }
  virtual void MoveTo(const tensor::TensorPtr &src_tensor, const tensor::TensorPtr &dst_tensor, const std::string &to,
                      bool blocking, bool *return_self) {
    MS_LOG(EXCEPTION) << "Unimplemented interface.";
  }

  virtual DeviceAddressPtr CreateDeviceAddress(void *ptr, size_t size, const ShapeVector &shape_vector,
                                               const Format &format, TypeId type_id, const std::string &device_name,
                                               uint32_t device_id, uint32_t stream_id) const {
    MS_LOG(EXCEPTION) << "Unimplemented interface.";
  }

  // Create a stream with assigning a stream id, the assigned stream id will be written to the parameter '*stream_id'.
  virtual bool CreateStream(size_t *stream_id) const {
    MS_LOG(WARNING) << "Unimplemented interface: 'CreateStream'.";
    *stream_id = kSizeZero;
    return false;
  }

  // Create a stream with priority.
  virtual bool CreateStreamWithPriority(size_t *stream_id, int32_t priority) const {
    *stream_id = kSizeZero;
    return false;
  }

  virtual size_t QueryStreamSize() const { return 0L; }
  virtual std::vector<uint32_t> GetStreamIds() const { return {}; }

  // If multi-stream used in pynative mode, other streams must be sync before the graph
  // is executed. Otherwise, out-of-order occurs. Therefore this flag is added.
  // This solution is a temporary solution, this flag will be removed after multi-stream is
  // supported in graph mode.
  virtual bool single_op_multi_stream_enable() const { return false; }
  virtual void set_single_op_multi_stream_enable(bool single_op_multi_stream_enable) {}

  // Get the stream pointer by stream_id.
  virtual void *GetStream(size_t stream_id) const { return nullptr; };

  // Set currently using stream id.
  virtual void SetCurrentStreamId(size_t stream_id) { return; }

  // Get currently using stream id.
  virtual size_t GetCurrentStreamId() const { return kSizeZero; }

  virtual void *GetStream() const { return nullptr; };

  // Destroy a stream bound to the input parameter "stream_id".
  virtual bool DestroyStream(size_t stream_id) const { return false; }

  // Query tasks' completion status of a stream.
  virtual bool QueryStream(size_t stream_id) const { return true; }

  // Synchronize stream, device such as GPU and Ascend need stream to launch kernel asynchronously,
  // Using 'SyncStream' to block thread and wait for completing all tasks on specific stream.
  // Using 'SyncAllStream' to block thread and wait for completing all tasks on all streams.
  // Devices without stream could ignore the implementation of these function.
  // Since the current entry for creating streams is not unified, the implementation of the 'SyncStream' and
  // "SyncAllStreams" interfaces are implemented by subclasses.
  virtual bool SyncStream(size_t stream_id) const { return true; }

  virtual bool SyncAllStreams() const { return true; }

  virtual bool SyncNotDefaultStreams() const { return true; }

  // Return default stream id. Normally it's 0.
  virtual size_t DefaultStream() const { return 0; }

  // Create device event for runtime.
  virtual DeviceEventPtr CreateRuntimeEvent(bool enable_blocking, bool enable_record_wait) { return nullptr; }

  // Create device event with flag.
  virtual DeviceEventPtr CreateEventWithFlag(bool enable_timing, bool blocking) { return nullptr; };

  // Destroy specified device event.
  virtual bool DestroyEvent(const DeviceEventPtr &event);

  // Destroy all device events.
  virtual bool DestroyAllEvents();

  // Dynamically load collective communication library.
  // Currently, four types are supported: OpenMPI and self developed framework for CPU. NCCL for GPU. HCCL for Ascend.
  virtual bool LoadCollectiveCommLib() { return true; }

  // Return collective communication object for caller to access
  CollectiveCommunicationLib *collective_comm_lib() const { return collective_comm_lib_; }

  std::shared_ptr<SwapManager> swap_manager() const { return swap_manager_; }

  std::shared_ptr<MemoryManager> mem_manager() const { return mem_manager_; }

 protected:
  // Ensure the thread safety for allocating device memory.
  mutable std::mutex alloc_mem_mutex_;

  // The collective communication library.
  CollectiveCommunicationLib *collective_comm_lib_;

  DeviceContext *device_context_{nullptr};

  std::shared_ptr<SwapManager> swap_manager_{nullptr};

  DeviceEventPtrList device_events_{};

  std::shared_ptr<MemoryManager> mem_manager_{nullptr};

 private:
  template <class... Args>
  friend class DeviceInterface;
  void SetDeviceContext(DeviceContext *device_context) { device_context_ = device_context; }
  std::shared_ptr<device::OffloadedMemPool> offloaded_mem_pool_;
};

class GraphExecutor {
 public:
  virtual ~GraphExecutor() = default;
  virtual bool CompileGraph(const FuncGraphPtr &graph, const std::map<string, string> &compile_options) { return true; }
  virtual bool RunGraph(const FuncGraphPtr &graph, const std::vector<tensor::Tensor> &inputs,
                        std::vector<tensor::Tensor> *outputs, const std::map<string, string> &compile_options) {
    MS_LOG(EXCEPTION) << "Unimplemented interface.";
  }
  virtual std::string GetRandomStatus(const std::vector<FuncGraphPtr> &graphs) { return ""; }
  virtual size_t GetGraphFeatureMemory(const FuncGraphPtr &graph) const { return 0; }
  virtual void InitGraphInfo(const FuncGraphPtr &graph) { return; };

 protected:
  DeviceContext *device_context_{nullptr};

 private:
  template <class... Args>
  friend class DeviceInterface;

  void SetDeviceContext(DeviceContext *device_context) { device_context_ = device_context; }
};

using CallbackFunc = std::function<void(void)>;

class BACKEND_EXPORT KernelExecutor {
 public:
  virtual ~KernelExecutor() = default;

  virtual void Initialize(){};
  virtual void Destroy(){};

  // Optimize the kernel graph for graph mode.
  virtual void OptimizeGraph(const FuncGraphPtr &graph) const {}

  // Generate 'KernelMod' for all kernels and set 'KernelMod' into kernel,
  // 'KernelMod' is real executive object of kernel.
  virtual void CreateKernel(const std::vector<CNodePtr> &nodes) const {}
  virtual kernel::KernelModPtr CreateKernelMod(const std::string &op_name) const { MS_LOG(EXCEPTION) << "Unrealized"; };

  // Adjust kernel graph before run graph.
  virtual void PreprocessBeforeRun(const FuncGraphPtr &graph) const {}

  // Launch a kernel via 'KernelMod' of the kernel, use KernelTensor input type.
  virtual bool LaunchKernel(const CNodePtr &kernel, const std::vector<KernelTensor *> &inputs,
                            const std::vector<KernelTensor *> &workspace, const std::vector<KernelTensor *> &outputs,
                            KernelMod *kernel_mod, void *stream) const {
    MS_LOG(EXCEPTION) << "Unimplemented interface.";
  }
  // Launch callback.
  virtual bool LaunchCallback(std::function<void(void)> callback_func, size_t stream_id) const {
    callback_func();
    return true;
  };
  // Unify the MindIR, the default behavior uses the common unified MindIR.
  virtual void UnifyMindIR(const KernelGraphPtr &graph) const;
  virtual void AddMindIRPass(const KernelGraphPtr &graph) const {};

  // Get rank id for distributed training.
  virtual uint32_t GetRankID() const { return 0; }

  void SetDeviceContext(DeviceContext *device_context) { device_context_ = device_context; }

  virtual bool ExecuteKernelTask(const runtime::KernelTaskType &task_type,
                                 const device::DeviceAddressPtrList &input_addr_list,
                                 const device::DeviceAddressPtrList &output_addr_list, const size_t &stream_id) const {
    return false;
  };

 protected:
  DeviceContext *device_context_{nullptr};
};

template <class... Args>
class DeviceInterface : public DeviceContext {};

template <>
class DeviceInterface<> : public DeviceContext {
 public:
  explicit DeviceInterface(const DeviceContextKey &key) : DeviceContext(key) {}

 protected:
  void CheckUnset(const void *ptr, const std::string &error_msg) const {
    if (ptr != nullptr) {
      MS_LOG(EXCEPTION) << error_msg;
    }
  }
};

template <class T, class... Args>
class DeviceInterface<T, Args...> : public DeviceInterface<Args...> {
 public:
  explicit DeviceInterface(const DeviceContextKey &key) : DeviceInterface<Args...>(key) {
    if constexpr (std::is_base_of_v<DeviceResManager, T>) {
      DeviceInterface::CheckUnset(reinterpret_cast<void *>(DeviceContext::device_res_manager_.get()),
                                  "DeviceResManager has been registered!");
      DeviceContext::device_res_manager_ = std::make_unique<T>();
      DeviceContext::device_res_manager_->SetDeviceContext(this);
    } else if constexpr (std::is_base_of_v<GraphExecutor, T>) {
      DeviceInterface::CheckUnset(reinterpret_cast<void *>(DeviceContext::graph_executor_.get()),
                                  "GraphExecutor has been registered!");
      DeviceContext::graph_executor_ = std::make_unique<T>();
      DeviceContext::graph_executor_->SetDeviceContext(this);
    } else if constexpr (std::is_base_of_v<KernelExecutor, T>) {
      DeviceInterface::CheckUnset(reinterpret_cast<void *>(DeviceContext::GetKernelExecutor(false).get()),
                                  "KernelExecutor has been registered!");
      DeviceInterface::CheckUnset(reinterpret_cast<void *>(DeviceContext::GetKernelExecutor(true).get()),
                                  "Dyn KernelExecutor has been registered!");
      DeviceContext::SetKernelExecutor(std::make_shared<T>());
      DeviceContext::GetKernelExecutor(false)->SetDeviceContext(this);
      // for GPU/CPU dynamic shape kernel executor
      DeviceContext::SetDynKernelExecutor(DeviceContext::GetKernelExecutor(false));
    }
  }

 private:
  template <typename = std::enable_if_t<std::is_base_of_v<DeviceResManager, T> || std::is_base_of_v<GraphExecutor, T> ||
                                        std::is_base_of_v<KernelExecutor, T>>>
  void Assert() const {}
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_DEVICE_CONTEXT_H_
