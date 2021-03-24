/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include <string>
#include <vector>
#include <memory>
#include "runtime/device/device_address.h"
#include "backend/session/kernel_graph.h"
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace device {
using mindspore::kernel::AddressPtr;
using mindspore::kernel::KernelMod;

struct DeviceContextKey {
  // device type name, such as 'GPU' 'Ascend' 'CPU'.
  std::string device_name_;
  uint32_t device_id_{0};

  // Use the result of ToString() as key to look up DeviceContext
  // in cache map which maintains created DeviceContext objects.
  std::string ToString() const { return device_name_ + "_" + std::to_string(device_id_); }
};

// DeviceContext is unified interface of interaction with device.
class DeviceContext {
 public:
  explicit DeviceContext(const DeviceContextKey &device_context_key) : device_context_key_(device_context_key) {}
  virtual ~DeviceContext() = default;

  // Initialize the device context and return success or not.
  virtual bool Initialize() = 0;

  // Destroy device context and release device resource.
  virtual void Destroy() {}

  // Relevant function to allocate and free device memory.
  virtual bool AllocateMemory(DeviceAddress *const &address, size_t size) const = 0;
  virtual void FreeMemory(DeviceAddress *const &address) const = 0;

  // Allocate continuous device memory end to end into 'addr_list'.
  // Communication operators may need continuous memory for input and output
  // to optimize the communication performance.
  virtual bool AllocateContinuousMemory(const std::vector<DeviceAddress *> &addr_list, size_t total_size,
                                        const std::vector<size_t> &size_list) const {
    return true;
  }

  // The two functions below will be merged to one in the future.
  // General graph optimezer ignore device data type and format.
  virtual void OptimizeGraphWithoutDeviceInfo(const KernelGraphPtr &graph) const {}
  // Optimize the kernel graph according to device data type and format.
  virtual void OptimizeGraphWithDeviceInfo(const KernelGraphPtr &graph) const {}

  // Optimize the single operator graph for PyNative mode.
  virtual void OptimizeSingleOpGraph(const KernelGraphPtr &graph) const {}

  // Select the matching backend kernels according to the data type and format of input and output for all
  // execution operators, and set final device data type and format information for backend kernels, device
  // data type and format which replace original data type and format will use for executing kernels.
  virtual void SetOperatorInfo(const std::vector<CNodePtr> &nodes) const = 0;

  // Generate 'KernelMod' for all kernels and set 'KernelMod' into kernel,
  // 'KernelMod' is real executive object of kernel.
  virtual void CreateKernel(const std::vector<CNodePtr> &nodes) const = 0;

  // Launch a kernel via 'KernelMod' of the kernel.
  virtual bool LaunchKernel(KernelMod *kernel_mod, const std::vector<AddressPtr> &inputs,
                            const std::vector<AddressPtr> &workspace, const std::vector<AddressPtr> &outputs) const = 0;

  // Synchronize stream, device such as GPU and Ascend need stream to launch kernel asynchronously,
  // using 'SyncStream' to block thread and wait for completing all tasks in stream.
  // Devices that do not need stream could ignore the implementation of this function.
  virtual bool SyncStream(size_t stream_id = 0) { return true; }

 protected:
  DeviceContextKey device_context_key_;
};
using DeviceContextPtr = std::shared_ptr<DeviceContext>;
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_DEVICE_CONTEXT_H_
