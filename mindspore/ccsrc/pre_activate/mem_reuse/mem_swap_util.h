/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PRE_ACTIVATE_MEM_REUSE_MEM_SWAP_UTIL_H_
#define MINDSPORE_CCSRC_PRE_ACTIVATE_MEM_REUSE_MEM_SWAP_UTIL_H_

#include <vector>
#include <map>
#include <queue>
#include <memory>
#include <utility>
#include "session/kernel_graph.h"
#include "device/gpu/cuda_driver.h"
#include "kernel/kernel.h"

using mindspore::device::gpu::DeviceEvent;
using mindspore::device::gpu::DeviceMemPtr;
using mindspore::device::gpu::DeviceStream;
using mindspore::device::gpu::HostMemPtr;
using HostAddress = mindspore::kernel::Address;
namespace mindspore {
namespace device {
namespace memswap {
enum class SwapKind { kDeviceToHost = 0, kHostToDevice = 1 };

struct TensorInfo {
  size_t tensor_size_{0};
  AnfNodePtr kernel_{nullptr};
  size_t output_idx_{0};
};

struct KernelExecutionInfo {
  size_t topo_order_{0};
  float execution_perform_{0.0};
  bool trigger_swap_{false};
  bool need_swap_{false};
  // output index to topo orders of node users
  std::map<size_t, std::vector<size_t>> node_users_map_;
  // kernel output idx to host addr
  std::map<size_t, HostAddress> host_addrs_;

  KernelExecutionInfo() : KernelExecutionInfo(0, 0.0, false, false) {}
  explicit KernelExecutionInfo(size_t topo_order)
      : topo_order_(topo_order), execution_perform_(0.0), trigger_swap_(false), need_swap_(false) {}
  KernelExecutionInfo(size_t topo_order, float execution_perform, bool trigger_swap, bool need_swap)
      : topo_order_(topo_order),
        execution_perform_(execution_perform),
        trigger_swap_(trigger_swap),
        need_swap_(need_swap) {}
};

// trigger swap
struct MemSwapInfo {
  SwapKind swap_kind_;
  // kernel need to be swapped
  AnfNodePtr kernel_{nullptr};
  size_t output_idx_{0};
};

class MemCopyManager {
 public:
  MemCopyManager() = default;

  ~MemCopyManager() = default;

  void Init();

  void AddMemSwapOutTask(const DeviceAddressPtr &device_address, const HostAddress &host_addr);

  void AddMemSwapInTask(const DeviceAddressPtr &device_address, const HostAddress &host_addr);

  bool SyncMemCopyStream(SwapKind swap_kind);

  DeviceAddressPtr UpdateSwapOutQueue();

  DeviceAddressPtr UpdateSwapInQueue();

  void ClearSwapQueue();

 private:
  DeviceStream swap_out_stream_{nullptr};
  DeviceStream swap_in_stream_{nullptr};
  std::queue<std::pair<DeviceAddressPtr, DeviceEvent>> swap_out_queue_;
  std::queue<std::pair<DeviceAddressPtr, DeviceEvent>> swap_in_queue_;
};
using MemCopyManagerPtr = std::shared_ptr<MemCopyManager>;
}  // namespace memswap
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PRE_ACTIVATE_MEM_REUSE_MEM_SWAP_UTIL_H_
