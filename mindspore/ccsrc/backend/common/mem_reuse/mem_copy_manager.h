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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MEM_REUSE_MEM_COPY_MANAGER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MEM_REUSE_MEM_COPY_MANAGER_H_

#include <vector>
#include <map>
#include <set>
#include <queue>
#include <memory>
#include <utility>
#include "include/backend/kernel_graph.h"
#include "kernel/kernel.h"
#include "include/backend/device_address.h"

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
  bool trigger_swap_out_{false};
  bool trigger_swap_in_{false};
  size_t swap_in_task_num_{0};
  // Key: output index, value: topo orders of node users
  std::map<size_t, std::vector<size_t>> node_users_map_;
  // Key: output index, value: pair (host addr, dirty or not)
  std::map<size_t, std::pair<HostAddress, bool>> host_addrs_;

  KernelExecutionInfo() {}
  explicit KernelExecutionInfo(size_t topo_order) : KernelExecutionInfo(topo_order, 0.0, false, false, 0) {}
  KernelExecutionInfo(size_t topo_order, float execution_perform, bool trigger_swap_out, bool trigger_swap_in,
                      size_t swap_in_task_num)
      : topo_order_(topo_order),
        execution_perform_(execution_perform),
        trigger_swap_out_(trigger_swap_out),
        trigger_swap_in_(trigger_swap_in),
        swap_in_task_num_(swap_in_task_num) {}
};

struct MemSwapInfo {
  SwapKind swap_kind_;
  // Topo order of kernel need be swapped
  size_t topo_order_;
  size_t output_idx_{0};
  // Record the swapping out position of swapping in tensor
  size_t swap_out_pos_;
};

struct SwapInfoComp {
  bool operator()(const MemSwapInfo &a, const MemSwapInfo &b) const {
    int swap_kind_a = static_cast<int>(a.swap_kind_);
    int swap_kind_b = static_cast<int>(b.swap_kind_);
    if (swap_kind_a < swap_kind_b) {
      return true;
    } else if (swap_kind_a > swap_kind_b) {
      return false;
    }

    if (a.swap_out_pos_ < b.swap_out_pos_) {
      return true;
    } else if (a.swap_out_pos_ > b.swap_out_pos_) {
      return false;
    }

    if (a.topo_order_ < b.topo_order_) {
      return true;
    } else if (a.topo_order_ > b.topo_order_) {
      return false;
    }

    return a.output_idx_ < b.output_idx_;
  }
};

class MemCopyManager {
 public:
  MemCopyManager() = default;

  virtual ~MemCopyManager() = default;

  virtual void Init() {}

  virtual void AddMemSwapOutTask(const DeviceAddressPtr &device_address, const HostAddress &host_addr) {}

  virtual void AddMemSwapInTask(const DeviceAddressPtr &device_address, const HostAddress &host_addr, bool profiling,
                                float *cost_time) {}

  virtual void AddMemSwapOutTaskMock(const DeviceAddressPtr &device_address) {}

  virtual void AddMemSwapInTaskMock(const DeviceAddressPtr &device_address) {}

  virtual bool SyncMemCopyStream(SwapKind swap_kind) { return true; }

  virtual DeviceAddressPtr UpdateSwapOutQueue() { return nullptr; }

  virtual DeviceAddressPtr UpdateSwapInQueue() { return nullptr; }

  virtual DeviceAddressPtr UpdateSwapOutQueueMock() { return nullptr; }

  virtual DeviceAddressPtr UpdateSwapInQueueMock() { return nullptr; }

  virtual bool AllocHostPinnedMem(size_t size, void **addr) const { return true; }

  virtual void FreeHostPinnedMem(void *addr) const {}

  virtual void ClearSwapQueue() {}

  virtual void ClearSwapQueueMock() {}
};
using MemCopyManagerPtr = std::shared_ptr<MemCopyManager>;
using MemSwapInfoSet = std::set<MemSwapInfo, SwapInfoComp>;
}  // namespace memswap
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MEM_REUSE_MEM_COPY_MANAGER_H_
