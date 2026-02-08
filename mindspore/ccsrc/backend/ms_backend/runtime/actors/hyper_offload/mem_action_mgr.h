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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_HYPER_OFFLOAD_ACTION_MGR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_HYPER_OFFLOAD_ACTION_MGR_H_

#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "backend/ms_backend/runtime/actors/hyper_offload/hyper_offload_utils.h"
#include "runtime/hardware_abstract/event/device_event.h"

namespace mindspore {
namespace runtime {

class MemActionMgr {
 public:
  MemActionMgr() {}
  ~MemActionMgr() { ClearMAMInstance(); }

  KernelTensorPtrPairList CreateRemoteEvents(const RemoteActionPtrList &remote_events,
                                             const device::DeviceContext *device_context);

  bool IsKernelTensorCanBeMoved(const KernelTensorPtr &kernel_tensor);

  void ClearMAMInstance() {
    auto size = kernel_events_map.size();
    if (size != 0) {
      MS_VLOG(VL_HYPER_OFFLOAD_WARNING) << "Some kernel tensors did not insert wait event: " << size;
    }
    kernel_events_map.clear();
  }

 private:
  // Create CopyData task for kernel_tensor
  kernel::KernelTensorPtr InsertCopyDataTask(uint32_t src_stream_id, const device::DeviceContext *device_context,
                                             HyperOffloadEventType event_type,
                                             const kernel::KernelTensorPtr &kernel_tensor);

  // Check the corresponding event for kernel_tensor, if not found, return false
  // Insert Wait event for kernel_tensor on stream_id, erase this kernel_tensor's event info
  bool InsertWaitWithMemoryEvent(uint32_t stream_id, const device::DeviceContext *device_context,
                                 const kernel::KernelTensorPtr &kernel_tensor);

  // Insert Record event for kernel_tensor on stream_id, using kernel_tensor as key, manage this event
  bool InsertRecordWithMemoryEvent(uint32_t stream_id, const device::DeviceContext *device_context,
                                   const kernel::KernelTensorPtr &kernel_tensor);

  bool InsertRecordWaitEvents(uint32_t src_stream_id, uint32_t dst_stream_id,
                              const device::DeviceContext *device_context);

  std::unordered_map<kernel::KernelTensorPtr, DeviceEventPtr> kernel_events_map;
};

}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_HYPER_OFFLOAD_ACTION_MGR_H_
