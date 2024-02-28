/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_KERNEL_RUNTIME_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_KERNEL_RUNTIME_H_
#include <dirent.h>
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include "runtime/device/kernel_runtime.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "backend/common/session/session_basic.h"
#include "acl/acl_rt.h"

using std::unordered_map;
using std::vector;
namespace mindspore::device::ascend {
class AscendKernelRuntime : public KernelRuntime {
 public:
  AscendKernelRuntime() = default;
  ~AscendKernelRuntime() override;
  bool Init() override;
  bool RunDynamicKernelAsync(const session::KernelGraph &graph) override;
  bool RunTask(const session::KernelGraph &graph);
  bool Run(const session::KernelGraph &graph, bool is_task_sink) override;
  void ClearGraphRuntimeResource(uint32_t graph_id) override;
  void ClearGlobalIdleMem() override;
  bool SyncStream() override;
  bool MemcpyAsync(void *dst, const void *src, uint64_t size, int32_t kind) override;
  void SetContext() override;
  void SetContextForce() override;
  void ResetStreamAndCtx() override;
  const void *context() const override { return rt_context_; }
  DeviceAddressPtr GetInternalDeviceAddress(const session::KernelGraph &graph, const AnfNodePtr &node) override;
  void GetShadowBackendNodeMap(const session::KernelGraph &graph,
                               std::map<AnfNodePtr, AnfNodePtr> *shadow_backend_node_map) override;
#ifndef ENABLE_SECURITY
  void PreInit() override;
#endif
  DeviceType GetTargetDeviceType() const override { return DeviceType::kAscend; };
  std::shared_ptr<DeviceEvent> CreateDeviceEvent() override;
  std::shared_ptr<DeviceEvent> CreateDeviceTimeEvent() override;
  void *compute_stream() const override { return stream_; }
  void *communication_stream() const override { return communication_stream_; }
  void *GetKernelStream(const AnfNodePtr &kernel) const override;
  // add for MindRT
  void ReleaseDeviceRes() override;
  uint64_t GetMsUsedHbmSize() const override;
  void SetRtDevice(uint32_t device_id);

 protected:
  static void TaskExceptionCallback(aclrtExceptionInfo *task_fail_info);
  DeviceAddressPtr CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format,
                                       TypeId type_id) const override;
  DeviceAddressPtr CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format, TypeId type_id,
                                       const KernelWithIndex &node_index) const override;
  bool KernelMemNotReuse(const AnfNodePtr &node) override;
  inline static const session::KernelGraph *current_graph_ = nullptr;

 private:
  bool InitDevice();
  bool ResetDevice(uint32_t device_id);
  static bool NeedDestroyHccl();
  static bool DestroyHccl();
  void ClearGraphModelMap();
  void CreateDefaultStream();

  aclrtContext rt_context_{nullptr};
  bool initialized_{false};
  std::set<uint32_t> initialized_device_set_{};
};

MS_REG_KERNEL_RUNTIME(kAscendDevice, AscendKernelRuntime);
}  // namespace mindspore::device::ascend
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_KERNEL_RUNTIME_H_
