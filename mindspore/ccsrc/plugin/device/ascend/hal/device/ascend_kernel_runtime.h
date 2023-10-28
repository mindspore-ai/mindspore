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
#include "runtime/context.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "backend/common/session/session_basic.h"
#ifndef ENABLE_SECURITY
#include "plugin/device/ascend/hal/device/dump/data_dumper.h"
#endif

using std::unordered_map;
using std::vector;
namespace mindspore::device::ascend {
class AscendKernelRuntime : public KernelRuntime {
 public:
  AscendKernelRuntime() = default;
  ~AscendKernelRuntime() override;
  bool Init() override;
  virtual void InitCore() {}
  bool LoadData(const session::KernelGraph &graph) override;
  virtual bool LoadDataCore() { return true; }
  void GenKernelEvents(const session::KernelGraph &graph) override;
  virtual void GenKernelEventsCore(const session::KernelGraph &graph) {}
  bool RunDynamicKernelAsync(const session::KernelGraph &graph) override;
  bool RunTask(const session::KernelGraph &graph);
  virtual bool RunTaskCore(const session::KernelGraph &graph) { return true; }
  bool Load(const session::KernelGraph &graph, bool is_task_sink) override;
  virtual bool LoadCore(const session::KernelGraph &graph, bool is_task_sink) { return true; }
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
  void *GetModelStream(uint32_t graph_id) const override;
  virtual void *GetModelStreamCore(uint32_t graph_id) const { return nullptr; }
  void *GetKernelStream(const AnfNodePtr &kernel) const override;
  // add for MindRT
  void ReleaseDeviceRes() override;
  uint64_t GetMsUsedHbmSize() const;
  void SetReuseCommunicationAddress(const session::KernelGraph &graph);
  void SetRtDevice(uint32_t device_id);
  virtual void UnloadModelCore(uint32_t graph_id = UINT32_MAX) {}
  virtual void RegTaskFailCallback(const bool &is_release = false) {}
  virtual bool CheckAndUnloadModelInAdvance(uint32_t model_id) { return true; }

 protected:
  DeviceAddressPtr CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format,
                                       TypeId type_id) const override;
  DeviceAddressPtr CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format, TypeId type_id,
                                       const KernelWithIndex &node_index) const override;
  bool KernelMemNotReuse(const AnfNodePtr &node) override;

  void KernelLaunchProfiling(const std::string &kernel_name) override;
  inline static const session::KernelGraph *current_graph_ = nullptr;

 private:
  bool InitDevice();
  bool ResetDevice(uint32_t device_id);
  static bool NeedDestroyHccl();
  static bool DestroyHccl();
  void ClearGraphModelMap();
  void CreateDefaultStream();

  rtContext_t rt_context_{nullptr};
  bool initialized_{false};
  std::map<std::pair<uint32_t, uint32_t>, std::string> stream_id_task_id_op_name_map_;
  std::set<uint32_t> initialized_device_set_{};
  AscendKernelRuntime *runtime_core_{nullptr};
};

MS_REG_KERNEL_RUNTIME(kAscendDevice, AscendKernelRuntime);
}  // namespace mindspore::device::ascend
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_KERNEL_RUNTIME_H_
