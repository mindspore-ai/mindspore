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
#include "plugin/device/ascend/hal/device/ge_runtime/davinci_model.h"
#include "plugin/device/ascend/hal/device/tasksink/rtmodel_zero_copy.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "backend/common/session/session_basic.h"
#ifndef ENABLE_SECURITY
#include "plugin/device/ascend/hal/device/dump/data_dumper.h"
#endif

using std::unordered_map;
using std::vector;
namespace mindspore::device::ascend {
using ge::model_runner::TaskInfo;
class AscendKernelRuntime : public KernelRuntime {
 public:
  AscendKernelRuntime() = default;
  ~AscendKernelRuntime() override;
  bool Init() override;
  bool LoadData(const session::KernelGraph &graph) override;
  bool GenTask(const session::KernelGraph &graph);
  void GenKernelEvents(const session::KernelGraph &graph) override;
  void GetLastNodesOnStream(const std::vector<CNodePtr> &kernels, std::vector<size_t> *stream_last_nodes) const;
  void ProcessBoundaryEvent(const std::vector<CNodePtr> &kernels,
                            std::map<AnfNodePtr, std::vector<std::function<void()>>> *kernel_run_events,
                            const std::vector<size_t> &last_stream_nodes);
  bool RunDynamicKernelAsync(const session::KernelGraph &graph) override;
  bool LoadTask(const session::KernelGraph &graph);
  bool RunTask(const session::KernelGraph &graph);
  bool Load(const session::KernelGraph &graph, bool is_task_sink) override;
  bool Run(const session::KernelGraph &graph, bool is_task_sink) override;
  void ClearGraphRuntimeResource(uint32_t graph_id) override;
  void ClearGlobalIdleMem() override;
  bool SyncStream() override;
  bool MemcpyAsync(void *dst, const void *src, uint64_t size, int32_t kind) override;
  void SetContext() override;
  void CreateContext() override;
  const void *context() const override { return rt_context_; }
  DeviceAddressPtr GetInternalDeviceAddress(const session::KernelGraph &graph, const AnfNodePtr &node) override;
  void GetShadowBackendNodeMap(const session::KernelGraph &graph,
                               std::map<AnfNodePtr, AnfNodePtr> *shadow_backend_node_map) override;
#ifndef ENABLE_SECURITY
  void PreInit() override;
#endif
  uint64_t GetAvailableMemMaxSize() const override;
  DeviceType GetTargetDeviceType() const override { return DeviceType::kAscend; };
  std::shared_ptr<DeviceEvent> CreateDeviceEvent() override;
  std::shared_ptr<DeviceEvent> CreateDeviceTimeEvent() override;
  void *compute_stream() const override { return stream_; }
  void *communication_stream() const override { return communication_stream_; }
  void *GetModelStream(uint32_t graph_id) const override;
  void *GetKernelStream(const AnfNodePtr &kernel) const override;
  // add for MindRT
  void ReleaseDeviceRes() override;
  uint64_t GetMsUsedHbmSize() const;
  void SetReuseCommunicationAddress(const session::KernelGraph &graph);
  void SetRtDevice(uint32_t device_id);

 protected:
  DeviceAddressPtr CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format,
                                       TypeId type_id) const override;
  DeviceAddressPtr CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format, TypeId type_id,
                                       const KernelWithIndex &node_index) const override;
  bool KernelMemNotReuse(const AnfNodePtr &node) override;

  void KernelLaunchProfiling(const std::string &kernel_name) override;

 private:
  bool InitDevice();
  bool ResetDevice(uint32_t device_id);
  static bool NeedDestroyHccl();
  static bool DestroyHccl();
  void SetCurrentContext();

  void ClearGraphModelMap();
  bool GraphWithEmptyTaskList(const session::KernelGraph &graph) const;
  bool CheckGraphIdValid(GraphId graph_id) const;
#ifndef ENABLE_SECURITY
  void DistributeDebugTask(const session::KernelGraph &graph, const NotNull<std::function<void *()>> &model_handle);
  void LaunchDataDump(GraphId graph_id);
#endif
  static CNodePtr GetErrorNodeName(uint32_t streamid, uint32_t taskid);
  static std::string GetDumpPath();
#ifndef ENABLE_SECURITY
  static void DumpTaskExceptionInfo(const session::KernelGraph &graph);
#endif
  static void TaskFailCallback(rtExceptionInfo *task_fail_info);
  static bool DeleteDumpDir(const std::string &path);
  static int DeleteDumpFile(std::string path);
  static std::string GetRealPath(const std::string &path);
  void CreateDefaultStream();

  rtContext_t rt_context_{nullptr};
  bool initialized_{false};
  unordered_map<GraphId, vector<std::shared_ptr<TaskInfo>>> task_map_;
  unordered_map<GraphId, std::shared_ptr<ge::model_runner::DavinciModel>> graph_model_map_;
#ifndef ENABLE_SECURITY
  unordered_map<GraphId, std::shared_ptr<DataDumper>> graph_data_dumper_;
#endif
  std::map<std::pair<uint32_t, uint32_t>, std::string> stream_id_task_id_op_name_map_;
  static std::map<std::string, uint32_t> overflow_tasks_;
  static std::vector<rtExceptionInfo> task_fail_infoes_;
  std::set<uint32_t> initialized_device_set_{};
  tasksink::RtModelZeroCopy rt_model_zero_copy_;
};

MS_REG_KERNEL_RUNTIME(kAscendDevice, AscendKernelRuntime);
}  // namespace mindspore::device::ascend
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_KERNEL_RUNTIME_H_
