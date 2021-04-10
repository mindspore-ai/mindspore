/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include "runtime/device/kernel_runtime.h"
#include "runtime/context.h"
#include "framework/ge_runtime/davinci_model.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "backend/session/session_basic.h"
#include "runtime/device/ascend/dump/data_dumper.h"

using ge::model_runner::TaskInfo;
using std::unordered_map;
using std::vector;
namespace mindspore::device::ascend {
class AscendKernelRuntime : public KernelRuntime {
 public:
  AscendKernelRuntime() = default;
  ~AscendKernelRuntime() override;
  bool Init() override;
  bool LoadData(session::KernelGraph *graph) override;
  bool GenTask(const session::KernelGraph *graph);
  bool GenDynamicKernel(const session::KernelGraph *graph) override;
  bool RunDynamicKernelAsync(const session::KernelGraph *graph) override;
  bool LoadTask(const session::KernelGraph *graph);
  bool RunTask(const session::KernelGraph *graph);
  bool Load(session::KernelGraph *graph, bool is_task_sink) override;
  bool Run(session::KernelGraph *const graph, bool is_task_sink) override;
  void ClearGraphRuntimeResource(uint32_t graph_id, const std::vector<AnfNodePtr> &inputs,
                                 const std::unordered_set<ValueNodePtr> &value_nodes,
                                 const std::vector<CNodePtr> &execution_order) override;
  void ClearGlobalIdleMem() override;
  bool SyncStream() override;
  bool MemcpyAsync(void *dst, const void *src, uint64_t size, int32_t kind) override;
  void SetContext() override;
  void CreateContext() override;
  const void *context() const override { return rt_context_; }
  void PreInit() override;
  uint64_t GetAvailableMemMaxSize() const override;
  DeviceAddressType GetTargetDeviceAddressType() const override { return DeviceAddressType::kAscend; };
  void *compute_stream() const override { return stream_; }
  void *communication_stream() const override { return communication_stream_; }

 protected:
  DeviceAddressPtr CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format,
                                       TypeId type_id) override;
  bool KernelMemNotReuse(const AnfNodePtr &node) override;

  void KernelLaunchProfiling(const std::string &kernel_name) override;

 private:
  bool InitDevice();
  bool ResetDevice(uint32_t device_id);
  static bool HcclInit();
  static bool NeedDestroyHccl();
  static bool DestroyHccl();
  static bool DestroySingleOpHccl();
  void InnerSetContext();

  void ClearGraphModelMap();
  void ReleaseDeviceRes() override;
  bool GraphWithEmptyTaskList(const session::KernelGraph *graph) const;
  bool CheckGraphIdValid(GraphId graph_id) const;
  void DistributeDebugTask(NotNull<const session::KernelGraph *> graph,
                           const NotNull<std::function<void *()>> &model_handle);
  void LaunchDataDump(GraphId graph_id);
  static CNodePtr GetErrorNodeName(uint32_t streamid, uint32_t taskid);
  static void DumpTaskExceptionInfo(const session::KernelGraph *graph);
  static void TaskFailCallback(rtExceptionInfo *task_fail_info);
  void ReportProfilingData();

  rtContext_t rt_context_{nullptr};
  bool initialized_{false};
  unordered_map<GraphId, vector<std::shared_ptr<TaskInfo>>> task_map_;
  unordered_map<GraphId, std::shared_ptr<ge::model_runner::DavinciModel>> graph_model_map_;
  unordered_map<GraphId, std::shared_ptr<DataDumper>> graph_data_dumper_;
  std::map<std::pair<uint32_t, uint32_t>, std::string> stream_id_task_id_op_name_map_;
  static std::map<std::string, uint32_t> overflow_tasks_;
  static std::vector<rtExceptionInfo> task_fail_infoes_;
};

MS_REG_KERNEL_RUNTIME(kAscendDevice, AscendKernelRuntime);
}  // namespace mindspore::device::ascend
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_KERNEL_RUNTIME_H_
