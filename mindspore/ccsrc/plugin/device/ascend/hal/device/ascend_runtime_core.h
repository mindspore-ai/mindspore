/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_DEVICE_ASCEND_RUNTIME_CORE_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_DEVICE_ASCEND_RUNTIME_CORE_H_
#include <dirent.h>
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include "utils/ms_context.h"
#include "runtime/context.h"
#include "plugin/device/ascend/hal/device/ge_runtime/davinci_model.h"
#include "plugin/device/ascend/hal/device/tasksink/rtmodel_zero_copy.h"
#include "plugin/device/ascend/hal/device/ascend_runtime_manager.h"

#ifndef ENABLE_SECURITY
#include "plugin/device/ascend/hal/device/dump/data_dumper.h"
#endif

namespace mindspore::device::ascend {
using ge::model_runner::TaskInfo;
class AscendRuntimeCore : public AscendKernelRuntime {
 public:
  AscendRuntimeCore() = default;
  ~AscendRuntimeCore() override;
  void InitCore() override;
  bool GenTask(const session::KernelGraph &graph);
  void GenKernelEventsCore(const session::KernelGraph &graph) override;
  void GetLastNodesOnStream(const std::vector<CNodePtr> &kernels, std::vector<size_t> *stream_last_nodes) const;
  void ProcessBoundaryEvent(const std::vector<CNodePtr> &kernels,
                            std::map<AnfNodePtr, std::vector<std::function<void()>>> *kernel_run_events,
                            const std::vector<size_t> &last_stream_nodes);
  bool LoadTask(const session::KernelGraph &graph);
  bool RunTaskCore(const session::KernelGraph &graph) override;
  bool LoadCore(const session::KernelGraph &graph, bool is_task_sink) override;
  void *GetModelStreamCore(uint32_t graph_id) const override;
  bool LoadDataCore() override;
  void UnloadModelCore(uint32_t graph_id = UINT32_MAX) override;
  void RegTaskFailCallback(const bool &is_release = false) override;
  bool CheckAndUnloadModelInAdvance(uint32_t model_id) override;

 private:
#ifndef ENABLE_SECURITY
  static void PrintDebugInfoAndDumpFailNode(const session::KernelGraph &graph);
  static void DumpDebugInfoFile(const session::KernelGraph &graph);
  void DistributeDebugTask(const session::KernelGraph &graph, const NotNull<std::function<void *()>> &model_handle);
  void LaunchDataDump(GraphId graph_id);
  std::unordered_map<GraphId, std::shared_ptr<DataDumper>> graph_data_dumper_;
#endif
  bool GraphWithEmptyTaskList(const session::KernelGraph &graph) const;
  bool CheckGraphIdValid(GraphId graph_id) const;

  static void TaskFailCallback(rtExceptionInfo *task_fail_info);
  static std::pair<CNodePtr, std::string> GetErrorNodeInfo(uint32_t streamid, uint32_t taskid);
  static std::string GetDumpPath(const std::string &suffix);
  static bool DeleteDumpDir(const std::string &path);
  static std::string GetRealPath(const std::string &path);
  static int DeleteDumpFile(std::string path);

  static std::vector<rtExceptionInfo> task_fail_infos_;
  static std::map<std::string, uint32_t> overflow_tasks_;
  tasksink::RtModelZeroCopy rt_model_zero_copy_;
  std::unordered_map<GraphId, vector<std::shared_ptr<TaskInfo>>> task_map_;
  std::unordered_map<GraphId, std::shared_ptr<ge::model_runner::DavinciModel>> graph_model_map_;
};

REG_ASCEND_RUNTIME(kAscendVM, AscendRuntimeCore);
}  // namespace mindspore::device::ascend

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_DEVICE_ASCEND_RUNTIME_CORE_H_
