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
#include "debug/rdr/running_data_recorder.h"
#include <utility>

#include "debug/rdr/graph_exec_order_recorder.h"
#include "debug/rdr/recorder_manager.h"
#include "debug/rdr/string_recorder.h"
#include "debug/rdr/stream_exec_order_recorder.h"
#include "debug/rdr/mem_address_recorder.h"
#include "mindspore/core/ir/func_graph.h"
#include "mindspore/core/ir/anf.h"
#include "backend/kernel_compiler/kernel.h"
#ifdef ENABLE_D
#include "runtime/device/ascend/tasksink/task_generator.h"
#include "debug/rdr/task_debug_info_recorder.h"
#endif  // ENABLE_D
namespace mindspore {
namespace RDR {
#ifdef ENABLE_D
bool RecordTaskDebugInfo(SubModuleId module, const std::string &name,
                         const std::vector<TaskDebugInfoPtr> &task_debug_info_list) {
  if (!mindspore::RecorderManager::Instance().RdrEnable()) {
    return false;
  }
  std::string submodule_name = std::string(GetSubModuleName(module));
  TaskDebugInfoRecorderPtr task_debug_info_recorder =
    std::make_shared<TaskDebugInfoRecorder>(submodule_name, name, task_debug_info_list);
  bool ans = mindspore::RecorderManager::Instance().RecordObject(std::move(task_debug_info_recorder));
  return ans;
}
#endif  // ENABLE_D

bool RecordAnfGraph(const SubModuleId module, const std::string &name, const FuncGraphPtr &graph,
                    const DumpGraphParams &info, const std::string &file_type) {
  if (!mindspore::RecorderManager::Instance().RdrEnable()) {
    return false;
  }
  std::string submodule_name = std::string(GetSubModuleName(module));
  GraphRecorderPtr graph_recorder = std::make_shared<GraphRecorder>(submodule_name, name, graph, file_type);
  graph_recorder->SetDumpFlag(info);
  bool ans = mindspore::RecorderManager::Instance().RecordObject(std::move(graph_recorder));
  return ans;
}

bool RecordGraphExecOrder(const SubModuleId module, const std::string &name,
                          const std::vector<CNodePtr> &final_exec_order) {
  if (!mindspore::RecorderManager::Instance().RdrEnable()) {
    return false;
  }
  std::string submodule_name = std::string(GetSubModuleName(module));
  GraphExecOrderRecorderPtr graph_exec_order_recorder =
    std::make_shared<GraphExecOrderRecorder>(submodule_name, name, final_exec_order);
  bool ans = mindspore::RecorderManager::Instance().RecordObject(std::move(graph_exec_order_recorder));
  return ans;
}

bool RecordString(SubModuleId module, const std::string &name, const std::string &data) {
  if (!mindspore::RecorderManager::Instance().RdrEnable()) {
    return false;
  }
  std::string submodule_name = std::string(GetSubModuleName(module));
  StringRecorderPtr string_recorder = std::make_shared<StringRecorder>(submodule_name, name, data);
  bool ans = mindspore::RecorderManager::Instance().RecordObject(std::move(string_recorder));
  return ans;
}

bool RecordStreamExecOrder(const SubModuleId module, const std::string &name, const std::vector<CNodePtr> &exec_order) {
  if (!mindspore::RecorderManager::Instance().RdrEnable()) {
    return false;
  }
  std::string submodule_name = std::string(GetSubModuleName(module));
  StreamExecOrderRecorderPtr stream_exec_order_recorder =
    std::make_shared<StreamExecOrderRecorder>(submodule_name, name, exec_order);
  bool ans = mindspore::RecorderManager::Instance().RecordObject(std::move(stream_exec_order_recorder));
  return ans;
}

bool RecordMemAddressInfo(const SubModuleId module, const std::string &name) {
  if (!mindspore::RecorderManager::Instance().RdrEnable()) {
    return false;
  }
  std::string submodule_name = std::string(GetSubModuleName(module));
  MemAddressRecorderPtr mem_info_recorder = std::make_shared<MemAddressRecorder>(submodule_name, name);
  mem_info_recorder->Reset();
  bool ans = mindspore::RecorderManager::Instance().RecordObject(std::move(mem_info_recorder));
  return ans;
}

bool UpdateMemAddress(const SubModuleId module, const std::string &name, const std::string &op_name,
                      const kernel::KernelLaunchInfo &mem_info) {
  if (!mindspore::RecorderManager::Instance().RdrEnable()) {
    return false;
  }
  std::string submodule_name = std::string(GetSubModuleName(module));
  auto recorder = mindspore::RecorderManager::Instance().GetRecorder(submodule_name, name);
  bool ans = false;
  if (recorder != nullptr) {
    auto mem_recorder = std::dynamic_pointer_cast<MemAddressRecorder>(recorder);
    mem_recorder->SaveMemInfo(op_name, mem_info);
    ans = true;
  }
  return ans;
}

void TriggerAll() { mindspore::RecorderManager::Instance().TriggerAll(); }

void ResetRecorder() { mindspore::RecorderManager::Instance().ClearAll(); }

void ClearMemAddressInfo() {
  if (!mindspore::RecorderManager::Instance().RdrEnable()) {
    return;
  }
  if (RecorderManager::Instance().CheckRdrMemIsRecord()) {
    std::string name = "mem_address_list";
    std::string submodule_name = "KERNEL";
    auto recorder = RecorderManager::Instance().GetRecorder(submodule_name, name);
    if (recorder != nullptr) {
      auto mem_recorder = std::dynamic_pointer_cast<MemAddressRecorder>(recorder);
      mem_recorder->CleanUp();
    }
  }
}
}  // namespace RDR
}  // namespace mindspore
