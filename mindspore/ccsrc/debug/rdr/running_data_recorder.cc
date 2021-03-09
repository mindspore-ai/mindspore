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
#include "debug/rdr/graph_recorder.h"
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
namespace {
static const char *GetSubModuleName(SubModuleId module_id) {
  static const char *sub_module_names[NUM_SUBMODUES] = {
    "UNKNOWN",     // SM_UNKNOWN
    "CORE",        // SM_CORE
    "ANALYZER",    // SM_ANALYZER
    "COMMON",      // SM_COMMON
    "DEBUG",       // SM_DEBUG
    "DEVICE",      // SM_DEVICE
    "GE_ADPT",     // SM_GE_ADPT
    "IR",          // SM_IR
    "KERNEL",      // SM_KERNEL
    "MD",          // SM_MD
    "ME",          // SM_ME
    "EXPRESS",     // SM_EXPRESS
    "OPTIMIZER",   // SM_OPTIMIZER
    "PARALLEL",    // SM_PARALLEL
    "PARSER",      // SM_PARSER
    "PIPELINE",    // SM_PIPELINE
    "PRE_ACT",     // SM_PRE_ACT
    "PYNATIVE",    // SM_PYNATIVE
    "SESSION",     // SM_SESSION
    "UTILS",       // SM_UTILS
    "VM",          // SM_VM
    "PROFILER",    // SM_PROFILER
    "PS",          // SM_PS
    "LITE",        // SM_LITE
    "HCCL_ADPT",   // SM_HCCL_ADPT
    "MINDQUANTUM"  // SM_MINDQUANTUM
  };

  return sub_module_names[module_id % NUM_SUBMODUES];
}
}  // namespace
namespace RDR {
#ifdef ENABLE_D
bool RecordTaskDebugInfo(SubModuleId module, const std::string &tag,
                         const std::vector<TaskDebugInfoPtr> &task_debug_info_list, int graph_id) {
  if (!mindspore::RecorderManager::Instance().RdrEnable()) {
    return false;
  }
  std::string submodule_name = std::string(GetSubModuleName(module));
  TaskDebugInfoRecorderPtr task_debug_info_recorder =
    std::make_shared<TaskDebugInfoRecorder>(submodule_name, tag, task_debug_info_list, graph_id);
  bool ans = mindspore::RecorderManager::Instance().RecordObject(std::move(task_debug_info_recorder));
  return ans;
}
#endif  // ENABLE_D

bool RecordAnfGraph(const SubModuleId module, const std::string &tag, const FuncGraphPtr &graph, bool full_name,
                    const std::string &file_type) {
  if (!mindspore::RecorderManager::Instance().RdrEnable()) {
    return false;
  }
  std::string submodule_name = std::string(GetSubModuleName(module));
  GraphRecorderPtr graph_recorder = std::make_shared<GraphRecorder>(submodule_name, tag, graph, file_type);
  graph_recorder->SetDumpFlag(full_name);
  bool ans = mindspore::RecorderManager::Instance().RecordObject(std::move(graph_recorder));
  return ans;
}

bool RecordGraphExecOrder(const SubModuleId module, const std::string &tag,
                          const std::vector<CNodePtr> &final_exec_order, int graph_id) {
  if (!mindspore::RecorderManager::Instance().RdrEnable()) {
    return false;
  }
  std::string submodule_name = std::string(GetSubModuleName(module));
  GraphExecOrderRecorderPtr graph_exec_order_recorder =
    std::make_shared<GraphExecOrderRecorder>(submodule_name, tag, final_exec_order, graph_id);
  bool ans = mindspore::RecorderManager::Instance().RecordObject(std::move(graph_exec_order_recorder));
  return ans;
}

bool RecordString(SubModuleId module, const std::string &tag, const std::string &data, const std::string &filename) {
  if (!mindspore::RecorderManager::Instance().RdrEnable()) {
    return false;
  }
  std::string submodule_name = std::string(GetSubModuleName(module));
  StringRecorderPtr string_recorder = std::make_shared<StringRecorder>(submodule_name, tag, data, filename);
  string_recorder->SetFilename(filename);
  bool ans = mindspore::RecorderManager::Instance().RecordObject(std::move(string_recorder));
  return ans;
}

bool RecordStreamExecOrder(const SubModuleId module, const std::string &tag, const int &graph_id,
                           const std::vector<CNodePtr> &exec_order) {
  if (!mindspore::RecorderManager::Instance().RdrEnable()) {
    return false;
  }
  std::string submodule_name = std::string(GetSubModuleName(module));
  StreamExecOrderRecorderPtr stream_exec_order_recorder =
    std::make_shared<StreamExecOrderRecorder>(submodule_name, tag, graph_id, exec_order);
  bool ans = mindspore::RecorderManager::Instance().RecordObject(std::move(stream_exec_order_recorder));
  return ans;
}

bool RecordMemAddressInfo(const SubModuleId module, const std::string &tag, const std::string &op_name,
                          const GPUMemInfo &mem_info) {
  if (!mindspore::RecorderManager::Instance().RdrEnable()) {
    return false;
  }
  std::string submodule_name = std::string(GetSubModuleName(module));
  std::string directory = mindspore::EnvConfigParser::GetInstance().rdr_path();
  MemAddressRecorder::Instance().SetModule(submodule_name);
  MemAddressRecorder::Instance().SetFilename(tag);  // set filename using tag
  MemAddressRecorder::Instance().SetDirectory(directory);
  MemAddressRecorder::Instance().SaveMemInfo(op_name, mem_info);
  return true;
}
void TriggerAll() {
  mindspore::RecorderManager::Instance().TriggerAll();
  MemAddressRecorder::Instance().Export();
}

void ClearAll() { mindspore::RecorderManager::Instance().ClearAll(); }
}  // namespace RDR
}  // namespace mindspore
