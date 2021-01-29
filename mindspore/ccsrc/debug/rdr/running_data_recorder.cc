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
#include "mindspore/core/ir/func_graph.h"
#include "mindspore/core/ir/anf.h"

namespace mindspore {
namespace {
static const char *GetSubModuleName(SubModuleId module_id) {
  static const char *sub_module_names[NUM_SUBMODUES] = {
    "UNKNOWN",    // SM_UNKNOWN
    "CORE",       // SM_CORE
    "ANALYZER",   // SM_ANALYZER
    "COMMON",     // SM_COMMON
    "DEBUG",      // SM_DEBUG
    "DEVICE",     // SM_DEVICE
    "GE_ADPT",    // SM_GE_ADPT
    "IR",         // SM_IR
    "KERNEL",     // SM_KERNEL
    "MD",         // SM_MD
    "ME",         // SM_ME
    "EXPRESS",    // SM_EXPRESS
    "OPTIMIZER",  // SM_OPTIMIZER
    "PARALLEL",   // SM_PARALLEL
    "PARSER",     // SM_PARSER
    "PIPELINE",   // SM_PIPELINE
    "PRE_ACT",    // SM_PRE_ACT
    "PYNATIVE",   // SM_PYNATIVE
    "SESSION",    // SM_SESSION
    "UTILS",      // SM_UTILS
    "VM",         // SM_VM
    "PROFILER",   // SM_PROFILER
    "PS",         // SM_PS
    "LITE",       // SM_LITE
    "HCCL_ADPT"   // SM_HCCL_ADPT
  };

  return sub_module_names[module_id % NUM_SUBMODUES];
}
}  // namespace
namespace RDR {
#ifdef __linux__
bool RecordAnfGraph(const SubModuleId module, const std::string &tag, const FuncGraphPtr &graph, bool full_name,
                    const std::string &file_type, int graph_id) {
  std::string submodule_name = std::string(GetSubModuleName(module));
  GraphRecorderPtr graph_recorder = std::make_shared<GraphRecorder>(submodule_name, tag, graph, file_type, graph_id);
  graph_recorder->SetDumpFlag(full_name);
  bool ans = mindspore::RecorderManager::Instance().RecordObject(std::move(graph_recorder));
  return ans;
}

bool RecordGraphExecOrder(const SubModuleId module, const std::string &tag,
                          const std::vector<CNodePtr> &&final_exec_order) {
  std::string submodule_name = std::string(GetSubModuleName(module));
  GraphExecOrderRecorderPtr graph_exec_order_recorder =
    std::make_shared<GraphExecOrderRecorder>(submodule_name, tag, final_exec_order);
  bool ans = mindspore::RecorderManager::Instance().RecordObject(std::move(graph_exec_order_recorder));
  return ans;
}

bool RecordString(SubModuleId module, const std::string &tag, const std::string &data, const std::string &filename) {
  std::string submodule_name = std::string(GetSubModuleName(module));
  StringRecorderPtr string_recorder = std::make_shared<StringRecorder>(submodule_name, tag, data, filename);
  string_recorder->SetFilename(filename);
  bool ans = mindspore::RecorderManager::Instance().RecordObject(std::move(string_recorder));
  return ans;
}

void TriggerAll() { mindspore::RecorderManager::Instance().TriggerAll(); }

#else
bool RecordAnfGraph(const SubModuleId module, const std::string &tag, const FuncGraphPtr &graph, bool full_name,
                    const std::string &file_type, int graph_id) {
  static bool already_printed = false;
  std::string submodule_name = std::string(GetSubModuleName(module));
  if (already_printed) {
    return false;
  }
  already_printed = true;
  MS_LOG(WARNING) << "The RDR presently only support linux os " << submodule_name;
  return false;
}

bool RecordGraphExecOrder(const SubModuleId module, const std::string &tag, std::vector<CNodePtr> &&final_exec_order) {
  static bool already_printed = false;
  if (already_printed) {
    return false;
  }
  already_printed = true;
  MS_LOG(WARNING) << "The RDR presently only support linux os.";
  return false;
}

bool RecordString(SubModuleId module, const std::string &tag, const std::string &data, const std::string &filename) {
  static bool already_printed = false;
  if (already_printed) {
    return false;
  }
  already_printed = true;
  MS_LOG(WARNING) << "The RDR presently only support linux os.";
  return false;
}

void TriggerAll() {
  static bool already_printed = false;
  if (already_printed) {
    return;
  }
  already_printed = true;
  MS_LOG(WARNING) << "The RDR presently only support linux os.";
}
#endif  // __linux__
}  // namespace RDR
}  // namespace mindspore
