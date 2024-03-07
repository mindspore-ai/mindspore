/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "runtime/graph_scheduler/actor/debug_actor.h"
#include <vector>
#include <memory>
#include <string>
#include "runtime/graph_scheduler/actor/debug_aware_actor.h"
#include "mindrt/include/async/async.h"
#include "utils/log_adapter.h"
#ifndef ENABLE_SECURITY
#include "debug/data_dump/cpu_e2e_dump.h"
#include "include/backend/debug/data_dump/e2e_dump.h"
#include "utils/ms_context.h"
#endif
#ifdef ENABLE_DEBUGGER
#include "include/backend/debug/debugger/debugger.h"
#include "debug/debugger/debugger_utils.h"
#endif
#include "debug/data_dump/data_dumper.h"
#include "include/common/debug/common.h"
#include "utils/file_utils.h"
#include "include/backend/debug/profiler/profiling.h"

namespace mindspore {
namespace runtime {
void DebugActor::ACLDump(uint32_t device_id, const std::vector<KernelGraphPtr> &graphs) {
  std::string env_enable_str = common::GetEnv("MS_ACL_DUMP_CFG_PATH");
  std::string kbk_enable_str = common::GetEnv("GRAPH_OP_RUN");
  auto step_count_num = 0;
  step_count_num = step_count;
  if (step_count == 1 && is_dataset_sink == 1) {
    step_count_num = 0;
  }
  if (!graphs.empty()) {
    auto graph = graphs[0];
    is_dataset_sink = graph->IsDatasetGraph();
  }
  if (DumpJsonParser::GetInstance().async_dump_enabled() &&
      ((DumpJsonParser::GetInstance().IsDumpIter(step_count_num) && kbk_enable_str == "1") || env_enable_str == "1")) {
    bool is_init = false;
    if ((env_enable_str == "1") && !(DumpJsonParser::GetInstance().IsDumpIter(step_count_num))) {
      is_init = true;
    } else {
      std::string dump_path = DumpJsonParser::GetInstance().path();
      std::string dump_path_step = dump_path + "/" + std::to_string(step_count_num);
      auto real_path = FileUtils::CreateNotExistDirs(dump_path_step, false);
      if (!real_path.has_value()) {
        MS_LOG(WARNING) << "Fail to create acl dump dir " << real_path.value();
        return;
      }
    }
    dump_flag = true;
    auto registered_dumper = datadump::DataDumperRegister::Instance().GetDumperForBackend(device::DeviceType::kAscend);
    if (registered_dumper != nullptr) {
      registered_dumper->Initialize();
      registered_dumper->EnableDump(device_id, step_count_num, is_init);
    }
  }
}
/*
 * Feature group: Dump, Online debugger.
 * Target device group: GPU.
 * Runtime category: MindRT.
 * Description: Load and read data for the given node if needed. Dump the node if dump is enabled and free the loaded
 * memory after the dump (for GPU and ascend kernel-by-kernel).
 */
void DebugActor::Debug(const AnfNodePtr &node, const KernelLaunchAddr *launch_info, const DeviceContext *device_context,
                       OpContext<DeviceTensor> *const op_context, const AID *) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(op_context);
  std::lock_guard<std::mutex> locker(debug_mutex_);

  if (!node->isa<CNode>()) {
    return;
  }
  const auto &cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(DEBUG) << "kernel by kernel debug for node: " << cnode->fullname_with_scope() << ".";
  if (device_context->GetDeviceType() == device::DeviceType::kAscend) {
    MS_LOG(DEBUG) << "On ascend, this funct is not need now.";
    return;
  } else if (device_context->GetDeviceType() == device::DeviceType::kCPU) {
#ifndef ENABLE_SECURITY
    if (DumpJsonParser::GetInstance().GetIterDumpFlag()) {
      auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(cnode->func_graph());
      MS_EXCEPTION_IF_NULL(kernel_graph);
      CPUE2eDump::DumpCNodeData(cnode, kernel_graph->graph_id());
      CPUE2eDump::DumpRunIter(kernel_graph);
    }
#endif
  } else if (device_context->GetDeviceType() == device::DeviceType::kGPU) {
#ifdef ENABLE_DEBUGGER
    auto debugger = Debugger::GetInstance();
    if (debugger != nullptr) {
      auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(cnode->func_graph());
      debugger->InsertExecutedGraph(kernel_graph);
      std::string kernel_name = cnode->fullname_with_scope();
      debugger->SetCurNode(kernel_name);
      bool read_data = CheckReadData(cnode);
      if (read_data) {
        ReadDataAndDump(cnode, launch_info, exec_order_, device_context);
      }
    }
    exec_order_ += 1;
#endif
  }
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: Ascend.
 * Runtime category: MindRT.
 * Description: Load data for online debugger and dump graph for e2e dump mode (Ascend super kernel mode).
 */
void DebugActor::DebugForGraph(const KernelGraphPtr &graph, const DeviceContext *device_context,
                               OpContext<DeviceTensor> *const op_context, const AID *) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(op_context);
  std::lock_guard<std::mutex> locker(debug_mutex_);

  MS_LOG(DEBUG) << "Super kernel debug for graph: " << graph->graph_id() << ".";
#ifdef ENABLE_DEBUGGER
  auto debugger = Debugger::GetInstance();
  if (debugger != nullptr) {
    debugger->InsertExecutedGraph(graph);
  }
  LoadDataForDebugger(graph);
  // This function updates graph history file and cur_dump_iter if dump is enabled.
  // When e2e dump is enabled, this function dumps the graph.
  SuperKernelE2eDump(graph);

#endif
}

/*
 * Feature group: ascend step start timestamp
 * Target device group: Ascend.
 * Description: Add step start timestamp when profiler is started.
 */
void DebugActor::AscendStepStart(const std::vector<KernelGraphPtr> &graphs,
                                 std::vector<DeviceContext *> device_contexts) {
  auto profiler = profiler::Profiler::GetInstance(kAscendDevice);
  if (profiler == nullptr || !profiler->IsInitialized() || graphs.empty()) {
    return;
  }
  if (profiler->GetEnableFlag() && !graphs[0]->IsDatasetGraph()) {
    profile_started_ = false;
    for (size_t i = 0; i < graphs.size(); ++i) {
      MS_EXCEPTION_IF_NULL(graphs[i]);
      MS_EXCEPTION_IF_NULL(device_contexts[i]);
      if (device_contexts[i]->GetDeviceType() == device::DeviceType::kAscend && !profile_started_) {
        device_ctx_ = device_contexts[i];
        device_ctx_->device_res_manager_->BindDeviceToCurrentThread(false);
        MS_LOG(INFO) << "Dot step start timestamp.";
        profiler->StepStart(current_step++, device_contexts[i]->device_res_manager_->GetStream());
        profile_started_ = true;
      }
    }
  }
}

/*
 * Feature group: ascend step end timestamp
 * Target device group: Ascend.
 * Description: Add step end timestamp when profiler is end.
 */
void DebugActor::AscendStepEnd() {
  auto profiler = profiler::Profiler::GetInstance(kAscendDevice);
  if (profile_started_ && profiler != nullptr && profiler->GetEnableFlag()) {
    MS_EXCEPTION_IF_NULL(device_ctx_);
    device_ctx_->device_res_manager_->BindDeviceToCurrentThread(false);
    device_ctx_->device_res_manager_->SyncAllStreams();
    MS_LOG(INFO) << "Dot step end timestamp.";
    profiler->StepStop();
    profile_started_ = false;
  }
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: MindRT.
 * Description: Checks dataset_sink_mode and generates the related error if any exist and calls PreExecuteGraphDebugger.
 */
void DebugActor::DebugOnStepBegin(const std::vector<KernelGraphPtr> &graphs,
                                  const std::vector<AnfNodePtr> &origin_parameters_order,
                                  std::vector<DeviceContext *> device_contexts,
                                  OpContext<DeviceTensor> *const op_context, const AID *) {
  MS_LOG(INFO) << "Debug on step begin.";
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  std::string backend = context->backend_policy();
  device_ctx_ = device_contexts[0];
  auto profiler = profiler::Profiler::GetInstance(kAscendDevice);
  if ((profiler == nullptr || !profiler->IsInitialized()) &&
      device_ctx_->GetDeviceType() == device::DeviceType::kAscend) {
    auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    if (common::GetEnv("MS_ACL_DUMP_CFG_PATH") == "1" || common::GetEnv("GRAPH_OP_RUN") == "1") {
      ACLDump(device_id, graphs);
    }
  }
  if (backend == "ge") {
    AscendStepStart(graphs, device_contexts);
    return;
  }
  MS_EXCEPTION_IF_NULL(op_context);
  std::lock_guard<std::mutex> locker(debug_mutex_);
#ifdef ENABLE_DEBUGGER
  if (!graphs.empty()) {
    // First graph is the dataset graph when dataset_sink_mode = True
    auto graph = graphs[0];
    std::string error_info = CheckDatasetSinkMode(graph);
    if (!error_info.empty()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*op_context), error_info);
    }
  }
  auto debugger = Debugger::GetInstance();
  if (debugger != nullptr && debugger->DebuggerBackendEnabled()) {
    debugger->PreExecuteGraphDebugger(graphs, origin_parameters_order);
  }
#endif
#ifndef ENABLE_SECURITY
  if (DumpJsonParser::GetInstance().e2e_dump_enabled()) {
    DumpJsonParser::GetInstance().ClearGraph();
    if (graphs.size() != device_contexts.size()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*op_context), "Graph num:" + std::to_string(graphs.size()) +
                                                         " is not equal to device context size:" +
                                                         std::to_string(device_contexts.size()) + " for debug actor.");
    }
    for (size_t i = 0; i < graphs.size(); ++i) {
      MS_EXCEPTION_IF_NULL(graphs[i]);
      MS_EXCEPTION_IF_NULL(device_contexts[i]);
      if (device_contexts[i]->GetDeviceType() == device::DeviceType::kCPU) {
        DumpJsonParser::GetInstance().SaveGraph(graphs[i].get());
      }
    }
  }
#endif
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: Ascend, GPU and CPU.
 * Runtime category: MindRT.
 * Description: Dump parameters and constants and update dump iter for CPU. Call PostExecuteGraph Debugger for GPU and
 * Ascend and update step number of online debugger GPU.
 */
void DebugActor::DebugOnStepEnd(OpContext<DeviceTensor> *const op_context, const AID *, int total_running_count_) {
  MS_LOG(INFO) << "Debug on step begin.";
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  std::string backend = context->backend_policy();
  step_count = total_running_count_;
  if (dump_flag == true) {
    auto registered_dumper = datadump::DataDumperRegister::Instance().GetDumperForBackend(device::DeviceType::kAscend);
    if (registered_dumper != nullptr) {
      device_ctx_->device_res_manager_->SyncAllStreams();
      registered_dumper->Finalize();
    }
    dump_flag = false;
  }
  if (backend == "ge") {
    AscendStepEnd();
    MS_LOG(INFO) << "On GE backend, debug_actor is not supported.";
    return;
  }
  MS_EXCEPTION_IF_NULL(op_context);
  std::lock_guard<std::mutex> locker(debug_mutex_);

#ifndef ENABLE_SECURITY
  if (DumpJsonParser::GetInstance().GetIterDumpFlag()) {
    CPUE2eDump::DumpParametersData();
    CPUE2eDump::DumpConstantsData();
  }
#endif

#ifdef ENABLE_DEBUGGER
  auto debugger = Debugger::GetInstance();
  if (debugger != nullptr) {
    // Reset exec_order for the next step
    exec_order_ = 0;
    debugger->Debugger::PostExecuteGraphDebugger();
    debugger->Debugger::UpdateStepNumGPU();
  }
#else
#ifndef ENABLE_SECURITY
  DumpJsonParser::GetInstance().UpdateDumpIter();
#endif
#endif
}
}  // namespace runtime
}  // namespace mindspore
