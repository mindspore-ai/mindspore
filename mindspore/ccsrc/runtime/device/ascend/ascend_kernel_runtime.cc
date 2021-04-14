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
#define PATH_MAX 4096
#include "runtime/device/ascend/ascend_kernel_runtime.h"
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <algorithm>
#include "runtime/device/ascend/signal_util.h"
#include "debug/data_dump/e2e_dump.h"
#include "runtime/device/ascend/ascend_device_address.h"
#include "utils/ms_context.h"
#include "utils/context/context_extends.h"
#include "utils/mpi/mpi_config.h"
#include "runtime/device/ascend/profiling/profiling_manager.h"
#include "common/trans.h"
#include "runtime/context.h"
#include "runtime/device/ascend/ascend_stream_assign.h"
#include "framework/ge_runtime/model_runner.h"
#include "runtime/device/ascend/tasksink/task_generator.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "runtime/device/ascend/profiling/profiling_utils.h"
#include "runtime/device/ascend/ascend_memory_manager.h"
#include "debug/data_dump/dump_json_parser.h"
#include "toolchain/adx_datadump_server.h"
#include "utils/trace_base.h"
#include "graphengine/inc/external/acl/error_codes/rt_error_codes.h"
#include "utils/runtime_error_codes.h"
#include "debug/anf_ir_dump.h"
#ifdef MEM_REUSE_DEBUG
#include "backend/optimizer/mem_reuse/mem_reuse_checker.h"
#include "debug/env_config_parser.h"
#endif
#include "runtime/device/ascend/executor/tiling/op_tiling_calculater.h"
#include "runtime/device/ascend/executor/hccl_dynamic_kernel.h"
#include "utils/config_manager.h"
#include "runtime/device/ascend/profiling/reporter/op_name_task_stream_reporter.h"
#include "runtime/hccl_adapter/hccl_adapter.h"
#include "runtime/device/ascend/profiling/profiling_callback_register.h"
#include "backend/kernel_compiler/hccl/hccl_context.h"
#ifdef ENABLE_TDTQUE
#include "minddata/dataset/engine/tdt/tdt_handle.h"
using mindspore::dataset::TdtHandle;
#endif
#ifdef ENABLE_DUMP_IR
#include "debug/rdr/running_data_recorder.h"
#endif

using ge::model_runner::ModelRunner;
using mindspore::device::ascend::ProfilingManager;
using mindspore::device::ascend::ProfilingUtils;
using mindspore::device::ascend::tasksink::TaskGenerator;
using mindspore::kernel::tbe::TbeUtils;
using std::vector;

constexpr uint32_t kTupleTaskId = 0;
constexpr uint32_t kTupleStreamId = 1;
constexpr uint32_t kTupleArgs = 2;
constexpr uint32_t kProfilingMaxTaskIdInStream = 65531;
constexpr auto kModuleName = "MindSpore";

namespace mindspore::device::ascend {
static thread_local rtContext_t thread_local_rt_context{nullptr};
namespace {
std::string GetRankId() {
  std::string rank_id_str;
#ifdef ENABLE_MPI
  auto mpi_config_ptr = MpiConfig::GetInstance();
  MS_EXCEPTION_IF_NULL(mpi_config_ptr);
  if (mpi_config_ptr->enable_mpi()) {
    int rank_id = GetMPIRankId();
    const char *offset = std::getenv("RANK_OFFSET");
    if (offset != nullptr) {
      try {
        int rank_offset = std::stoi(offset);
        rank_id += rank_offset;
      } catch (std::invalid_argument) {
        MS_LOG(EXCEPTION) << "Call stoi invalid argument:" << offset;
      } catch (std::out_of_range) {
        MS_LOG(EXCEPTION) << "Call stoi out_of_range:" << offset;
      }
    }
    rank_id_str = std::to_string(rank_id);
  } else {
    rank_id_str = std::getenv("RANK_ID");
  }
#else
  rank_id_str = std::getenv("RANK_ID");
#endif
  if (rank_id_str.empty()) {
    MS_LOG(ERROR) << "Get hccl rankid failed, please set env RANK_ID";
  }
  return rank_id_str;
}
}  // namespace

std::vector<rtExceptionInfo> AscendKernelRuntime::task_fail_infoes_ = {};
const session::KernelGraph *current_graph_ = nullptr;
std::map<std::string, uint32_t> AscendKernelRuntime::overflow_tasks_;
AscendKernelRuntime::~AscendKernelRuntime() {
  graph_model_map_.clear();
  current_graph_ = nullptr;
}

void AscendKernelRuntime::SetContext() {
  if (rt_context_ == nullptr) {
    return;
  }
  if (thread_local_rt_context == rt_context_) {
    return;
  }
  auto ret = rtCtxSetCurrent(rt_context_);
  thread_local_rt_context = rt_context_;
  if (ret != RT_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "Call rtCtxSetCurrent, ret[" << ret << "]";
  }
}

void AscendKernelRuntime::InnerSetContext() {
  if (rt_context_ == nullptr) {
    return;
  }
  auto ret = rtCtxSetCurrent(rt_context_);
  if (ret != RT_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "Call rtCtxSetCurrent, ret[" << ret << "]";
  }
}

void AscendKernelRuntime::ClearGraphModelMap() {
  InnerSetContext();
  for (auto &iter : graph_data_dumper_) {
    MS_LOG(INFO) << "[DataDump] Unload data dumper:" << iter.first;
    auto &data_dumper = iter.second;
    MS_EXCEPTION_IF_NULL(data_dumper);
    data_dumper->UnloadDumpInfo();
    data_dumper->OpDebugUnregister();
  }
  graph_data_dumper_.clear();
  // tell users which dump kernel name not used
  DumpJsonParser::GetInstance().PrintUnusedKernel();

  graph_dynamic_kernel_map_.clear();

  for (auto &iter : graph_model_map_) {
    MS_LOG(INFO) << "Ge UnloadModel " << iter.first;
    auto ret = ModelRunner::Instance().UnloadModel(iter.first);
    if (!ret) {
      MS_LOG(ERROR) << "UnloadModel failed";
    }
  }
}

void AscendKernelRuntime::ClearGraphRuntimeResource(uint32_t graph_id, const std::vector<AnfNodePtr> &,
                                                    const std::unordered_set<ValueNodePtr> &,
                                                    const std::vector<CNodePtr> &) {
  InnerSetContext();
  MS_LOG(DEBUG) << "Clear graph:" << graph_id << " data dumper";
  if (auto dumper_iter = graph_data_dumper_.find(graph_id); dumper_iter != graph_data_dumper_.end()) {
    MS_LOG(DEBUG) << "Unload dump info " << graph_id;
    auto &data_dumper = dumper_iter->second;
    MS_EXCEPTION_IF_NULL(data_dumper);
    data_dumper->UnloadDumpInfo();
    data_dumper->OpDebugUnregister();
    graph_data_dumper_.erase(dumper_iter);
  } else {
    MS_LOG(DEBUG) << "GraphId:" << graph_id << " not found";
  }

  MS_LOG(DEBUG) << "Clear graph:" << graph_id << " dynamic kernels";
  if (auto dynamic_kernel_iter = graph_dynamic_kernel_map_.find(graph_id);
      dynamic_kernel_iter != graph_dynamic_kernel_map_.end()) {
    MS_LOG(DEBUG) << "Start Clear graph:" << graph_id << " dynamic kernel";
    graph_dynamic_kernel_map_.erase(dynamic_kernel_iter);
  }

  MS_LOG(DEBUG) << "Clear graph:" << graph_id << " runtime resource";
  if (auto model_iter = graph_model_map_.find(graph_id); model_iter != graph_model_map_.end()) {
    MS_LOG(DEBUG) << "Ge UnloadModel " << graph_id;
    auto ret = ModelRunner::Instance().UnloadModel(graph_id);
    if (!ret) {
      MS_LOG(ERROR) << "UnloadModel failed";
    }
    graph_model_map_.erase(model_iter);
  } else {
    MS_LOG(DEBUG) << "GraphId:" << graph_id << " not found";
  }
}

void AscendKernelRuntime::ClearGlobalIdleMem() { mem_manager_->ClearGlobalIdleMem(); }

bool AscendKernelRuntime::NeedDestroyHccl() {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (!context_ptr->get_param<bool>(MS_CTX_ENABLE_HCCL)) {
    MS_LOG(INFO) << "Hccl is not enabled";
    return false;
  }
  // Note: make sure hcom_connectivity_detection api never be used.
  return true;
}

void AsyncDataDumpUninit() {
  if (DumpJsonParser::GetInstance().async_dump_enabled()) {
    if (AdxDataDumpServerUnInit() != 0) {
      MS_LOG(ERROR) << "Adx data dump server uninit failed";
    }
  }
}

void AscendKernelRuntime::ReportProfilingData() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->get_param<bool>(MS_CTX_ENABLE_PROFILING) &&
      context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
    // Save Profiling Framework data
    OpNameTaskStreamReporter reporter(device_id_, "nonsink", stream_id_task_id_op_name_map_);
    reporter.ReportData();
  }
}

void AscendKernelRuntime::ReleaseDeviceRes() {
  MS_LOG(INFO) << "Ascend finalize start";
#ifdef ENABLE_DEBUGGER
  if (debugger_ && debugger_->debugger_enabled()) {
    debugger_->SetTrainingDone(true);
    debugger_->SendMetadata(false);
  }
#endif
  if (!initialized_) {
    return;
  }
  InnerSetContext();
  ReportProfilingData();
  // release ge runtime
  ClearGraphModelMap();

  AsyncDataDumpUninit();

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  uint32_t device_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);

  if (mem_manager_ != nullptr) {
    mem_manager_->FreeDeviceMemory();
  }

  auto rt_ret = rtRegTaskFailCallbackByModule(kModuleName, nullptr);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Reg SetTaskFailCallback failed, error: " << rt_ret;
  }

  (void)DestroySingleOpHccl();
  (void)DestroyHccl();
  (void)ResetDevice(device_id);
  (void)ProfilingManager::GetInstance().StopProfiling();
  current_graph_ = nullptr;
  MS_LOG(INFO) << "Ascend finalize end";
}

void AscendKernelRuntime::PreInit() {
  auto ret = ProfilingManager::GetInstance().StartupProfiling(device_id_);
  if (!ret) {
    MS_EXCEPTION(DeviceProcessError) << "StartupProfiling failed.";
  }
}

bool AscendKernelRuntime::Init() {
  if (initialized_) {
    InnerSetContext();
    return true;
  }
  OpTilingCalculater::GetInstance().Init();
  // Start up profiling before rtSetDevice

  bool ret = InitDevice();
  if (!ret) {
    return ret;
  }

  SetDebugger();
  mem_manager_ = std::make_shared<AscendMemoryManager>();
  MS_EXCEPTION_IF_NULL(mem_manager_);
  mem_manager_->MallocDeviceMemory();

  // Set callback func when exception error
  auto rt_ret = rtRegTaskFailCallbackByModule(kModuleName, TaskFailCallback);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Reg SetTaskFailCallback failed, error: " << rt_ret;
  }

  initialized_ = true;
  return ret;
}

bool AscendKernelRuntime::LoadData(mindspore::session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
#ifdef ENABLE_DEBUGGER
  MS_LOG(INFO) << "Start load step";
  for (const auto &graph_ptr : debugger_->GetGraphPtrList()) {
    debugger_->SetGraphPtr(graph_ptr);
    // load output
    debugger_->LoadGraphOutputs();
    // load parameters
    debugger_->LoadParametersAndConst();
  }
#endif
  return true;
}

bool AscendKernelRuntime::KernelMemNotReuse(const AnfNodePtr &node) {
  bool need_dump = false;
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (dump_json_parser.e2e_dump_enabled() && dump_json_parser.dump_mode() == 1) {
    auto op_name = node->fullname_with_scope();
    if (dump_json_parser.NeedDump(op_name)) {
      need_dump = true;
    }
  }
  return need_dump;
}

DeviceAddressPtr AscendKernelRuntime::CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format,
                                                          TypeId type_id) {
  return std::make_shared<AscendDeviceAddress>(device_ptr, device_size, format, type_id);
}

bool AscendKernelRuntime::Load(session::KernelGraph *graph, bool is_task_sink) {
  if (!is_task_sink) {
    return true;
  }

  // Do HcomExecutorInitialize
  if (graph->is_dynamic_shape() && !HcclExecutorManager::GetInstance().Initialize()) {
    MS_LOG(ERROR) << "Init Hccl Executor Failed";
    return false;
  }
  if (!GenTask(graph)) {
    return false;
  }
  if (!LoadTask(graph)) {
    return false;
  }
  return true;
}

bool AscendKernelRuntime::GenDynamicKernel(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "GenDynamicKernel start";
  auto cnode_list = graph->execution_order();
  std::vector<DynamicKernelPtr> dynamic_kernels;
  for (const auto &cnode : cnode_list) {
    MS_EXCEPTION_IF_NULL(cnode);
    MS_LOG(INFO) << "Generate node:" << cnode->fullname_with_scope() << " dynamic kernel";
    auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto dynamic_kernel = kernel_mod->GenDynamicKernel(cnode, stream_);
    if (dynamic_kernel == nullptr) {
      MS_LOG(EXCEPTION) << cnode->fullname_with_scope() << " does not support dynamic shape.";
    }
    dynamic_kernel->Initialize();
    dynamic_kernels.emplace_back(dynamic_kernel);
  }
  graph_dynamic_kernel_map_[graph->graph_id()] = dynamic_kernels;
  MS_LOG(INFO) << "GenDynamicKernel end";
  return true;
}

bool AscendKernelRuntime::GenTask(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  InnerSetContext();
  if (graph->is_dynamic_shape()) {
    if (ConfigManager::GetInstance().dataset_mode() == DS_SINK_MODE && (ConfigManager::GetInstance().iter_num() > 1)) {
      MS_LOG(EXCEPTION) << "Dynamic shape is not supported with sink mode.";
    }
    if (DumpJsonParser::GetInstance().async_dump_enabled()) {
      MS_LOG(EXCEPTION) << "Dynamic shape is not supported with asyn dump. Please use other debugging methods.";
    }
    MS_LOG(INFO) << "Dynamic Shape Graph Generate Dynamic kernel";
    return GenDynamicKernel(graph);
  }
  MS_LOG(INFO) << "GenTask start. GraphId:" << graph->graph_id();
  DumpJsonParser::GetInstance().UpdateNeedDumpKernels(NOT_NULL(graph));
#ifdef MEM_REUSE_DEBUG
  if (!EnvConfigParser::GetInstance().GetSysMemreuse()) {
    // Get normal graph ir for memreuse
    mindspore::memreuse::MemReuseChecker::GetInstance().CheckNormalIR(graph);
  }
#endif
  vector<std::shared_ptr<TaskInfo>> task_info_list;
  auto anf_node_list = graph->execution_order();
  auto task_generator = TaskGenerator();
  task_generator.GenTasks(anf_node_list, &task_info_list, graph->graph_id());
  // Store the task_info_list
  auto insert_ret = task_map_.insert(std::make_pair(graph->graph_id(), task_info_list));
  if (!insert_ret.second) {
    MS_LOG(EXCEPTION) << "Duplicate GraphId! Please check in ascend_session.";
  }
  // Graph may have no compute node, such TensorAddGrad.
  if (task_info_list.empty()) {
    MS_LOG(WARNING) << "Graph " << graph->graph_id() << " have no compute node";
    return true;
  }
  AscendStreamAssign &assign_instance = AscendStreamAssign::GetInstance();
  AscendResourceMng &resource_manager = AscendResourceMng::GetInstance();
  // the streams' flag not HEAD_STREAM
  std::vector<uint32_t> wait_active_stream_list;
  assign_instance.GetWaitStreams(&wait_active_stream_list);
  std::vector<uint32_t> force_copy_stream_list;
  assign_instance.GetHcomStreams(&force_copy_stream_list);
  MS_LOG(INFO) << "Call DavinciModel total stream num:" << resource_manager.get_cur_stream_num()
               << ", total event num:" << resource_manager.get_cur_event_num()
               << ", total label num:" << graph->label_num()
               << ", wait_active_stream_list size:" << wait_active_stream_list.size()
               << ", force_copy_stream_list size:" << force_copy_stream_list.size();
  std::vector<std::shared_ptr<ge::model_runner::OpInfo>> empty_list;
  auto model = std::make_shared<ge::model_runner::DavinciModel>(
    task_info_list, empty_list, empty_list, empty_list, empty_list, wait_active_stream_list, force_copy_stream_list, 0,
    0, 0, 0, 0, 0, resource_manager.get_cur_stream_num(), graph->label_num(), resource_manager.get_cur_event_num(), 0);
  auto ret = graph_model_map_.insert(std::make_pair(graph->graph_id(), model));
  if (!ret.second) {
    MS_LOG(EXCEPTION) << "Duplicate GraphId! Please check in ascend_session.";
  }
  MS_LOG(INFO) << "TaskGenerator GetTaskInfo end...";
  return true;
}

bool AscendKernelRuntime::LoadTask(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  InnerSetContext();
  if (graph->is_dynamic_shape()) {
    MS_LOG(INFO) << "Dynamic Shape Graph Skip Load Task Step";
    return true;
  }

  MS_LOG(INFO) << "LoadTask start. GraphId:" << graph->graph_id();
  if (GraphWithEmptyTaskList(graph)) {
    MS_LOG(WARNING) << "LoadTask end, task list is empty";
    return true;
  }

  auto model_iter = graph_model_map_.find(graph->graph_id());
  if (model_iter == graph_model_map_.end()) {
    MS_LOG(ERROR) << "GraphId:" << graph->graph_id() << " Invalid! Graph LoadTask without GenTask.";
    return false;
  }

  std::shared_ptr<ge::ModelListener> listener;
  MS_LOG(INFO) << "LoadDavinciModel mode_id:" << model_iter->first;
  bool status =
    ModelRunner::Instance().LoadDavinciModel(device_id_, 0, model_iter->first, model_iter->second, listener);
  if (!status) {
    MS_LOG(EXCEPTION) << "Load Model Failed";
  }

  std::function<void *()> model_handle =
    std::bind(&ModelRunner::GetModelHandle, &ModelRunner::Instance(), model_iter->first);
  DistributeDebugTask(NOT_NULL(graph), NOT_NULL(model_handle));

  status = ModelRunner::Instance().DistributeTask(model_iter->first);
  if (!status) {
#ifdef ENABLE_DUMP_IR
    mindspore::RDR::TriggerAll();
#endif
    MS_LOG(EXCEPTION) << "Distribute Task Failed";
  }

  if (ProfilingManager::GetInstance().IsProfiling()) {
    auto task_ids = ModelRunner::Instance().GetTaskIdList(model_iter->first);
    auto stream_ids = ModelRunner::Instance().GetStreamIdList(model_iter->first);
    ProfilingUtils::ReportProfilingData(task_ids, stream_ids, NOT_NULL(graph));
  }

  LaunchDataDump(graph->graph_id());

  if (!ModelRunner::Instance().LoadModelComplete(model_iter->first)) {
    MS_LOG(ERROR) << "Call ge runtime LoadModelComplete failed";
    return false;
  }
  return true;
}

void AscendKernelRuntime::DistributeDebugTask(NotNull<const session::KernelGraph *> graph,
                                              const NotNull<std::function<void *()>> &model_handle) {
  if (!DumpJsonParser::GetInstance().async_dump_enabled()) {
    return;
  }
  MS_LOG(INFO) << "Start Distribute Debug Task";
  auto data_dumper = std::make_shared<DataDumper>(graph.get(), model_handle);
  MS_EXCEPTION_IF_NULL(data_dumper);
  auto ret = graph_data_dumper_.try_emplace(graph->graph_id(), data_dumper);
  data_dumper->OpDebugRegister();
  if (!ret.second) {
    MS_LOG(WARNING) << "[DataDump] Insert graphId:" << graph->graph_id() << " data dumper failed";
  }
}

void AscendKernelRuntime::LaunchDataDump(GraphId graph_id) {
  if (!DumpJsonParser::GetInstance().async_dump_enabled()) {
    return;
  }
  MS_LOG(INFO) << "Start Launch Dump Data";
  auto runtime_info_map = ModelRunner::Instance().GetRuntimeInfoMap(graph_id);
  if (auto dumper_iter = graph_data_dumper_.find(graph_id); dumper_iter != graph_data_dumper_.end()) {
    auto &data_dumper = dumper_iter->second;
    MS_EXCEPTION_IF_NULL(data_dumper);
    data_dumper->set_runtime_info(runtime_info_map);
    data_dumper->LoadDumpInfo();
  } else {
    MS_LOG(EXCEPTION) << "GraphId:" << graph_id << " not found";
  }
}

void AscendKernelRuntime::TaskFailCallback(rtExceptionInfo *task_fail_info) {
  MS_EXCEPTION_IF_NULL(task_fail_info);
  static std::mutex exception_mutex;
  std::lock_guard<std::mutex> lock(exception_mutex);
  if (task_fail_info->retcode == ACL_ERROR_RT_AICORE_OVER_FLOW) {
    auto node = AscendKernelRuntime::GetErrorNodeName(task_fail_info->streamid, task_fail_info->taskid);

    if (!node) {
      MS_LOG(WARNING) << "Node run task overflow, node name is unknown.";
    } else {
      auto key = std::to_string(task_fail_info->streamid) + std::to_string(task_fail_info->taskid) +
                 std::to_string(current_graph_->graph_id());
      if (overflow_tasks_.find(key) == overflow_tasks_.end() || overflow_tasks_[key] == 5) {
        // print overflow info
        MS_LOG(WARNING) << "Node run task overflow, node name: " << node->fullname_with_scope()
                        << "Task overflow infos task_id: " << task_fail_info->taskid
                        << ", stream_id: " << task_fail_info->streamid << ", tid: " << task_fail_info->tid
                        << ", device_id: " << task_fail_info->deviceid << ", retcode: " << task_fail_info->retcode
                        << ", trace: " << trace::DumpSourceLines(node);
        overflow_tasks_[key] = 1;
      } else {
        overflow_tasks_[key]++;
      }
    }
  } else {
    task_fail_infoes_.push_back(*task_fail_info);
  }
}

CNodePtr AscendKernelRuntime::GetErrorNodeName(uint32_t streamid, uint32_t taskid) {
  auto runtime_info_map = ModelRunner::Instance().GetRuntimeInfoMap(current_graph_->graph_id());
  for (const auto &iter : runtime_info_map) {
    auto task_id = std::get<kTupleTaskId>(*iter.second);
    auto stream_id = std::get<kTupleStreamId>(*iter.second);
    if (task_id == taskid && stream_id == streamid) {
      auto &execute_node = current_graph_->execution_order();
      auto node = std::find_if(execute_node.begin(), execute_node.end(),
                               [&iter](const auto &node) { return node->fullname_with_scope() == iter.first; });
      if (node != execute_node.end()) {
        return *node;
      }
    }
  }
  return nullptr;
}

void AscendKernelRuntime::DumpTaskExceptionInfo(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  const std::string local_path = std::string("./task_error_dump/") + std::to_string(task_fail_infoes_.at(0).deviceid);
  for (const auto &task_fail_info : task_fail_infoes_) {
    MS_LOG(ERROR) << "Task fail infos task_id: " << task_fail_info.taskid << ", stream_id: " << task_fail_info.streamid
                  << ", tid: " << task_fail_info.tid << ", device_id: " << task_fail_info.deviceid
                  << ", retcode: " << task_fail_info.retcode;
    auto node = AscendKernelRuntime::GetErrorNodeName(task_fail_info.streamid, task_fail_info.taskid);
    // Dump error data in local path
    if (node == nullptr) {
      continue;
    }
    auto full_scope_name = node->fullname_with_scope();
    MS_LOG(ERROR) << "Dump node (" << full_scope_name << ") task error input/output data to: " << local_path
                  << " trace: " << trace::DumpSourceLines(node);
    E2eDump::DumpInputImpl(node, false, local_path, &full_scope_name, nullptr);
    E2eDump::DumpOutputImpl(node, false, local_path, &full_scope_name, nullptr);
  }
}

bool AscendKernelRuntime::Run(session::KernelGraph *const graph, bool is_task_sink) {
  SignalGuard sg;
  MS_EXCEPTION_IF_NULL(graph);
  bool ret = false;
#if defined(_WIN32) || defined(_WIN64)
  auto start_time = std::chrono::steady_clock::now();
#else
  struct timeval start_time {};
  struct timeval end_time {};
  (void)gettimeofday(&start_time, nullptr);
#endif
  if (is_task_sink) {
    ret = RunTask(graph);
  } else {
    ret = LaunchKernel(graph);
  }
#if defined(_WIN32) || defined(_WIN64)
  auto end_time = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::ratio<1, 1000000>> cost = end_time - start_time;
  MS_LOG(INFO) << "Call MS Run Success in " << cost.count() << " us";
#else
  (void)gettimeofday(&end_time, nullptr);
  const uint64_t kUSecondInSecond = 1000000;
  uint64_t cost = kUSecondInSecond * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
  cost += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
  MS_LOG(INFO) << "Call MS Run Success in " << cost << " us";
#endif
  return ret;
}

bool AscendKernelRuntime::RunDynamicKernelAsync(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "RunExecutorAsync start. GraphId:" << graph->graph_id();
  auto iter = graph_dynamic_kernel_map_.find(graph->graph_id());
  if (iter == graph_dynamic_kernel_map_.end()) {
    MS_LOG(ERROR) << "GraphId:" << graph->graph_id() << " Not Found! Please generator executor first";
    return false;
  }

  auto dynamic_kernels = iter->second;
  for (const auto &dynamic_kernel : dynamic_kernels) {
    if (dynamic_kernel->have_depends() || dynamic_kernel->GetKernelType() == KernelType::HCCL_KERNEL) {
      MS_LOG(INFO) << "Match Dynamic Kernel, Start SyncStream";
      if (!SyncStream()) {
        MS_LOG(ERROR) << "SyncStream failed";
        return false;
      }
    }

    if (dynamic_kernel->is_dynamic_shape()) {
      dynamic_kernel->InferShape();
      dynamic_kernel->UpdateArgs();
    }

    dynamic_kernel->Execute();
    dynamic_kernel->PostExecute();
  }

  if (!SyncStream()) {
    MS_LOG(ERROR) << "SyncStream failed";
    return false;
  }

  return true;
}

bool AscendKernelRuntime::RunTask(const session::KernelGraph *graph) {
  current_graph_ = graph;
  InnerSetContext();
  MS_EXCEPTION_IF_NULL(graph);
  if (graph->is_dynamic_shape()) {
    MS_LOG(INFO) << "Dynamic Shape Graph Run Task Async";
    return RunDynamicKernelAsync(graph);
  }

  MS_LOG(INFO) << "RunTask start. GraphId:" << graph->graph_id();

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  ge::InputData input_tensors = ge::InputData();
  ge::OutputData *output_tensors = nullptr;
  if (GraphWithEmptyTaskList(graph)) {
    MS_LOG(WARNING) << "RunTask end, no task info found";
    return true;
  }

  if (!CheckGraphIdValid(graph->graph_id())) {
    MS_LOG(ERROR) << "GraphId:" << graph->graph_id() << " Invalid! Graph RunTask without GenTask.";
    return false;
  }

  bool status = ModelRunner::Instance().RunModel(graph->graph_id(), input_tensors, output_tensors);
  if (!status) {
    DumpTaskExceptionInfo(graph);
    std::string file_name = "task_error_debug" + std::to_string(graph->graph_id()) + ".ir";
    auto graph_tmp = std::make_shared<session::KernelGraph>(*graph);
    DumpIR(file_name, graph_tmp);
#ifdef ENABLE_TDTQUE
    // Run task error, we should call TdtHostDestroy to release tdt to avoid DeviceQueueOp hostPush hung
    // case1: cpu usage 100% cause thread/process exit, but some tdt thread remain in backend
    if (!TdtHandle::DestroyHandle()) {
      MS_LOG(WARNING) << "Destroy tdt channel failed.";
    } else {
      MS_LOG(INFO) << "Destroy tdt channel success.";
    }
#endif
    return false;
  }
  task_fail_infoes_.clear();
  return true;
}

bool AscendKernelRuntime::SyncStream() {
  InnerSetContext();
  if (stream_ == nullptr) {
    MS_LOG(ERROR) << "SyncStream failed. stream_ is nullptr";
    return false;
  }
  if (RT_ERROR_NONE != rtStreamSynchronize(stream_)) {  // o for switch stream
    MS_LOG(ERROR) << "Call runtime rtStreamSynchronize error.";
    return false;
  }
  if (RT_ERROR_NONE != rtStreamSynchronize(communication_stream_)) {  // o for switch stream
    MS_LOG(ERROR) << "Call runtime rtStreamSynchronize error.";
    return false;
  }
  FreeAndClearBufferPtrs();
  return true;
}

bool AscendKernelRuntime::MemcpyAsync(void *dst, const void *src, uint64_t size, int32_t kind) {
  InnerSetContext();
  if (stream_ == nullptr) {
    MS_LOG(ERROR) << "MemcpyAsync failed. stream_ is nullptr";
    return false;
  }

  std::shared_ptr<char[]> buffer(new char[size]());
  MS_EXCEPTION_IF_NULL(buffer);
  std::copy(reinterpret_cast<const char *>(src), reinterpret_cast<const char *>(src) + size, buffer.get());
  AddBufferPtr(buffer);
  if (RT_ERROR_NONE != rtMemcpyAsync(dst, size, buffer.get(), size, static_cast<rtMemcpyKind_t>(kind), stream_)) {
    MS_LOG(ERROR) << "Call runtime rtMemcpyAsync error.";
    return false;
  }
  return true;
}

void AscendKernelRuntime::CreateContext() {
  if (rt_context_ == nullptr) {
    auto ret = rtCtxCreate(&rt_context_, 0, device_id_);
    if (ret != RT_ERROR_NONE) {
      MS_EXCEPTION(DeviceProcessError) << "Call rtCtxCreate, ret[" << static_cast<int>(ret) << "]";
    }
  }
  InnerSetContext();
}

bool AscendKernelRuntime::InitDevice() {
  int device_count = 0;
  auto ret = rtGetDeviceCount(&device_count);
  if (ret != RT_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "Call rtGetDeviceCount, ret[" << static_cast<int>(ret) << "]";
  }

  ret = rtSetDevice(device_id_);
  if (ret != RT_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "Call rtSetDevice, ret[" << static_cast<int>(ret) << "]";
  }

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr == nullptr) {
    MS_LOG(ERROR) << "Get MsContext instance failed";
    return false;
  }
  if (context_ptr->get_param<bool>(MS_CTX_ENABLE_HCCL)) {
    if (!HcclInit()) {
      MS_LOG(ERROR) << "HcclInit init failed";
      return false;
    }
  }

  // Context will be created by rtSetDevice
  ret = rtCtxGetCurrent(&rt_context_);
  if (ret != RT_ERROR_NONE || rt_context_ == nullptr) {
    MS_LOG(ERROR) << "Call rtCtxGetCurrent failed, ret[" << ret << "]";
    return false;
  }

  ret = rtStreamCreateWithFlags(&stream_, 0, RT_STREAM_HUGE);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rtStreamCreate, ret[" << ret << "]";
  }
  ret = rtStreamCreate(&communication_stream_, 0);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "create communication stream failed, ret:" << ret;
  }
  return true;
}

bool AscendKernelRuntime::ResetDevice(uint32_t device_id) {
  InnerSetContext();
  if (stream_ != nullptr) {
    auto ret = rtStreamDestroy(stream_);
    if (ret != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "Call rtStreamDestroy, ret[" << ret << "]";
    }
    stream_ = nullptr;
  }

  if (communication_stream_ != nullptr) {
    auto ret = rtStreamDestroy(communication_stream_);
    if (ret != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "Call rtStreamDestroy, ret[" << ret << "]";
    }
    communication_stream_ = nullptr;
  }

  auto ret = rtDeviceReset(device_id);
  if (ret != RT_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "Call rtDeviceReset, ret[" << ret << "]";
  }
  // set to nullptr as its not created, only bounded to existing context
  rt_context_ = nullptr;
  return true;
}

bool AscendKernelRuntime::HcclInit() {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (!context::IsTsdOpened(context_ptr)) {
    MS_LOG(EXCEPTION) << "Hccl dependent tsd is not open";
  }
  MS_LOG(INFO) << "Do hcom init";
  auto config_path_str = std::getenv("MINDSPORE_HCCL_CONFIG_PATH");
  if (config_path_str == nullptr) {
    config_path_str = std::getenv("RANK_TABLE_FILE");
    if (config_path_str == nullptr) {
      MS_LOG(ERROR) << "Get hccl json config failed, please set env MINDSPORE_HCCL_CONFIG_PATH or RANK_TABLE_FILE";
      return false;
    }
  }
  if (strlen(config_path_str) > PATH_MAX) {
    MS_LOG(ERROR) << "File path oversize";
    return false;
  }
  std::string rank_id_str = GetRankId();
  auto full_path = realpath(config_path_str, nullptr);
  if (full_path == nullptr) {
    MS_LOG(ERROR) << "File path " << config_path_str << " does not exist";
    return false;
  }
  MS_LOG(INFO) << "MINDSPORE_HCCL_CONFIG_PATH : " << full_path << ", RANK_ID: " << rank_id_str;
  bool ret = hccl::InitHccl(context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID), rank_id_str, full_path);
  free(full_path);
  if (!ret) {
    MS_LOG(ERROR) << "Hcom init failed.";
    return false;
  }
  if (context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
    MS_LOG(INFO) << "PyNative hccl init";
    return kernel::HcclContext::GetInstance().InitHccl();
  }
  return true;
}

bool AscendKernelRuntime::DestroySingleOpHccl() {
  if (!kernel::HcclContext::GetInstance().Finalize()) {
    MS_LOG(ERROR) << "Hccl finalize failed";
    return false;
  }
  return true;
}

bool AscendKernelRuntime::DestroyHccl() {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
    return true;
  }
  if (!NeedDestroyHccl()) {
    MS_LOG(INFO) << "Hccl is not enable, no need to close.";
    return true;
  }
  // Dynamic Shape Hccl Finalize
  if (!HcclExecutorManager::GetInstance().Finalize()) {
    MS_LOG(ERROR) << "Dynamic Shape Hccl Finalize Failed";
  }
  bool res = hccl::FinalizeHccl();
  if (!res) {
    MS_LOG(ERROR) << "Hccl destroy failed";
    return false;
  }
  MS_LOG(INFO) << "Hccl destroy successful.";
  context_ptr->set_param<bool>(MS_CTX_ENABLE_HCCL, false);
  return true;
}

bool AscendKernelRuntime::GraphWithEmptyTaskList(const session::KernelGraph *graph) const {
  auto iter = task_map_.find(graph->graph_id());
  if (iter == task_map_.end()) {
    MS_LOG(EXCEPTION) << "Unknown graph ptr";
  }
  return iter->second.empty();
}

bool AscendKernelRuntime::CheckGraphIdValid(GraphId graph_id) const {
  return task_map_.find(graph_id) != task_map_.end() && graph_model_map_.find(graph_id) != graph_model_map_.end();
}

void AscendKernelRuntime::KernelLaunchProfiling(const std::string &kernel_name) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (!context->get_param<bool>(MS_CTX_ENABLE_PROFILING)) {
    return;
  }
  // save task info
  uint32_t stream_id;
  uint32_t task_id;
  auto rt_ret = rtGetTaskIdAndStreamID(&task_id, &stream_id);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Profiling get task_id stream_id failed";
  }
  std::pair<uint32_t, uint32_t> stream_task_pair = {stream_id, task_id};
  auto try_emplace_ret = stream_id_task_id_op_name_map_.try_emplace(stream_task_pair, kernel_name);
  if (!try_emplace_ret.second) {
    MS_LOG(WARNING) << "Profiling duplicate key, task_id:" << stream_task_pair.second
                    << " stream_id:" << stream_task_pair.first << " name:" << kernel_name;
  }
  if (stream_id_task_id_op_name_map_.size() > kProfilingMaxTaskIdInStream) {
    MS_LOG(EXCEPTION) << "Too many profiling data";
  }
}

uint64_t AscendKernelRuntime::GetAvailableMemMaxSize() const {
  auto ascend_mem_manager = dynamic_pointer_cast<AscendMemoryManager>(mem_manager_);
  return ascend_mem_manager->GetDeviceMemSize();
}

}  // namespace mindspore::device::ascend
