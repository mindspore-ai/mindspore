/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/hal/device/ascend_kernel_runtime.h"
#include <locale>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <algorithm>
#include <set>
#include "include/common/utils/signal_util.h"
#include "plugin/device/ascend/hal/device/ascend_device_address.h"
#include "utils/ms_context.h"
#include "include/common/utils/mpi/mpi_config.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "runtime/rt.h"
#include "acl/acl_rt.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "plugin/device/ascend/hal/device/ascend_stream_assign.h"
#include "plugin/device/ascend/hal/device/ge_runtime/model_runner.h"
#include "plugin/device/ascend/hal/device/tasksink/task_generator.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "backend/common/optimizer/dynamic_shape/dynamic_shape_helper.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/session/kernel_build_client.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_kernel_load.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "kernel/oplib/op_info_utils.h"
#ifndef ENABLE_SECURITY
#include "plugin/device/ascend/hal/device/profiling/profiling_manager.h"
#include "plugin/device/ascend/hal/device/profiling/profiling_utils.h"
#endif
#include "plugin/device/ascend/hal/device/ascend_memory_manager.h"
#include "plugin/device/ascend/hal/device/ascend_event.h"
#ifndef ENABLE_SECURITY
#include "plugin/device/ascend/hal/device/dump/ascend_dump.h"
#include "debug/data_dump/dump_json_parser.h"
#include "debug/data_dump/e2e_dump.h"
#include "plugin/device/ascend/hal/device/dump/kernel_dumper.h"
#endif
#include "toolchain/adx_datadump_server.h"
#include "utils/trace_base.h"
#include "graphengine/inc/external/acl/error_codes/rt_error_codes.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/parallel_context.h"
#include "include/common/utils/comm_manager.h"
#ifdef MEM_REUSE_DEBUG
#include "common/mem_reuse/mem_reuse_checker.h"
#include "include/common/debug/env_config_parser.h"
#endif
#include "include/common/utils/config_manager.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#include "plugin/device/ascend/hal/device/ascend_data_queue.h"
#ifdef ENABLE_DUMP_IR
#include "include/common/debug/rdr/recorder_manager.h"
#endif

#include "profiler/device/profiling.h"
#include "kernel/common_utils.h"
#include "plugin/device/ascend/optimizer/platform.h"
#ifndef ENABLE_SECURITY
using mindspore::device::ascend::ProfilingManager;
using mindspore::device::ascend::ProfilingUtils;
#endif
using mindspore::device::ascend::tasksink::TaskGenerator;
using mindspore::ge::model_runner::ModelRunner;
using mindspore::kernel::tbe::TbeUtils;
using mindspore::opt::PlatformInfoInitialization;
using std::vector;

constexpr uint32_t kTupleTaskId = 0;
constexpr uint32_t kTupleStreamId = 1;
constexpr uint32_t kTupleArgs = 2;
constexpr uint32_t kProfilingMaxTaskIdInStream = 65531;
constexpr auto kModuleName = "MindSpore";
constexpr size_t kPathMax = 4096;

namespace mindspore::device::ascend {
static thread_local rtContext_t thread_local_rt_context{nullptr};
namespace {
void IntHandler(int, siginfo_t *, void *) {
  mindspore::kernel::AscendKernelBuildClient::Instance().Close();
  int this_pid = getpid();
  MS_LOG(WARNING) << "Process " << this_pid << " receive KeyboardInterrupt signal.";
  (void)kill(this_pid, SIGTERM);
}

void AscendEnableDynamicRuntimeCache(const session::KernelGraph *graph) {
  const auto &node_list = FuncGraph::TopoSort(graph->get_return());
  for (auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    auto kernel_info = node->kernel_info();
    if (!kernel_info) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(kernel_info);
    auto runtime_cache = kernel_info->runtime_cache();
    runtime_cache.runtime_cache().set_is_valid(true);
  }
}
}  // namespace

struct TbeLaunchKernelModRegister {
  TbeLaunchKernelModRegister() {
    KernelRuntime::tbe_call_setter(
      [](const AnfNodePtr &kernel, const kernel::KernelMod *kernel_mod, std::vector<AddressPtr> *workspace_addr) {
        MS_EXCEPTION_IF_NULL(kernel);
        MS_EXCEPTION_IF_NULL(kernel_mod);
        MS_EXCEPTION_IF_NULL(workspace_addr);
        auto workspace_size_list = kernel_mod->GetWorkspaceSizeList();
        auto ms_context = MsContext::GetInstance();
        MS_EXCEPTION_IF_NULL(ms_context);
        auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
        auto runtime_instance = KernelRuntimeManager::Instance().GetSingleKernelRuntime(kAscendDevice, device_id);
        MS_EXCEPTION_IF_NULL(runtime_instance);
        for (auto size : workspace_size_list) {
          auto device_address_ptr =
            std::make_shared<ascend::AscendDeviceAddress>(nullptr, size, kAscendDevice, device_id);
          device_address_ptr->set_is_ptr_persisted(true);
          auto ret = runtime_instance->GetMemoryManager()->MallocMemFromMemPool(device_address_ptr, size);
          if (!ret) {
            MS_LOG(EXCEPTION) << "MallocMem from memory pool failed. Node info :" << kernel->fullname_with_scope();
          }
          AddressPtr workspace_addr_ptr =
            std::make_shared<kernel::Address>(device_address_ptr->GetMutablePtr(), device_address_ptr->GetSize());
          (void)workspace_addr->emplace_back(workspace_addr_ptr);
        }
      });
  }
  TbeLaunchKernelModRegister(const TbeLaunchKernelModRegister &) = delete;
  TbeLaunchKernelModRegister &operator=(const TbeLaunchKernelModRegister &) = delete;
  ~TbeLaunchKernelModRegister() = default;
} tbe_launch_kernel_mod_register;

std::vector<rtExceptionInfo> AscendKernelRuntime::task_fail_infoes_ = {};
const session::KernelGraph *current_graph_ = nullptr;
std::map<std::string, uint32_t> AscendKernelRuntime::overflow_tasks_;
AscendKernelRuntime::~AscendKernelRuntime() {
  graph_model_map_.clear();
  current_graph_ = nullptr;
  rt_context_ = nullptr;
}

void AscendKernelRuntime::SetContext() {
  ErrorManagerAdapter::BindToCurrentThread();
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

void AscendKernelRuntime::SetCurrentContext() {
  if (rt_context_ == nullptr) {
    return;
  }
  auto ret = rtCtxSetCurrent(rt_context_);
  if (ret != RT_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "Call rtCtxSetCurrent, ret[" << ret << "]";
  }
}

void AscendKernelRuntime::ClearGraphModelMap() {
  SetCurrentContext();
#ifndef ENABLE_SECURITY
  for (auto &iter : graph_data_dumper_) {
    MS_LOG(INFO) << "[DataDump] Unload data dumper:" << iter.first;
    auto &data_dumper = iter.second;
    MS_EXCEPTION_IF_NULL(data_dumper);
    try {
      data_dumper->UnloadDumpInfo();
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "UnloadDumpInfo failed: " << e.what();
    }
    try {
      data_dumper->OpDebugUnregister();
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "OpDebugUnregister failed: " << e.what();
    }
  }
  graph_data_dumper_.clear();
  // tell users which dump kernel name not used
  DumpJsonParser::GetInstance().PrintUnusedKernel();
  if (DumpJsonParser::GetInstance().async_dump_enabled()) {
    KernelDumper kernel_dumper;
    kernel_dumper.OpDebugUnregisterForStream();
  }
#endif

  graph_kernel_events_map_.clear();
  for (auto &iter : graph_model_map_) {
    MS_LOG(INFO) << "Ge UnloadModel " << iter.first;
    ModelRunner::Instance().UnloadModel(iter.first);
  }
  graph_model_map_.clear();
}

void AscendKernelRuntime::ClearGraphRuntimeResource(uint32_t graph_id) {
  SetCurrentContext();
  auto mem_scheduler = mem_scheduler_manager_.GetMemScheduler(graph_id);
  if (mem_scheduler != nullptr) {
    mem_scheduler->Clear();
  }
  MS_LOG(DEBUG) << "Clear graph:" << graph_id << " data dumper";
#ifndef ENABLE_SECURITY
  if (auto dumper_iter = graph_data_dumper_.find(graph_id); dumper_iter != graph_data_dumper_.end()) {
    MS_LOG(DEBUG) << "Unload dump info " << graph_id;
    auto &data_dumper = dumper_iter->second;
    MS_EXCEPTION_IF_NULL(data_dumper);
    data_dumper->UnloadDumpInfo();
    data_dumper->OpDebugUnregister();
    (void)graph_data_dumper_.erase(dumper_iter);
  } else {
    MS_LOG(DEBUG) << "GraphId:" << graph_id << " not found";
  }
#endif

  const auto events_iter = graph_kernel_events_map_.find(graph_id);
  if (events_iter != graph_kernel_events_map_.end()) {
    (void)graph_kernel_events_map_.erase(events_iter);
  }
  MS_LOG(DEBUG) << "Clear graph:" << graph_id << " runtime resource";
  if (auto model_iter = graph_model_map_.find(graph_id); model_iter != graph_model_map_.end()) {
    MS_LOG(DEBUG) << "Ge UnloadModel " << graph_id;
    ModelRunner::Instance().UnloadModel(graph_id);
    (void)graph_model_map_.erase(model_iter);
  } else {
    MS_LOG(DEBUG) << "GraphId:" << graph_id << " not found";
  }

  rt_model_zero_copy_.Release(graph_id);
}

void *AscendKernelRuntime::GetModelStream(uint32_t graph_id) const {
  return ModelRunner::Instance().GetModelStream(graph_id);
}

void *AscendKernelRuntime::GetKernelStream(const AnfNodePtr &kernel) const {
  const auto stream = AscendStreamMng::GetInstance().GetStream(AnfAlgo::GetStreamId(kernel));
  if (stream == nullptr) {
    // Stream id may not be assigned in some scenarios, such as PyNative. Use the default stream in those cases.
    return stream_;
  }
  return stream;
}

void AscendKernelRuntime::ClearGlobalIdleMem() {
  if (mem_manager_ != nullptr) {
    mem_manager_->ClearGlobalIdleMem();
  }
}

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

#ifndef ENABLE_SECURITY
void AsyncDataDumpUninit() {
  if (DumpJsonParser::GetInstance().async_dump_enabled()) {
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    auto device_type = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    if (device_type == kAscendDevice) {
      // When it is A+M dump mode, wait until file save is finished.
      if (DumpJsonParser::GetInstance().FileFormatIsNpy()) {
        mindspore::ascend::AscendAsyncDumpManager::GetInstance().WaitForWriteFileFinished();
      }
    }
    if (AdxDataDumpServerUnInit() != 0) {
      MS_LOG(ERROR) << "Adx data dump server uninit failed";
    }
  }
}
#endif

void AscendKernelRuntime::ReleaseDeviceRes() {
  MS_LOG(INFO) << "Ascend finalize start";
#ifdef ENABLE_DEBUGGER
  if (debugger_ && debugger_->debugger_enabled()) {
    debugger_->SetTrainingDone(true);
    bool ret = debugger_->SendMetadata(false);
    if (!ret) {
      MS_LOG(ERROR) << "Failed to SendMetadata when finalize";
    }
  }
#endif
  SetCurrentContext();

  // release ge runtime
  ClearGraphModelMap();

#ifndef ENABLE_SECURITY
  AsyncDataDumpUninit();
#endif

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  uint32_t device_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);

  // DestroyHccl must be called before FreeDeviceMemory
  (void)DestroyHccl();
  if (mem_manager_ != nullptr) {
    mem_manager_->Finalize();
  }
  mindspore::kernel::AicpuOpKernelLoad::GetInstance().FreeDeviceMemory();

  auto rt_ret = rtRegTaskFailCallbackByModule(kModuleName, nullptr);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Reg SetTaskFailCallback failed, error: " << rt_ret;
  }

  (void)ResetDevice(device_id);
  current_graph_ = nullptr;
  initialized_ = false;
  MS_LOG(INFO) << "Ascend finalize end";
}

#ifndef ENABLE_SECURITY
void AscendKernelRuntime::PreInit() {
  if (!ErrorManagerAdapter::Init()) {
    MS_LOG(WARNING) << "Init ErrorManager failed.";
  }
}
#endif

bool AscendKernelRuntime::Init() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto execution_mode = ms_context->get_param<int>(MS_CTX_EXECUTION_MODE);
#ifndef ENABLE_SECURITY
  auto profiler_manager = profiler::ProfilerManager::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_manager);
  auto profiling_flag = profiler_manager->GetProfilingEnableFlag();
  if (execution_mode == kPynativeMode && profiling_flag) {
    pynative_mode_profiling_flag_ = true;
  }
#endif
  if (initialized_) {
    SetCurrentContext();
    return true;
  }

  auto soc_version = device::ascend::GetSocVersion();
  auto ascend_path = device::ascend::GetAscendPath();
  if (!mindspore::kernel::OpInfoUtils::GenerateOpInfos(soc_version, ascend_path)) {
    MS_LOG(EXCEPTION) << "Load op info form json config failed, version: " << soc_version;
  }
  if (!ErrorManagerAdapter::Init()) {
    MS_LOG(WARNING) << "Init ErrorManager failed.";
  }
  bool init_device = false;
  try {
    // Start up profiling before rtSetDevice
    bool ret = InitDevice();
    if (!ret) {
      return ret;
    }
    init_device = true;
#ifdef ENABLE_DEBUGGER
    SetDebugger();
#endif
    mem_manager_ = std::make_shared<AscendMemoryManager>();
    MS_EXCEPTION_IF_NULL(mem_manager_);
    mem_manager_->Initialize();

    // Set callback func when exception error
    auto rt_ret = rtRegTaskFailCallbackByModule(kModuleName, TaskFailCallback);
    if (rt_ret != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "Reg SetTaskFailCallback failed, error: " << rt_ret;
    }
    if (!PlatformInfoInitialization(soc_version)) {
      MS_LOG(EXCEPTION) << "PlatformInfo Initialization failed.";
    }

    uint32_t op_timeout = ms_context->get_param<uint32_t>(MS_CTX_OP_TIMEOUT);
    auto acl_ret = aclrtSetOpWaitTimeout(op_timeout);
    if (acl_ret != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "Set op wait timeout failed, error: " << acl_ret;
    }
  } catch (const std::exception &e) {
    if (init_device) {
      ResetDevice(device_id_);
    }
    MS_LOG(EXCEPTION) << "Ascend kernel runtime initialization failed. The details refer to 'Ascend Error Message'."
                      << "#dmsg#Framework Error Message:#dmsg#" << e.what();
  }

  initialized_ = true;
  return true;
}

bool AscendKernelRuntime::LoadData(const session::KernelGraph & /* graph */) {
#ifdef ENABLE_DEBUGGER
  MS_LOG(INFO) << "Start load step";
  MS_EXCEPTION_IF_NULL(debugger_);
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
  MS_EXCEPTION_IF_NULL(node);
  bool need_dump = false;
#ifndef ENABLE_SECURITY
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (dump_json_parser.e2e_dump_enabled() && dump_json_parser.dump_mode() == 1) {
    auto op_name = node->fullname_with_scope();
    if (dump_json_parser.NeedDump(op_name)) {
      need_dump = true;
    }
  }
#endif
  return need_dump;
}

DeviceAddressPtr AscendKernelRuntime::CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format,
                                                          TypeId type_id) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto ascend_device_address_ptr =
    std::make_shared<AscendDeviceAddress>(device_ptr, device_size, format, type_id, kAscendDevice, device_id);
  MS_EXCEPTION_IF_NULL(ascend_device_address_ptr);
  ascend_device_address_ptr->set_is_ptr_persisted(true);
  return ascend_device_address_ptr;
}

DeviceAddressPtr AscendKernelRuntime::CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format,
                                                          TypeId type_id, const KernelWithIndex &node_index) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto ascend_device_address_ptr = std::make_shared<AscendDeviceAddress>(device_ptr, device_size, format, type_id,
                                                                         node_index, kAscendDevice, device_id);
  MS_EXCEPTION_IF_NULL(ascend_device_address_ptr);
  ascend_device_address_ptr->set_is_ptr_persisted(true);
  return ascend_device_address_ptr;
}

bool AscendKernelRuntime::Load(const session::KernelGraph &graph, bool is_task_sink) {
  if (!is_task_sink) {
    MS_LOG(INFO) << "Graph mode with not task sink";
    GenKernelEvents(graph);
    return true;
  }

  if (!GenTask(graph)) {
    return false;
  }
  if (!LoadTask(graph)) {
    return false;
  }
  return mindspore::kernel::AicpuOpKernelLoad::GetInstance().LaunchAicpuKernelSo();
}

bool AscendKernelRuntime::GenTask(const session::KernelGraph &graph) {
  SetCurrentContext();
  if (graph.is_dynamic_shape()) {
    if (ConfigManager::GetInstance().dataset_mode() == DS_SINK_MODE && (ConfigManager::GetInstance().iter_num() > 1)) {
      MS_LOG(EXCEPTION) << "Dynamic shape is not supported with dataset_sink_mode.";
    }
#ifndef ENABLE_SECURITY
    if (DumpJsonParser::GetInstance().async_dump_enabled()) {
      MS_LOG(EXCEPTION) << "Dynamic shape is not supported with Asynchronous Dump. Please use Synchronous Dump.";
    }
#endif
    MS_LOG(INFO) << "Dynamic Shape Graph Generate Dynamic kernel";
    return true;
  }
  MS_LOG(INFO) << "GenTask start. GraphId:" << graph.graph_id();
#ifndef ENABLE_SECURITY
  if (!MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
    // Update needed dump kernels for old runtime.
    DumpJsonParser::GetInstance().UpdateNeedDumpKernels(graph);
  }
#endif
#ifdef MEM_REUSE_DEBUG
  if (!EnvConfigParser::GetInstance().GetSysMemreuse()) {
    // Get normal graph ir for memreuse
    mindspore::memreuse::MemReuseChecker::GetInstance().CheckNormalIR(&graph);
  }
#endif
  vector<std::shared_ptr<TaskInfo>> task_info_list;
  auto anf_node_list = graph.execution_order();
  auto task_generator = TaskGenerator();
  if (!task_generator.GenTasks(anf_node_list, &task_info_list, graph.graph_id())) {
    return false;
  }
  // Store the task_info_list
  auto insert_ret = task_map_.insert(std::make_pair(graph.graph_id(), task_info_list));
  if (!insert_ret.second) {
    MS_LOG(EXCEPTION) << "Duplicate GraphId! Please check in ascend_session.";
  }
  // Graph may have no compute node, such TensorAddGrad.
  if (task_info_list.empty()) {
    MS_LOG(INFO) << "Graph " << graph.graph_id() << " have no compute node";
    return true;
  }
  AscendStreamAssign &assign_instance = AscendStreamAssign::GetInstance();
  AscendStreamMng &resource_manager = AscendStreamMng::GetInstance();
  // the streams' flag not HEAD_STREAM
  std::vector<uint32_t> wait_active_stream_list;
  assign_instance.GetWaitStreams(&wait_active_stream_list);
  std::vector<uint32_t> force_copy_stream_list;
  assign_instance.GetHcomStreams(&force_copy_stream_list);
  MS_LOG(INFO) << "Call DavinciModel total stream num:" << resource_manager.cur_stream_num()
               << ", total event num:" << resource_manager.cur_event_num() << ", total label num:" << graph.label_num()
               << ", wait_active_stream_list size:" << wait_active_stream_list.size()
               << ", force_copy_stream_list size:" << force_copy_stream_list.size();
  auto model = std::make_shared<ge::model_runner::DavinciModel>(
    task_info_list, wait_active_stream_list, force_copy_stream_list, stream_, 0, 0, 0, 0, 0, 0,
    resource_manager.cur_stream_num(), graph.label_num(), resource_manager.cur_event_num(), 0);
  auto ret = graph_model_map_.insert(std::make_pair(graph.graph_id(), model));
  if (!ret.second) {
    MS_LOG(EXCEPTION) << "Duplicate GraphId! Please check in ascend_session.";
  }
  MS_LOG(INFO) << "TaskGenerator GetTaskInfo end...";
  return true;
}

bool AscendKernelRuntime::LoadTask(const session::KernelGraph &graph) {
  SetCurrentContext();
  if (graph.is_dynamic_shape()) {
    MS_LOG(INFO) << "Dynamic Shape Graph Skip Load Task Step";
    return true;
  }

  MS_LOG(INFO) << "LoadTask start. GraphId:" << graph.graph_id();
  if (GraphWithEmptyTaskList(graph)) {
    MS_LOG(INFO) << "LoadTask end, task list is empty";
    return true;
  }

  auto model_iter = graph_model_map_.find(graph.graph_id());
  if (model_iter == graph_model_map_.end()) {
    MS_LOG(ERROR) << "GraphId:" << graph.graph_id() << " Invalid! Graph LoadTask without GenTask.";
    return false;
  }

  MS_LOG(INFO) << "LoadDavinciModel mode_id:" << model_iter->first;
  ModelRunner::Instance().LoadDavinciModel(device_id_, model_iter->first, model_iter->first, model_iter->second);

#ifndef ENABLE_SECURITY
  std::function<void *()> model_handle =
    std::bind(&ModelRunner::GetModelHandle, &ModelRunner::Instance(), model_iter->first);
  DistributeDebugTask(graph, NOT_NULL(model_handle));
#endif

  try {
    ModelRunner::Instance().DistributeTask(model_iter->first);
  } catch (const std::exception &e) {
#ifdef ENABLE_DUMP_IR
    mindspore::RDR::TriggerAll();
#endif
    MS_LOG(EXCEPTION) << "Distribute Task Failed, \nerror msg: " << e.what();
  }

  if (!rt_model_zero_copy_.GenerateZeroCopyTasks(graph)) {
    MS_LOG(ERROR) << "Generate ZeroCopyTask failed, graph id " << graph.graph_id();
    return false;
  }

#ifndef ENABLE_SECURITY
  if (ProfilingManager::GetInstance().IsProfilingInitialized()) {
    auto task_ids = ModelRunner::Instance().GetTaskIdList(model_iter->first);
    auto stream_ids = ModelRunner::Instance().GetStreamIdList(model_iter->first);
    uint32_t rt_model_id = 0;
    rtModel_t rt_model_handle = ModelRunner::Instance().GetModelHandle(model_iter->first);
    rtError_t rt_model_ret = rtModelGetId(rt_model_handle, &rt_model_id);
    if (rt_model_ret != RT_ERROR_NONE) {
      MS_LOG(WARNING) << "[profiler] Call rt api rtModelGetId failed, ret: " << rt_model_ret;
    } else {
      MS_LOG(INFO) << "[profiler] Call rt api rtModelGetId success, rt_model_id: " << rt_model_id;
    }
    // Report data directly if profiling is start
    if (ProfilingUtils::ValidComputeGraph(graph)) {
      if (ProfilingManager::GetInstance().IsProfilingStart()) {
        ProfilingUtils::ReportProfilingData(task_ids, stream_ids, graph.graph_id(), rt_model_id);
      } else {
        // Cache data and save when profiling is start
        ProfilingUtils::SetReportProfilingData(task_ids, stream_ids, graph.graph_id(), rt_model_id);
      }
    }
  }
  LaunchDataDump(graph.graph_id());
#endif

  ModelRunner::Instance().LoadModelComplete(model_iter->first);
  return true;
}

#ifndef ENABLE_SECURITY
void AscendKernelRuntime::DistributeDebugTask(const session::KernelGraph &graph,
                                              const NotNull<std::function<void *()>> &model_handle) {
  if (!DumpJsonParser::GetInstance().async_dump_enabled()) {
    return;
  }
  MS_LOG(INFO) << "Start Distribute Debug Task";
  auto data_dumper = std::make_shared<DataDumper>(&graph, model_handle);
  MS_EXCEPTION_IF_NULL(data_dumper);
  auto ret = graph_data_dumper_.try_emplace(graph.graph_id(), data_dumper);
  data_dumper->OpDebugRegister();
  if (!ret.second) {
    MS_LOG(WARNING) << "[DataDump] Insert graphId:" << graph.graph_id() << " data dumper failed";
  }
}

void AscendKernelRuntime::LaunchDataDump(GraphId graph_id) {
  if (!DumpJsonParser::GetInstance().async_dump_enabled()) {
    return;
  }
  MS_LOG(INFO) << "Start Launch Dump Data";
  auto runtime_info_map = ModelRunner::Instance().GetRuntimeInfoMap(graph_id);
  auto end_graph_info_map = ModelRunner::Instance().GetEndGraphInfoMap(graph_id);
  if (auto dumper_iter = graph_data_dumper_.find(graph_id); dumper_iter != graph_data_dumper_.end()) {
    auto &data_dumper = dumper_iter->second;
    MS_EXCEPTION_IF_NULL(data_dumper);
    data_dumper->set_runtime_info(runtime_info_map);
    data_dumper->set_end_graph(end_graph_info_map);
    data_dumper->LoadDumpInfo();
  } else {
    MS_LOG(EXCEPTION) << "GraphId:" << graph_id << " not found";
  }
}
#endif

void AscendKernelRuntime::TaskFailCallback(rtExceptionInfo *task_fail_info) {
  if (task_fail_info == nullptr) {
    MS_LOG(ERROR) << "Execute TaskFailCallback failed. task_fail_info is nullptr";
    return;
  }
  if (task_fail_info->retcode == ACL_ERROR_RT_AICORE_OVER_FLOW && KernelDumper::stream_task_graphs.size() > 0) {
    MS_LOG(WARNING) << "Graph in kernelByKernel mode task overflow, "
                    << "Task overflow infos task_id: " << task_fail_info->taskid
                    << ", stream_id: " << task_fail_info->streamid;
    return;
  }
  if (current_graph_ == nullptr) {
    MS_LOG(ERROR) << "Execute TaskFailCallback failed. current_graph_ is nullptr";
    return;
  }
  static std::mutex exception_mutex;
  constexpr uint32_t kOverflowThreshold = 5;
  std::lock_guard<std::mutex> lock(exception_mutex);
  if (task_fail_info->retcode == ACL_ERROR_RT_AICORE_OVER_FLOW) {
    auto node = AscendKernelRuntime::GetErrorNodeName(task_fail_info->streamid, task_fail_info->taskid);
    if (!node) {
      MS_LOG(WARNING) << "Node run task overflow, node name is unknown.";
    } else {
      auto key = std::to_string(task_fail_info->streamid) + std::to_string(task_fail_info->taskid) +
                 std::to_string(current_graph_->graph_id());
      if (overflow_tasks_.find(key) == overflow_tasks_.end() || overflow_tasks_[key] == kOverflowThreshold) {
        // print overflow info
        MS_LOG(WARNING) << "Node run task overflow, node name: " << node->fullname_with_scope()
                        << "Task overflow infos task_id: " << task_fail_info->taskid
                        << ", stream_id: " << task_fail_info->streamid << ", tid: " << task_fail_info->tid
                        << ", device_id: " << task_fail_info->deviceid << ", retcode: " << task_fail_info->retcode
                        << " (" << GetErrorMsg(task_fail_info->retcode) << ")" << trace::DumpSourceLines(node, false);
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
  if (current_graph_ == nullptr) {
    return nullptr;
  }
  auto runtime_info_map = ModelRunner::Instance().GetRuntimeInfoMap(current_graph_->graph_id());
  for (const auto &iter : runtime_info_map) {
    MS_EXCEPTION_IF_NULL(iter.second);
    auto task_id = std::get<kTupleTaskId>(*iter.second);
    auto stream_id = std::get<kTupleStreamId>(*iter.second);
    if (task_id == taskid && stream_id == streamid) {
      MS_EXCEPTION_IF_NULL(current_graph_);
      auto &execute_node = current_graph_->execution_order();
      auto node = std::find_if(execute_node.begin(), execute_node.end(), [&iter](const auto &node) {
        MS_EXCEPTION_IF_NULL(node);
        return node->UniqueName() == iter.first;
      });
      if (node != execute_node.end()) {
        return *node;
      }
    }
  }
  return nullptr;
}

std::string AscendKernelRuntime::GetDumpPath() {
  uint32_t rank_id = 0;
  auto inst = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(inst);
  if (inst->parallel_mode() != parallel::kStandalone) {
    if (!CommManager::GetInstance().GetRankID(kHcclWorldGroup, &rank_id)) {
      MS_LOG(WARNING) << "Get rank id failed, now using the default value 0.";
    }
  }

  auto ms_om_path = common::GetEnv("MS_OM_PATH");
  std::string path;
  const auto kSuffix = "/node_dump";
  if (ms_om_path.empty()) {
    MS_LOG(WARNING) << "The environment variable 'MS_OM_PATH' is not set, the files of node dump will save to the "
                    << "process local path, as ./rank_id/node_dump/...";
    path = "./rank_" + std::to_string(rank_id) + kSuffix;
  } else {
    path = ms_om_path + "/rank_" + std::to_string(rank_id) + kSuffix;
  }
  return path;
}

#ifndef ENABLE_SECURITY
void AscendKernelRuntime::DumpTaskExceptionInfo(const session::KernelGraph & /* graph */) {
  const std::string path = GetDumpPath();
  if (access(path.c_str(), F_OK) == 0) {
    if (!DeleteDumpDir(path)) {
      MS_LOG(ERROR) << "Delete dump directory " << path << " failed";
    }
  }
  for (const auto &task_fail_info : task_fail_infoes_) {
    MS_LOG(ERROR) << "Task fail infos task_id: " << task_fail_info.taskid << ", stream_id: " << task_fail_info.streamid
                  << ", tid: " << task_fail_info.tid << ", device_id: " << task_fail_info.deviceid
                  << ", retcode: " << task_fail_info.retcode << " (" << GetErrorMsg(task_fail_info.retcode) << ")";
    auto node = AscendKernelRuntime::GetErrorNodeName(task_fail_info.streamid, task_fail_info.taskid);
    // Dump error data in local path
    if (node == nullptr) {
      continue;
    }
    auto full_scope_name = node->fullname_with_scope();
    MS_LOG(ERROR) << "Dump node (" << full_scope_name << ") task error input/output data to: " << path
                  << trace::DumpSourceLines(node, false);

    // full_scope_name: Default/GetNext-op1
    std::string lower_full_scope_name(full_scope_name.length(), ' ');
    (void)std::transform(full_scope_name.begin(), full_scope_name.end(), lower_full_scope_name.begin(), ::tolower);
    if (lower_full_scope_name.find("getnext") != std::string::npos) {
      MS_LOG(WARNING) << "GetNext error may be caused by slow data processing (bigger than 20s / batch) or "
                      << "transfer data to device error.";
      MS_LOG(WARNING) << "Suggestion: ";
      MS_LOG(WARNING) << "    1) Set the parameter dataset_sink_mode=False of model.train(...) or "
                      << "model.eval(...) and try again.";
      MS_LOG(WARNING) << "    2) Reduce the batch_size in data processing and try again.";
      MS_LOG(WARNING) << "    3) You can create iterator by interface create_dict_iterator() of dataset class to "
                      << "independently verify the performance of data processing without training. "
                      << "Refer to the link for data processing optimization suggestions: "
                      << "https://mindspore.cn/tutorials/experts/zh-CN/master/dataset/optimize.html";
    }

    E2eDump::DumpInputData(node, false, path, &full_scope_name);
    E2eDump::DumpOutputData(node, false, path, &full_scope_name);
  }
}
#endif

bool AscendKernelRuntime::Run(const session::KernelGraph &graph, bool is_task_sink) {
  const uint64_t kUSecondInSecond = 1000000;
  SignalGuard sg(IntHandler);
  bool ret = false;

  if (is_task_sink) {
#if defined(_WIN32) || defined(_WIN64)
    auto start_time = std::chrono::steady_clock::now();
#else
    struct timeval start_time {};
    struct timeval end_time {};
    (void)gettimeofday(&start_time, nullptr);
#endif
    ret = RunTask(graph);
#if defined(_WIN32) || defined(_WIN64)
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::ratio<1, kUSecondInSecond>> cost = end_time - start_time;
    MS_LOG(INFO) << "Call MS Run Success in " << cost.count() << " us";
#else
    (void)gettimeofday(&end_time, nullptr);
    uint64_t cost = kUSecondInSecond * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
    cost += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
    MS_LOG(INFO) << "Call MS Run Success in " << cost << " us";
#endif
  } else {
    ret = LaunchKernels(graph);
  }

  return ret;
}

void AscendKernelRuntime::GetLastNodesOnStream(const std::vector<CNodePtr> &kernels,
                                               std::vector<size_t> *stream_last_nodes) const {
  std::map<size_t, size_t> last_kernel;
  for (size_t i = 0; i < kernels.size(); ++i) {
    const auto stream_id = AnfAlgo::GetStreamId(kernels[i]);
    if (stream_id > 0) {
      last_kernel[stream_id] = i;
    }
  }
  (void)std::transform(last_kernel.begin(), last_kernel.end(), std::back_inserter(*stream_last_nodes),
                       [](const std::pair<size_t, size_t> &item) { return item.second; });
}

void AscendKernelRuntime::GetShadowBackendNodeMap(const session::KernelGraph &graph,
                                                  std::map<AnfNodePtr, AnfNodePtr> *shadow_backend_node_map) {
  auto input_nodes = graph.input_nodes();
  MS_EXCEPTION_IF_NULL(shadow_backend_node_map);
  for (auto &node : input_nodes) {
    auto front_node = AnfAlgo::FetchFrontNodeByBackendNode(node, graph);
    for (auto &knode : input_nodes) {
      if (knode == node) {
        break;
      }
      if (!common::AnfAlgo::IsTupleOutput(front_node) && front_node != nullptr &&
          front_node == AnfAlgo::FetchFrontNodeByBackendNode(knode, graph)) {
        (void)shadow_backend_node_map->emplace(node, knode);
        break;
      }
    }
  }
}

DeviceAddressPtr AscendKernelRuntime::GetInternalDeviceAddress(const session::KernelGraph &graph,
                                                               const AnfNodePtr &node) {
  auto front_node = graph.GetFrontNodeByInternalParameter(node);
  if (front_node.first == nullptr) {
    return nullptr;
  }
  auto pre_graphs = graph.get_pre_graphs();
  for (auto pre_graph_item : pre_graphs) {
    auto pre_graph = pre_graph_item.second.lock();
    MS_EXCEPTION_IF_NULL(pre_graph);
    auto graph_output = pre_graph->GetGraphOutputByFrontNode(front_node);
    if (graph_output.first == nullptr) {
      continue;
    }
    if (!AnfAlgo::OutputAddrExist(graph_output.first, graph_output.second)) {
      return nullptr;
    }
    auto output_device_address = AnfAlgo::GetMutableOutputAddr(graph_output.first, graph_output.second);
    MS_EXCEPTION_IF_NULL(output_device_address);
    if (output_device_address->GetDeviceType() == DeviceType::kAscend) {
      return output_device_address;
    }
  }
  return nullptr;
}

void AscendKernelRuntime::GenKernelEvents(const session::KernelGraph &graph) {
  auto &kernels = graph.execution_order();
  if (kernels.empty() || graph_kernel_events_map_.find(graph.graph_id()) != graph_kernel_events_map_.end()) {
    return;
  }
  std::vector<size_t> stream_last_nodes;
  GetLastNodesOnStream(kernels, &stream_last_nodes);
  auto kernel_events = std::pair<std::map<AnfNodePtr, std::vector<std::function<void()>>>,
                                 std::map<AnfNodePtr, std::vector<std::function<void()>>>>();
  auto &kernel_pre_run_events = kernel_events.first;
  auto &kernel_post_run_events = kernel_events.second;
  auto stream_num = kWorldGroupStreamIndex + 1;
  std::vector<std::vector<bool>> kernel_hit(kernels.size(), std::vector<bool>(stream_num, false));
  for (size_t i = 0; i < kernels.size(); ++i) {
    auto &kernel = kernels[i];
    auto curr_stream_id = AnfAlgo::GetStreamId(kernel);
    auto wait_stream = AscendStreamMng::GetInstance().GetStream(curr_stream_id);
    MS_EXCEPTION_IF_NULL(wait_stream);
    std::vector<bool> stream_hit(stream_num, false);
    std::vector<AnfNodePtr> used_kernels;
    std::set<AnfNodePtr> visited_kernels;
    common::AnfAlgo::GetAllVisitedCNode(kernel, &used_kernels, &visited_kernels);
    bool found_depend = false;
    for (int k = SizeToInt(i) - 1; k >= 0; --k) {
      auto pre_cnode = kernels[IntToSize(k)];
      auto pre_cnode_stream_id = AnfAlgo::GetStreamId(pre_cnode);
      if (pre_cnode_stream_id == curr_stream_id) {
        found_depend = true;
        continue;
      }
      for (auto &visited : used_kernels) {
        if (visited == pre_cnode && !stream_hit[pre_cnode_stream_id] && !kernel_hit[IntToSize(k)][curr_stream_id]) {
          stream_hit[pre_cnode_stream_id] = true;
          kernel_hit[IntToSize(k)][curr_stream_id] = true;
          found_depend = true;
          auto record_stream = AscendStreamMng::GetInstance().GetStream(pre_cnode_stream_id);
          MS_EXCEPTION_IF_NULL(record_stream);
          auto event = CreateDeviceEvent();
          event->set_wait_stream(wait_stream);
          event->set_record_stream(record_stream);
          (void)kernel_post_run_events[pre_cnode].emplace_back([event]() { event->RecordEvent(); });
          (void)kernel_pre_run_events[kernel].emplace_back([event]() { event->WaitEvent(); });
        }
      }
    }
    if (!found_depend && wait_stream != stream_) {
      auto pre_event = CreateDeviceEvent();
      pre_event->set_wait_stream(wait_stream);
      pre_event->set_record_stream(stream_);
      (void)kernel_pre_run_events[kernel].emplace_back([pre_event]() { pre_event->RecordEvent(); });
      (void)kernel_pre_run_events[kernel].emplace_back([pre_event]() { pre_event->WaitEvent(); });
    }
  }
  ProcessBoundaryEvent(kernels, &kernel_post_run_events, stream_last_nodes);
  graph_kernel_events_map_[graph.graph_id()] = std::move(kernel_events);
}

void AscendKernelRuntime::ProcessBoundaryEvent(
  const std::vector<CNodePtr> &kernels, std::map<AnfNodePtr, std::vector<std::function<void()>>> *kernel_run_events,
  const std::vector<size_t> &last_stream_nodes) {
  for (auto &i : last_stream_nodes) {
    if (i >= kernels.size()) {
      MS_LOG(ERROR) << "Node index exceed kernel size.";
      continue;
    }
    auto &kernel = kernels[i];
    MS_EXCEPTION_IF_NULL(kernel);
    bool found_nearest_child = false;
    for (size_t j = i + 1; j < kernels.size(); ++j) {
      auto &child = kernels[j];
      MS_EXCEPTION_IF_NULL(child);
      auto input_size = child->inputs().size() - 1;
      for (size_t k = 0; k < input_size; ++k) {
        auto kernel_index =
          common::AnfAlgo::VisitKernelWithReturnType(common::AnfAlgo::GetInputNode(child, k), 0, true);
        if (kernel_index.first == kernel) {
          found_nearest_child = true;
          break;
        }
      }
      if (found_nearest_child) {
        break;
      }
    }
    if (!found_nearest_child) {
      auto post_event = CreateDeviceEvent();
      MS_EXCEPTION_IF_NULL(post_event);
      auto id = AnfAlgo::GetStreamId(kernel);
      auto record_stream = AscendStreamMng::GetInstance().GetStream(id);
      MS_EXCEPTION_IF_NULL(record_stream);
      post_event->set_wait_stream(stream_);
      post_event->set_record_stream(record_stream);
      (void)(*kernel_run_events)[kernel].emplace_back([post_event]() { post_event->RecordEvent(); });
      (void)(*kernel_run_events)[kernel].emplace_back([post_event]() { post_event->WaitEvent(); });
    }
  }
}

bool AscendKernelRuntime::RunDynamicKernelAsync(const session::KernelGraph &graph) {
  MS_LOG(INFO) << "RunExecutorAsync start. GraphId:" << graph.graph_id();
  AscendEnableDynamicRuntimeCache(&graph);

  const auto &kernels = graph.execution_order();
  for (size_t i = 0; i < kernels.size(); ++i) {
    auto &kernel = kernels[i];
    MS_EXCEPTION_IF_NULL(kernel);
    if (common::AnfAlgo::GetCNodeName(kernel) == kDynamicAtomicAddrCleanOpName) {
      continue;
    }
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto depends = abstract::GetValueDependArgIndices(kernel);
    if (!depends.empty() || AnfAlgo::GetKernelType(kernel) == KernelType::HCCL_KERNEL) {
      MS_LOG(INFO) << "Match Dynamic Kernel, Start SyncStream";
      if (!SyncStream()) {
        MS_LOG(ERROR) << "SyncStream failed";
        return false;
      }
    }

    if (common::AnfAlgo::IsDynamicShape(kernel)) {
      opt::dynamic_shape::InferOp(kernel);
      auto args = kernel->user_data<kernel::KernelArgs>();
      MS_EXCEPTION_IF_NULL(args);
      (void)kernel_mod->Resize(args->op, args->inputs, args->outputs, args->depend_tensor_map);
    }
    KernelLaunchInfo kernel_launch_info;
    device::KernelRuntime::GenLaunchArgs(*kernel_mod, kernel, &kernel_launch_info);
    // allocate workspace size
    std::vector<AddressPtr> workspace_addr;
    if (common::AnfAlgo::IsDynamicShape(kernel) && AnfAlgo::GetKernelType(kernel) == KernelType::TBE_KERNEL) {
      auto workspace_size_list = kernel_mod->GetWorkspaceSizeList();
      auto ms_context = MsContext::GetInstance();
      MS_EXCEPTION_IF_NULL(ms_context);
      auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
      auto runtime_instance = KernelRuntimeManager::Instance().GetSingleKernelRuntime(kAscendDevice, device_id);
      MS_EXCEPTION_IF_NULL(runtime_instance);

      for (auto size : workspace_size_list) {
        auto device_address_ptr = std::make_shared<AscendDeviceAddress>(nullptr, size, kAscendDevice, device_id);
        MS_EXCEPTION_IF_NULL(device_address_ptr);
        device_address_ptr->set_is_ptr_persisted(true);
        auto device_ptr = runtime_instance->MallocMem(MemType::kDynamicMem, size, device_address_ptr);
        if (device_ptr == nullptr) {
          MS_LOG(EXCEPTION) << "MallocMem from memory pool failed. Node info :" << kernel->fullname_with_scope();
        }

        AddressPtr workspace_addr_ptr =
          std::make_shared<kernel::Address>(device_address_ptr->GetMutablePtr(), device_address_ptr->GetSize());
        (void)workspace_addr.emplace_back(workspace_addr_ptr);
      }
    } else {
      workspace_addr = kernel_launch_info.workspaces_;
    }

    auto ret = kernel_mod->Launch(kernel_launch_info.inputs_, workspace_addr, kernel_launch_info.outputs_, stream_);
    if (!ret) {
      MS_LOG(ERROR) << "Launch kernel failed, kernel full name: " << kernel->fullname_with_scope();
      return false;
    }
    if (common::AnfAlgo::IsDynamicShape(kernel)) {
      kernel::UpdateNodeShape(kernel);
    }
  }

  if (!SyncStream()) {
    MS_LOG(ERROR) << "SyncStream failed";
    return false;
  }

  return true;
}

bool AscendKernelRuntime::RunTask(const session::KernelGraph &graph) {
  current_graph_ = &graph;
  SetCurrentContext();
  if (graph.is_dynamic_shape()) {
    MS_LOG(INFO) << "Dynamic Shape Graph Run Task Async";
    return RunDynamicKernelAsync(graph);
  }

  MS_LOG(INFO) << "RunTask start. GraphId:" << graph.graph_id();

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (GraphWithEmptyTaskList(graph)) {
    MS_LOG(INFO) << "RunTask end, no task info found";
    return true;
  }

  if (!CheckGraphIdValid(graph.graph_id())) {
    MS_LOG(ERROR) << "GraphId:" << graph.graph_id() << " Invalid! Graph RunTask without GenTask.";
    return false;
  }

  if (!rt_model_zero_copy_.UpdateTaskArgs(graph, compute_stream())) {
    MS_LOG(ERROR) << "Update rtModel task args failed, graph id " << graph.graph_id();
    return false;
  }

  try {
    ModelRunner::Instance().RunModel(graph.graph_id());
  } catch (const std::exception &) {
    const auto &exec_order = graph.execution_order();
    for (const auto &node : exec_order) {
      if (!IsPrimitiveCNode(node, prim::kPrimLabelSwitch)) {
        continue;
      }

      size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
      for (size_t i = 0; i < input_num; ++i) {
        auto real_input_index = AnfAlgo::GetInputGraphIdxByKernelIdx(node, i);
        auto device_address = AnfAlgo::GetPrevNodeOutputAddr(node, real_input_index);
        MS_EXCEPTION_IF_NULL(device_address);
        MS_LOG(INFO) << "Input idx " << i << " size " << device_address->size_ << " addr " << device_address->ptr_;
        int32_t value = 0;
        auto ret =
          rtMemcpy(&value, sizeof(int32_t), device_address->ptr_, device_address->size_, RT_MEMCPY_DEVICE_TO_HOST);
        if (ret == RT_ERROR_NONE) {
          MS_LOG(INFO) << "Value = " << value;
        }
      }
    }
#ifndef ENABLE_SECURITY
    DumpTaskExceptionInfo(graph);
#endif
#ifdef WITH_BACKEND
    // Run task error, we should call TdtHostDestroy to release tdt to avoid DataQueueOp hostPush hung
    // case1: cpu usage 100% cause thread/process exit, but some tdt thread remain in backend
    if (!tdt_handle::DestroyHandle()) {
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
  SetCurrentContext();
  if (stream_ != nullptr) {
    // cppcheck-suppress unreadVariable
    auto lock = device::KernelRuntime::LockRuntime(stream_);
    if (!AscendStreamMng::GetInstance().SyncStream(stream_)) {
      MS_LOG(ERROR) << "Sync default stream failed.";
      return false;
    }
  }
  if (communication_stream_ != nullptr) {
    // cppcheck-suppress unreadVariable
    auto lock = device::KernelRuntime::LockRuntime(communication_stream_);
    if (!AscendStreamMng::GetInstance().SyncStream(communication_stream_)) {
      MS_LOG(ERROR) << "Sync default stream failed.";
      return false;
    }
  }
  return true;
}

bool AscendKernelRuntime::MemcpyAsync(void *dst, const void *src, uint64_t size, int32_t kind) {
  SetCurrentContext();
  if (size == 0) {
    MS_LOG(DEBUG) << "rtMemcpyAsync size is 0, copy kind:" << kind;
    return true;
  }
  if (stream_ == nullptr) {
    MS_LOG(ERROR) << "MemcpyAsync failed. stream_ is nullptr";
    return false;
  }

  auto copy_kind = static_cast<rtMemcpyKind_t>(kind);
  if (copy_kind != RT_MEMCPY_HOST_TO_DEVICE_EX && copy_kind != RT_MEMCPY_DEVICE_TO_DEVICE) {
    MS_LOG(EXCEPTION) << "Memory copy async not support cache host buffer in kind: " << kind;
  }
  if (dst == nullptr) {
    MS_LOG(ERROR) << "rtMemcpyAsync dst ptr is null, copy kind:" << kind;
    return false;
  }
  if (src == nullptr) {
    MS_LOG(ERROR) << "rtMemcpyAsync src ptr is null, copy kind:" << kind;
    return false;
  }
  // cppcheck-suppress unreadVariable
  auto lock = device::KernelRuntime::LockRuntime(stream_);
  if (RT_ERROR_NONE != rtMemcpyAsync(dst, size, src, size, static_cast<rtMemcpyKind_t>(kind), stream_)) {
    MS_LOG(ERROR) << "Call runtime rtMemcpyAsync error.";
    return false;
  }
  return true;
}

void AscendKernelRuntime::CreateContext() {
  if (rt_context_ == nullptr) {
    auto ret = rtCtxCreate(&rt_context_, 0, UintToInt(device_id_));
    if (ret != RT_ERROR_NONE) {
      MS_EXCEPTION(DeviceProcessError) << "Call rtCtxCreate, ret[" << static_cast<int>(ret) << "]";
    }
  }
  SetCurrentContext();
}

void AscendKernelRuntime::SetRtDevice(uint32_t device_id) {
  MS_LOG(INFO) << "Enter SetRtDevice, current initialize device number:" << initialized_device_set_.size();
  if (initialized_device_set_.find(device_id) != initialized_device_set_.end()) {
    MS_LOG(INFO) << "Device " << device_id << " has been set";
  }

  int device_count = 0;
  auto ret = rtGetDeviceCount(&device_count);
  if (ret != RT_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "Call rtGetDeviceCount, ret[" << static_cast<int>(ret) << "]";
  }

  ret = rtSetDevice(UintToInt(device_id));
  if (ret != RT_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "Call rtSetDevice, ret[" << static_cast<int>(ret) << "]";
  }
  (void)initialized_device_set_.insert(device_id);
}

void AscendKernelRuntime::CreateDefaultStream() {
  size_t compute_stream_id;
  AscendStreamMng::GetInstance().CreateStreamWithFlags(&compute_stream_id, RT_STREAM_HUGE);
  MS_LOG(INFO) << "Create ascend default stream, stream id: " << compute_stream_id;
  stream_ = AscendStreamMng::GetInstance().GetStream(compute_stream_id);
  MS_EXCEPTION_IF_NULL(stream_);

  size_t communication_stream_id;
  AscendStreamMng::GetInstance().CreateStream(&communication_stream_id);
  MS_LOG(INFO) << "Create ascend communication stream, stream id: " << communication_stream_id;
  communication_stream_ = AscendStreamMng::GetInstance().GetStream(communication_stream_id);
  MS_EXCEPTION_IF_NULL(communication_stream_);
}

bool AscendKernelRuntime::InitDevice() {
  SetRtDevice(device_id_);

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr == nullptr) {
    MS_LOG(ERROR) << "Get MsContext instance failed";
    return false;
  }

  // Context will be created by rtSetDevice
  const auto rt_ret = rtCtxGetCurrent(&rt_context_);
  if (rt_ret != RT_ERROR_NONE || rt_context_ == nullptr) {
    MS_LOG(ERROR) << "Call rtCtxGetCurrent failed, ret[" << rt_ret << "]";
    return false;
  }

  CreateDefaultStream();
  return true;
}

bool AscendKernelRuntime::ResetDevice(uint32_t device_id) {
  SetCurrentContext();
  AscendStreamMng::GetInstance().DestroyAllRtEvents();
  if (!AscendStreamMng::GetInstance().DestroyAllStreams()) {
    MS_LOG(ERROR) << "Fail to destroy all streams when reset device.";
    return false;
  }
  stream_ = nullptr;
  communication_stream_ = nullptr;

  if (initialized_device_set_.find(device_id) != initialized_device_set_.end()) {
    auto ret = rtDeviceReset(UintToInt(device_id));
    if (ret != RT_ERROR_NONE) {
      MS_EXCEPTION(DeviceProcessError) << "Call rtDeviceReset, ret[" << ret << "]";
    }
    (void)initialized_device_set_.erase(device_id);
  }

  // set to nullptr as its not created, only bounded to existing context
  rt_context_ = nullptr;
  return true;
}

bool AscendKernelRuntime::DestroyHccl() {
  if (!NeedDestroyHccl()) {
    MS_LOG(INFO) << "Hccl is not enable, no need to close.";
    return true;
  }
  if (!AscendCollectiveCommLib::GetInstance().DestroyHcclComm()) {
    MS_LOG(ERROR) << "Hccl destroy failed.";
    return false;
  }
  MS_LOG(INFO) << "Hccl destroy successful.";
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  context_ptr->set_param<bool>(MS_CTX_ENABLE_HCCL, false);
  return true;
}

bool AscendKernelRuntime::GraphWithEmptyTaskList(const session::KernelGraph &graph) const {
  auto iter = task_map_.find(graph.graph_id());
  if (iter == task_map_.end()) {
    MS_LOG(EXCEPTION) << "Unknown graph ptr";
  }
  return iter->second.empty();
}

bool AscendKernelRuntime::CheckGraphIdValid(GraphId graph_id) const {
  return task_map_.find(graph_id) != task_map_.end() && graph_model_map_.find(graph_id) != graph_model_map_.end();
}

void AscendKernelRuntime::KernelLaunchProfiling(const std::string &kernel_name) {
#ifndef ENABLE_SECURITY
  auto profiler_manager = profiler::ProfilerManager::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_manager);
  if (!profiler_manager->GetProfilingEnableFlag()) {
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
#endif
}

std::shared_ptr<DeviceEvent> AscendKernelRuntime::CreateDeviceEvent() {
  auto ascend_event = std::make_shared<AscendEvent>();
  MS_EXCEPTION_IF_NULL(ascend_event);
  return ascend_event;
}

std::shared_ptr<DeviceEvent> AscendKernelRuntime::CreateDeviceTimeEvent() {
  auto ascend_time_event = std::make_shared<AscendTimeEvent>();
  MS_EXCEPTION_IF_NULL(ascend_time_event);
  return ascend_time_event;
}

uint64_t AscendKernelRuntime::GetAvailableMemMaxSize() const {
  auto ascend_mem_manager = std::dynamic_pointer_cast<AscendMemoryManager>(mem_manager_);
  MS_EXCEPTION_IF_NULL(ascend_mem_manager);
  return ascend_mem_manager->GetMsMaxMemSize();
}

uint64_t AscendKernelRuntime::GetMsUsedHbmSize() const {
  auto ascend_mem_manager = std::dynamic_pointer_cast<AscendMemoryManager>(mem_manager_);
  MS_EXCEPTION_IF_NULL(ascend_mem_manager);
  return ascend_mem_manager->GetMsUsedHbmSize();
}

bool AscendKernelRuntime::DeleteDumpDir(const std::string &path) {
  string real_path = GetRealPath(path);
  if (DeleteDumpFile(real_path) == -1) {
    return false;
  }
  if (rmdir(real_path.c_str()) == -1) {
    MS_LOG(WARNING) << "Delete dir " << real_path << " failed!";
  }
  return true;
}

int AscendKernelRuntime::DeleteDumpFile(std::string path) {
  DIR *dir;
  struct dirent *dirinfo;
  struct stat statbuf;
  string filepath;
  int result = 0;
  if (lstat(path.c_str(), &statbuf) != 0) {
    return -1;
  }

  if (S_ISREG(statbuf.st_mode)) {
    result = remove(path.c_str());
  } else if (S_ISDIR(statbuf.st_mode)) {
    if ((dir = opendir(path.c_str())) == nullptr) {
      return -1;
    }

    while (result == 0) {
      dirinfo = readdir(dir);
      if (dirinfo == nullptr) {
        break;
      }
      if (path[path.size() - 1] != '/') {
        path = path + "/";
      }
      filepath = path + dirinfo->d_name;
      if (strcmp(dirinfo->d_name, ".") == 0 || strcmp(dirinfo->d_name, "..") == 0) {
        continue;
      }
      result = DeleteDumpFile(filepath);
      if (result == 0) {
        if (rmdir(filepath.c_str()) == -1) {
          MS_LOG(WARNING) << "Delete dir " << filepath << " failed!";
        }
      }
    }
    if (closedir(dir) == -1) {
      MS_LOG(WARNING) << "Dump dir " << path << " close failed!";
    }
  }
  return result;
}

std::string AscendKernelRuntime::GetRealPath(const std::string &path) {
  char real_path_mem[kPathMax] = {0};
  char *real_path_ret = realpath(path.c_str(), real_path_mem);
  if (real_path_ret == nullptr) {
    return "";
  }
  return std::string(real_path_mem);
}

void AscendKernelRuntime::SetReuseCommunicationAddress(const session::KernelGraph &graph) {
  auto cnode_list = graph.execution_order();
  for (const auto &cnode : cnode_list) {
    MS_EXCEPTION_IF_NULL(cnode);
    if (common::AnfAlgo::HasNodeAttr(kAttrReuseCommunication, cnode)) {
      auto reuse_index = common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrReuseCommunication);
      if (reuse_communication_address_.find(reuse_index) == reuse_communication_address_.end()) {
        (void)reuse_communication_address_.emplace(reuse_index, std::make_pair(nullptr, nullptr));
      }
    }
  }
}
}  // namespace mindspore::device::ascend
