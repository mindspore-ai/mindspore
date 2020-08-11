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
#include <exception>
#include <algorithm>
#include "runtime/device/ascend/ascend_device_address.h"
#include "runtime/device/cpu/mpi/mpi_interface.h"
#include "utils/ms_context.h"
#include "utils/context/context_extends.h"
#include "utils/mpi/mpi_config.h"
#include "runtime/device/ascend/profiling/profiling_manager.h"
#include "hccl/hcom.h"
#include "common/trans.h"
#include "runtime/context.h"
#include "runtime/device/ascend/ascend_label_assign.h"
#include "runtime/device/ascend/ascend_stream_assign.h"
#include "runtime/device/ascend/ascend_memory_pool.h"
#include "framework/ge_runtime/model_runner.h"
#include "runtime/device/ascend/tasksink/task_generator.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "runtime/device/ascend/profiling/profiling_utils.h"
#include "backend/kernel_compiler/tbe/tbe_utils.h"
#include "backend/optimizer/mem_reuse/mem_reuse_checker.h"
#include "runtime/device/ascend/ascend_memory_manager.h"
#include "debug/tensor_load.h"

using ge::model_runner::ModelRunner;
using mindspore::device::ascend::ProfilingManager;
using mindspore::device::ascend::ProfilingUtils;
using mindspore::device::ascend::tasksink::TaskGenerator;
using mindspore::kernel::tbe::TbeUtils;
using std::vector;

constexpr uint32_t kTupleTaskId = 0;
constexpr uint32_t kTupleStreamId = 1;
constexpr uint32_t kTupleArgs = 2;

namespace mindspore {
namespace device {
namespace ascend {
static const size_t PRAMATER_OUTPUT_INDEX = 0;
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

AscendKernelRuntime::~AscendKernelRuntime() { graph_model_map_.clear(); }

void AscendKernelRuntime::ClearGraphModelMap() {
  for (auto &iter : graph_data_dumper_) {
    MS_LOG(INFO) << "[DataDump] Unload data dumper:" << iter.first;
    auto &data_dumper = iter.second;
    MS_EXCEPTION_IF_NULL(data_dumper);
    data_dumper->UnloadDumpInfo();
    data_dumper->OpDebugUnregister();
  }
  graph_data_dumper_.clear();
  // tell users which dump kernel name not used
  DataDumpParser::GetInstance().PrintUnusedKernel();

  for (auto &iter : graph_model_map_) {
    MS_LOG(INFO) << "Ge UnloadModel " << iter.first;
    auto ret = ModelRunner::Instance().UnloadModel(iter.first);
    if (!ret) {
      MS_LOG(ERROR) << "UnloadModel failed";
    }
  }
}

void AscendKernelRuntime::ClearGraphRuntimeResource(uint32_t graph_id) {
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

bool AscendKernelRuntime::NeedDestroyHccl() {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (!context_ptr->enable_hccl()) {
    MS_LOG(INFO) << "Hccl is not enabled";
    return false;
  }
  // Note: make sure hcom_connectivity_detection api never be used.
  return true;
}

void AscendKernelRuntime::ReleaseDeviceRes() {
  MS_LOG(INFO) << "Ascend finalize start";
  // release ge runtime
  ClearGraphModelMap();

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto ret = rtSetDevice(context_ptr->device_id());
  if (ret != RT_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "Call rtSetDevice, ret[" << static_cast<int>(ret) << "]";
  }

  if (mem_manager_ != nullptr) {
    mem_manager_->FreeDeviceMemory();
  }

  (void)DestroyHccl();
  (void)ResetDevice();
  (void)ProfilingManager::GetInstance().StopProfiling();
  MS_LOG(INFO) << "Ascend finalize end";
}

bool AscendKernelRuntime::Init() {
  if (initialized_) {
    return true;
  }
  bool ret = false;
#ifdef ENABLE_DUMP_E2E
  ret = SetDumpConf();
  if (!ret) {
    MS_LOG(INFO) << "No dump conf to set!";
  }
#endif

  DataDumpParser::GetInstance().ParseDumpConfig();

  // Start up profiling before rtSetDevice
  ret = ProfilingManager::GetInstance().StartupProfiling(device_id_);
  if (!ret) {
    MS_EXCEPTION(DeviceProcessError) << "StartupProfiling failed.";
  }

  ret = InitDevice();
  if (!ret) {
    return ret;
  }
  mem_manager_ = std::make_shared<AscendMemoryManager>();
  MS_EXCEPTION_IF_NULL(mem_manager_);
  mem_manager_->MallocDeviceMemory();

  initialized_ = true;
  return ret;
}

#ifdef ENABLE_DUMP_E2E
namespace {
void DumpOutput(mindspore::session::KernelGraph *graph, const string &dump_path, DumpConfPtr dump_conf) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(dump_conf);
  bool trans_flag = dump_conf->trans_flag();
  const auto &apply_kernels = graph->execution_order();
  for (const auto &node : apply_kernels) {
    MS_EXCEPTION_IF_NULL(node);
    auto node_name = AnfAlgo::GetCNodeName(node);
    std::string kernel_name = node->fullname_with_scope();
    if (!dump_conf->IsKernelNeedDump(kernel_name)) {
      continue;
    }
    const std::string strsrc = "/";
    const std::string strdst = "--";
    std::string::size_type pos = 0;
    std::string::size_type srclen = strsrc.size();
    std::string::size_type dstlen = strdst.size();
    while ((pos = kernel_name.find(strsrc, pos)) != std::string::npos) {
      kernel_name.replace(pos, srclen, strdst);
      pos += dstlen;
    }
    auto output_size = AnfAlgo::GetOutputTensorNum(node);
    for (size_t j = 0; j < output_size; ++j) {
      auto addr = AnfAlgo::GetOutputAddr(node, j);
      std::vector<int> int_shapes;
      if (trans_flag) {
        int_shapes = trans::GetRuntimePaddingShape(node, j);
      } else {
        auto shape = AnfAlgo::GetOutputDeviceShape(node, j);
        (void)std::transform(shape.begin(), shape.end(), std::back_inserter(int_shapes),
                             [](size_t inner_item) { return SizeToInt(inner_item); });
      }
      auto type = AnfAlgo::GetOutputInferDataType(node, j);
      auto format = kOpFormat_DEFAULT;
      string filepath = dump_path + '/' + kernel_name + '_' + "output_" + std::to_string(j);
      auto ascend_addr = dynamic_cast<const mindspore::device::ascend::AscendDeviceAddress *>(addr);
      auto ret = ascend_addr->DumpMemToFile(trans_flag, filepath, format, int_shapes, type);
      if (!ret) {
        MS_LOG(ERROR) << "DumpMemToFile Failed: flag:" << trans_flag << ", path:" << filepath
                      << ", host_format:" << format << ".!";
      }
    }
  }
}

void DumpParameters(mindspore::session::KernelGraph *graph, const string &dump_path, DumpConfPtr dump_conf) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(dump_conf);
  bool trans_flag = dump_conf->trans_flag();
  const auto &parameters = graph->inputs();
  for (auto &item : parameters) {
    if (!item->isa<Parameter>()) {
      continue;
    }
    std::string parameter_name = item->fullname_with_scope();
    if (!dump_conf->IsKernelNeedDump(parameter_name)) {
      continue;
    }
    auto addr = AnfAlgo::GetOutputAddr(item, PRAMATER_OUTPUT_INDEX);
    std::vector<int> int_shapes;
    if (trans_flag) {
      int_shapes = trans::GetRuntimePaddingShape(item, PRAMATER_OUTPUT_INDEX);
    } else {
      auto shape = AnfAlgo::GetOutputDeviceShape(item, PRAMATER_OUTPUT_INDEX);
      (void)std::transform(shape.begin(), shape.end(), std::back_inserter(int_shapes),
                           [](size_t inner_item) { return SizeToInt(inner_item); });
    }
    auto type = AnfAlgo::GetOutputInferDataType(item, PRAMATER_OUTPUT_INDEX);
    auto format = kOpFormat_DEFAULT;
    string filepath = dump_path + '/' + parameter_name + '_' + "output_0";
    auto ascend_addr = dynamic_cast<const mindspore::device::ascend::AscendDeviceAddress *>(addr);
    auto ret = ascend_addr->DumpMemToFile(trans_flag, filepath, format, int_shapes, type);
    if (!ret) {
      MS_LOG(ERROR) << "DumpMemToFile Failed: flag:" << trans_flag << ", path:" << filepath
                    << ", host_format:" << format << ".!";
    }
  }
}
}  // namespace
#endif

bool AscendKernelRuntime::DumpData(mindspore::session::KernelGraph *graph, Debugger *debugger) {
  MS_EXCEPTION_IF_NULL(graph);
#ifdef ENABLE_DUMP_E2E
  MS_LOG(INFO) << "Start dump step";
  DumpConfPtr dump_conf = GetDumpConf();
  MS_EXCEPTION_IF_NULL(dump_conf);
  dump_conf->UpdataCurIter();
  bool dump_flag = dump_conf->dump_enable();
  if (!dump_flag) {
    MS_LOG(INFO) << "Dump flag is disable, pass dump step";
    return true;
  }
  uint32_t cur_iter = dump_conf->cur_iter();
  if (dump_conf->dump_iter() != 0) {
    if (cur_iter != dump_conf->dump_iter()) {
      return true;
    }
  }
  MS_LOG(INFO) << "Cur iter is " << cur_iter;
  std::string net_name = dump_conf->dump_net_name();
  std::string iterator = to_string(cur_iter);
  std::string dump_path = dump_conf->dump_path();
  if (dump_path.back() == '/') {
    dump_path = dump_path + net_name + '/' + iterator;
  } else {
    dump_path = dump_path + '/' + net_name + '/' + iterator;
  }
  // dump output
  DumpOutput(graph, dump_path, dump_conf);
  // dump parameters
  DumpParameters(graph, dump_path, dump_conf);
#endif
  return true;
}

#ifdef ENABLE_DEBUGGER
namespace {
void LoadOutput(mindspore::session::KernelGraph *graph, Debugger *debugger) {
  MS_EXCEPTION_IF_NULL(graph);
  // trans_flag: "true" means tensor values will be transfered to host format, otherwise not.
  bool trans_flag = false;
  const auto &apply_kernels = graph->execution_order();
  // for kernels, execution order starts from 1
  int exec_order = 1;
  auto debugger_ = mindspore::Debugger::GetInstance();
  DebugServices *debug_services = debugger_->debug_services();
  auto watchpoint_table = debug_services->GetWatchpointTable();
  for (const auto &node : apply_kernels) {
    MS_EXCEPTION_IF_NULL(node);
    auto node_name = AnfAlgo::GetCNodeName(node);
    std::string kernel_name = node->fullname_with_scope();
    auto output_size = AnfAlgo::GetOutputTensorNum(node);
    if (debugger_->partial_memory()) {
      if (!debug_services->IsWatchPoint(kernel_name, watchpoint_table)) {
        continue;
      }
    }
    for (size_t j = 0; j < output_size; ++j) {
      auto addr = AnfAlgo::GetOutputAddr(node, j);
      auto type = AnfAlgo::GetOutputInferDataType(node, j);
      auto format = kOpFormat_DEFAULT;
      string tensor_name = kernel_name + ':' + std::to_string(j);
      auto ascend_addr = dynamic_cast<const mindspore::device::ascend::AscendDeviceAddress *>(addr);
      std::vector<int> int_shapes;
      if (trans_flag) {
        int_shapes = trans::GetRuntimePaddingShape(node, j);
      } else {
        auto shape = AnfAlgo::GetOutputDeviceShape(node, j);
        (void)std::transform(shape.begin(), shape.end(), std::back_inserter(int_shapes),
                             [](size_t inner_item) { return SizeToInt(inner_item); });
      }
      auto ret =
        ascend_addr->LoadMemToHost(trans_flag, tensor_name, exec_order, format, int_shapes, type, j, debugger, false);
      if (!ret) {
        MS_LOG(ERROR) << "LoadMemToHost: flag:" << trans_flag << ", tensor_name:" << tensor_name
                      << ", host_format:" << format << ".!";
      }
    }
    exec_order = exec_order + 1;
  }
}

void LoadParameters(mindspore::session::KernelGraph *graph, Debugger *debugger) {
  MS_EXCEPTION_IF_NULL(graph);
  // trans_flag: "true" means tensor values will be transfered to host format, otherwise not.
  bool trans_flag = false;
  const auto &parameters = graph->inputs();
  // for parameters, set its execution order to be 0;
  int exec_order = 0;
  for (auto &item : parameters) {
    if (!item->isa<Parameter>()) {
      continue;
    }
    std::string parameter_name = item->fullname_with_scope();
    auto addr = AnfAlgo::GetOutputAddr(item, PRAMATER_OUTPUT_INDEX);
    auto type = AnfAlgo::GetOutputInferDataType(item, PRAMATER_OUTPUT_INDEX);
    auto format = kOpFormat_DEFAULT;
    string tensor_name = parameter_name + ':' + "0";
    auto ascend_addr = dynamic_cast<const mindspore::device::ascend::AscendDeviceAddress *>(addr);
    std::vector<int> int_shapes;
    if (trans_flag) {
      int_shapes = trans::GetRuntimePaddingShape(item, PRAMATER_OUTPUT_INDEX);
    } else {
      auto shape = AnfAlgo::GetOutputDeviceShape(item, PRAMATER_OUTPUT_INDEX);
      (void)std::transform(shape.begin(), shape.end(), std::back_inserter(int_shapes),
                           [](size_t inner_item) { return SizeToInt(inner_item); });
    }
    auto ret =
      ascend_addr->LoadMemToHost(trans_flag, tensor_name, exec_order, format, int_shapes, type, 0, debugger, true);
    if (!ret) {
      MS_LOG(ERROR) << "LoadMemToHost Failed: flag:" << trans_flag << ", path:" << tensor_name
                    << ", host_format:" << format << ".!";
    }
  }
}
}  // namespace
#endif

bool AscendKernelRuntime::LoadData(mindspore::session::KernelGraph *graph, Debugger *debugger) {
  MS_EXCEPTION_IF_NULL(graph);
#ifdef ENABLE_DEBUGGER
  MS_LOG(INFO) << "Start load step";
  uint32_t cur_iter = 0;
  MS_LOG(INFO) << "Cur iter is " << cur_iter;
  // load output
  LoadOutput(graph, debugger);
  // load parameters
  LoadParameters(graph, debugger);
#endif
  return true;
}

bool AscendKernelRuntime::NodeOutputDeviceAddressExist(const AnfNodePtr &kernel, size_t index) {
  if (AnfAlgo::OutputAddrExist(kernel, index)) {
    auto address = AnfAlgo::GetOutputAddr(kernel, index);
    MS_EXCEPTION_IF_NULL(address);
    return address->DeviceType() == DeviceAddressType::kAscend;
  }
  return false;
}

DeviceAddressPtr AscendKernelRuntime::CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format,
                                                          TypeId type_id) {
  return std::make_shared<AscendDeviceAddress>(device_ptr, device_size, format, type_id);
}

bool AscendKernelRuntime::GenTask(const session::KernelGraph *graph) {
  if (graph == nullptr) {
    MS_EXCEPTION(NotExistsError) << "session::KernelGraph is NULL!";
  }
  MS_LOG(INFO) << "GenTask start. GraphId:" << graph->graph_id();
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool is_task_sink = context_ptr->enable_task_sink();
  if (!is_task_sink) {
    return true;
  }
#ifdef MEM_REUSE_DEBUG
  if (!context_ptr->enable_mem_reuse()) {
    // Get normal graph ir for memreuse
    mindspore::memreuse::MemReuseChecker::GetInstance().CheckNormalIR(graph);
  }
#endif
  vector<std::shared_ptr<TaskInfo>> task_info_list;
  auto anf_node_list = graph->execution_order();
  TaskGenerator::GenTasks(anf_node_list, &task_info_list, graph->graph_id());
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
  AscendLabelAssign &label_assign_instance = AscendLabelAssign::GetInstance();
  // the streams' flag not HEAD_STREAM
  std::vector<uint32_t> wait_active_stream_list;
  assign_instance.GetWaitStreams(&wait_active_stream_list);
  std::vector<uint32_t> force_copy_stream_list;
  assign_instance.GetHcomStreams(&force_copy_stream_list);
  MS_LOG(INFO) << "Call DavinciModel total stream num:" << resource_manager.get_cur_stream_num()
               << ", total event num:" << resource_manager.get_cur_event_num()
               << ", total label num:" << label_assign_instance.GetLabelNum(NOT_NULL(graph))
               << ", wait_active_stream_list size:" << wait_active_stream_list.size()
               << ", force_copy_stream_list size:" << force_copy_stream_list.size();
  std::vector<std::shared_ptr<ge::model_runner::OpInfo>> empty_list;
  auto model = std::make_shared<ge::model_runner::DavinciModel>(
    task_info_list, empty_list, empty_list, empty_list, empty_list, wait_active_stream_list, force_copy_stream_list, 0,
    0, 0, 0, 0, 0, resource_manager.get_cur_stream_num(), label_assign_instance.GetLabelNum(NOT_NULL(graph)),
    resource_manager.get_cur_event_num(), 0);
  auto ret = graph_model_map_.insert(std::make_pair(graph->graph_id(), model));
  if (!ret.second) {
    MS_LOG(EXCEPTION) << "Duplicate GraphId! Please check in ascend_session.";
  }
  MS_LOG(INFO) << "TaskGenerator GetTaskInfo end...";
  return true;
}

bool AscendKernelRuntime::LoadTask(const session::KernelGraph *graph) {
  if (graph == nullptr) {
    MS_EXCEPTION(NotExistsError) << "Null pointer graph, LoadTask failed. ";
  }
  MS_LOG(INFO) << "LoadTask start. GraphId:" << graph->graph_id();
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool is_task_sink = context_ptr->enable_task_sink();
  if (!is_task_sink) {
    return true;
  }

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
                                              NotNull<std::function<void *()>> model_handle) {
  if (!DataDumpParser::GetInstance().DumpEnabled()) {
    return;
  }
  auto data_dumper = std::make_shared<DataDumper>(graph.get(), model_handle);
  MS_EXCEPTION_IF_NULL(data_dumper);
  auto ret = graph_data_dumper_.try_emplace(graph->graph_id(), data_dumper);
  data_dumper->OpDebugRegister();
  if (!ret.second) {
    MS_LOG(WARNING) << "[DataDump] Insert graphId:" << graph->graph_id() << " data dumper failed";
  }
}

void AscendKernelRuntime::LaunchDataDump(GraphId graph_id) {
  if (!DataDumpParser::GetInstance().DumpEnabled()) {
    return;
  }
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

void AscendKernelRuntime::DebugTaskIdName(GraphId graph_id) {
  auto runtime_info_map = ModelRunner::Instance().GetRuntimeInfoMap(graph_id);
  for (auto iter : runtime_info_map) {
    MS_LOG(WARNING) << "Task name:" << iter.first << " task_id:" << std::get<kTupleTaskId>(*iter.second)
                    << " stream_id:" << std::get<kTupleStreamId>(*iter.second);
  }
}

bool AscendKernelRuntime::RunTask(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
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
    MS_LOG(ERROR) << "Run task failed";
    DebugTaskIdName(graph->graph_id());
    return false;
  }
  return true;
}

bool AscendKernelRuntime::SyncStream() {
  if (RT_ERROR_NONE != rtStreamSynchronize(stream_)) {  // o for switch stream
    MS_LOG(ERROR) << "Call runtime rtStreamSynchronize error.";
    return false;
  }
  return true;
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
  if (context_ptr->enable_hccl()) {
    if (!HcclInit()) {
      MS_LOG(ERROR) << "HcclInit init failed";
      return false;
    }
  }

  ret = rtCtxCreate(&rt_context_, 0, device_id_);
  if (ret != RT_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "Call rtCtxCreate, ret[" << static_cast<int>(ret) << "]";
  }

  ret = rtCtxSetCurrent(rt_context_);
  if (ret != RT_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "Call rtCtxSetCurrent, ret[" << ret << "]";
  }

  ret = rtStreamCreate(&stream_, 0);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rtStreamCreate, ret[" << ret << "]";
  }

  return true;
}

bool AscendKernelRuntime::ResetDevice() {
  auto ret = rtCtxSetCurrent(rt_context_);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Call rtCtxSetCurrent failed";
    return false;
  }

  if (stream_ != nullptr) {
    ret = rtStreamDestroy(stream_);
    if (ret != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "Call rtStreamDestroy, ret[" << ret << "]";
    }
    stream_ = nullptr;
  }

  if (rt_context_ != nullptr) {
    ret = rtCtxDestroy(rt_context_);
    if (ret != RT_ERROR_NONE) {
      MS_EXCEPTION(DeviceProcessError) << "Call rtCtxDestroy, ret[" << ret << "]";
    }
    rt_context_ = nullptr;
  }
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
  hcclResult_t res = hcom_init(full_path, rank_id_str.c_str());
  free(full_path);
  if (res != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "Hcom init failed, res is " << static_cast<int>(res);
    return false;
  }
  return true;
}

bool AscendKernelRuntime::DestroyHccl() {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (!NeedDestroyHccl()) {
    MS_LOG(INFO) << "Hccl is not enable, no need to close.";
    return true;
  }
  hcclResult_t res = hcom_destroy();
  if (res != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "Hccl destroy failed";
    return false;
  }
  MS_LOG(INFO) << "Hccl destroy successful, status = " << res << ".";
  context_ptr->set_enable_hccl(false);
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
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
