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

#include "device/ascend/ascend_kernel_runtime.h"
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <exception>
#include <algorithm>

#include "device/ascend/ascend_device_address.h"
#include "device/cpu/mpi/mpi_adapter.h"
#include "utils/context/ms_context.h"
#include "utils/mpi/mpi_config.h"
#include "device/ascend/profiling/profiling_manager.h"
#include "hccl/hcom.h"
#include "common/trans.h"
#include "runtime/context.h"
#include "device/ascend/ascend_label_assign.h"
#include "device/ascend/ascend_stream_assign.h"
#include "device/ascend/ascend_memory_pool.h"
#include "framework/ge_runtime/model_runner.h"
#include "device/ascend/tasksink/task_generator.h"
#include "session/anf_runtime_algorithm.h"
#include "device/ascend/profiling/profiling_utils.h"
#include "kernel/tbe/tbe_utils.h"
#include "kernel/tbe/tbe_python_funcs.h"
#include "pre_activate/mem_reuse/mem_reuse_checker.h"
#include "device/ascend/ascend_memory_manager.h"

using mindspore::device::ascend::ProfilingManager;
using mindspore::device::ascend::ProfilingUtils;
using mindspore::device::ascend::tasksink::TaskGenerator;
using mindspore::kernel::tbe::TbeUtils;
using std::vector;

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
    int rank_id = device::cpu::MPIAdapter::Instance().GetRankId();
    const char *offset = std::getenv("RANK_OFFSET");
    if (offset != nullptr) {
      try {
        int rank_offset = std::stoi(offset);
        rank_id += rank_offset;
      } catch (std::invalid_argument) {
        MS_LOG(EXCEPTION) << "stoi invalid argument:" << offset;
      } catch (std::out_of_range) {
        MS_LOG(EXCEPTION) << "stoi out_of_range:" << offset;
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
    MS_LOG(ERROR) << "get hccl rankid failed, please set env RANK_ID";
  }
  return rank_id_str;
}
}  // namespace

AscendKernelRuntime::~AscendKernelRuntime() { graph_model_map_.clear(); }

void AscendKernelRuntime::ClearGraphModelMap() {
  for (auto &iter : graph_model_map_) {
    MS_LOG(INFO) << "Ge UnloadModel " << iter.first;
    auto ret = ge::model_runner::ModelRunner::Instance().UnloadModel(iter.first);
    if (!ret) {
      MS_LOG(ERROR) << "UnloadModel failed";
    }
  }
}

void AscendKernelRuntime::ClearGraphRuntimeResource(uint32_t graph_id) {
  MS_LOG(DEBUG) << "clear graph:" << graph_id << " runtime resource";
  auto iter = graph_model_map_.find(graph_id);
  if (iter == graph_model_map_.end()) {
    MS_LOG(DEBUG) << "GraphId:" << graph_id << " not found";
    return;
  }
  MS_LOG(DEBUG) << "Ge UnloadModel " << iter->first;
  auto ret = ge::model_runner::ModelRunner::Instance().UnloadModel(iter->first);
  if (!ret) {
    MS_LOG(ERROR) << "UnloadModel failed";
  }
  graph_model_map_.erase(iter);
}

bool AscendKernelRuntime::NeedDestroyHccl() {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (!context_ptr->enable_hccl()) {
    MS_LOG(INFO) << "hccl is not enabled";
    return false;
  }
  // Note: make sure hcom_connectivity_detection api never be used.
  return true;
}

void AscendKernelRuntime::ReleaseDeviceRes() {
  MS_LOG(INFO) << "ascend finalize start";
  // release ge runtime
  ClearGraphModelMap();

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto ret = rtSetDevice(context_ptr->device_id());
  if (ret != RT_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "rtSetDevice, ret[" << static_cast<int>(ret) << "]";
  }

  if (mem_manager_ != nullptr) {
    mem_manager_->FreeDeviceMemory();
  }

  (void)DestroyHccl();
  (void)ResetDevice();
  (void)ProfilingManager::GetInstance().StopProfiling();
  MS_LOG(INFO) << "ascend finalize end";
}

bool AscendKernelRuntime::Init() {
  if (initialized_) {
    return true;
  }
  bool ret = false;
#ifdef ENABLE_DUMP_E2E
  ret = SetDumpConf();
  if (!ret) {
    MS_LOG(INFO) << "no dump conf to set!";
  }
#endif

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

bool AscendKernelRuntime::DumpData(mindspore::session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
#ifdef ENABLE_DUMP_E2E
  MS_LOG(INFO) << "start dump step";
  DumpConfPtr dump_conf = GetDumpConf();
  MS_EXCEPTION_IF_NULL(dump_conf);
  dump_conf->UpdataCurIter();
  bool dump_flag = dump_conf->dump_enable();
  if (!dump_flag) {
    MS_LOG(INFO) << "dump flag is disable, pass dump step";
    return true;
  }
  uint32_t cur_iter = dump_conf->cur_iter();
  if (dump_conf->dump_iter() != 0) {
    if (cur_iter != dump_conf->dump_iter()) {
      return true;
    }
  }
  MS_LOG(INFO) << "cur iter is " << cur_iter;
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
    MS_LOG(WARNING) << "graph " << graph->graph_id() << " have no compute node";
    return true;
  }

  AscendStreamAssign &stream_assign_instance = AscendStreamAssign::GetInstance();
  AscendLabelAssign &label_assign_instance = AscendLabelAssign::GetInstance();
  // the streams' flag not HEAD_STREAM
  std::vector<uint32_t> wait_active_stream_list;
  stream_assign_instance.GetWaitStreams(&wait_active_stream_list);
  auto force_copy_stream_list = stream_assign_instance.hcom_streams();

  MS_LOG(INFO) << "call DavinciModel total stream num:" << stream_assign_instance.GetTotalStreamNum()
               << ", total event num:" << stream_assign_instance.total_event_num()
               << ", total label num:" << label_assign_instance.GetLabelNum(NOT_NULL(graph))
               << ", wait_active_stream_list size:" << wait_active_stream_list.size()
               << ", force_copy_stream_list size:" << force_copy_stream_list.size();

  std::vector<std::shared_ptr<ge::model_runner::OpInfo>> empty_list;
  std::shared_ptr<ge::model_runner::DavinciModel> model = std::make_shared<ge::model_runner::DavinciModel>(
    task_info_list, empty_list, empty_list, empty_list, empty_list, wait_active_stream_list, force_copy_stream_list, 0,
    0, 0, 0, 0, 0, stream_assign_instance.GetTotalStreamNum(), label_assign_instance.GetLabelNum(NOT_NULL(graph)),
    stream_assign_instance.total_event_num(), 0);

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
  bool status = ge::model_runner::ModelRunner::Instance().LoadDavinciModel(device_id_, 0, model_iter->first,
                                                                           model_iter->second, listener);
  if (!status) {
    MS_LOG(EXCEPTION) << "Load Task Failed";
  }
  if (ProfilingManager::GetInstance().IsProfiling()) {
    auto task_ids = ge::model_runner::ModelRunner::Instance().GetTaskIdList(model_iter->first);
    auto stream_ids = ge::model_runner::ModelRunner::Instance().GetStreamIdList(model_iter->first);
    ProfilingUtils::ReportProfilingData(task_ids, stream_ids, NOT_NULL(graph));
  }
  return true;
}

void AscendKernelRuntime::DebugTaskIdName(GraphId graph_id) {
  auto task_ids = ge::model_runner::ModelRunner::Instance().GetTaskIdList(graph_id);
  auto graph_task_names = ProfilingUtils::graph_kernel_name();
  auto iter = graph_task_names.find(graph_id);
  if (iter != graph_task_names.end()) {
    const auto &task_names = iter->second;
    if (task_ids.size() != task_names.size()) {
      MS_LOG(WARNING) << "Task_ids and task_names size not match";
      return;
    }
    for (size_t i = 0; i < task_ids.size(); ++i) {
      MS_LOG(INFO) << "Task_id:" << task_ids[i] << " task_name:" << task_names[i];
    }
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

  bool status = ge::model_runner::ModelRunner::Instance().RunModel(graph->graph_id(), input_tensors, output_tensors);
  if (!status) {
    MS_LOG(ERROR) << "run task failed";
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
    MS_EXCEPTION(DeviceProcessError) << "rtGetDeviceCount, ret[" << static_cast<int>(ret) << "]";
  }

  ret = rtSetDevice(device_id_);
  if (ret != RT_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "rtSetDevice, ret[" << static_cast<int>(ret) << "]";
  }

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr == nullptr) {
    MS_LOG(ERROR) << "get MsContext instance failed";
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
    MS_EXCEPTION(DeviceProcessError) << "rtCtxCreate, ret[" << static_cast<int>(ret) << "]";
  }

  ret = rtCtxSetCurrent(rt_context_);
  if (ret != RT_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "rtCtxSetCurrent, ret[" << ret << "]";
  }

  ret = rtStreamCreate(&stream_, 0);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "rtStreamCreate, ret[" << ret << "]";
  }

  return true;
}

bool AscendKernelRuntime::ResetDevice() {
  auto ret = rtCtxSetCurrent(rt_context_);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "call rtCtxSetCurrent failed";
    return false;
  }

  if (stream_ != nullptr) {
    ret = rtStreamDestroy(stream_);
    if (ret != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "rtStreamDestroy, ret[" << ret << "]";
    }
    stream_ = nullptr;
  }

  if (rt_context_ != nullptr) {
    ret = rtCtxDestroy(rt_context_);
    if (ret != RT_ERROR_NONE) {
      MS_EXCEPTION(DeviceProcessError) << "rtCtxDestroy, ret[" << ret << "]";
    }
    rt_context_ = nullptr;
  }
  return true;
}

bool AscendKernelRuntime::HcclInit() {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (!context_ptr->IsTsdOpened()) {
    MS_LOG(EXCEPTION) << "Hccl dependent tsd is not open";
  }
  MS_LOG(INFO) << "do hcom init";
  auto config_path_str = std::getenv("MINDSPORE_HCCL_CONFIG_PATH");
  if (config_path_str == nullptr) {
    config_path_str = std::getenv("RANK_TABLE_FILE");
    if (config_path_str == nullptr) {
      MS_LOG(ERROR) << "get hccl json config failed, please set env MINDSPORE_HCCL_CONFIG_PATH or RANK_TABLE_FILE";
    }
    return false;
  }
  std::string rank_id_str = GetRankId();
  auto full_path = realpath(config_path_str, nullptr);
  if (full_path == nullptr) {
    MS_LOG(ERROR) << "file path " << config_path_str << " does not exist";
    return false;
  }
  MS_LOG(INFO) << "MINDSPORE_HCCL_CONFIG_PATH : " << full_path << ", RANK_ID: " << rank_id_str;
  hcclResult_t res = hcom_init(full_path, rank_id_str.c_str());
  free(full_path);
  if (res != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "hcom init failed, res is " << static_cast<int>(res);
    return false;
  }
  return true;
}

bool AscendKernelRuntime::DestroyHccl() {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (!NeedDestroyHccl()) {
    MS_LOG(INFO) << "hccl is not enable, no need to close.";
    return true;
  }
  hcclResult_t res = hcom_destroy();
  if (res != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "hccl destroy failed";
    return false;
  }
  MS_LOG(INFO) << "hccl destroy successful, status = " << res << ".";
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
