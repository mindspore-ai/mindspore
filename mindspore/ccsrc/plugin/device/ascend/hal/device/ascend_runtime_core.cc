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

#include "plugin/device/ascend/hal/device/ascend_runtime_core.h"
#include "plugin/device/ascend/hal/device/ascend_kernel_runtime.h"
#include <locale>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <algorithm>
#include <set>
#include "ops/ascend_op_name.h"
#include "ops/other_ops.h"
#include "include/common/utils/signal_util.h"
#include "plugin/device/ascend/hal/device/ascend_device_address.h"
#include "utils/ms_context.h"
#include "include/common/utils/mpi/mpi_config.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "runtime/rt.h"
#include "acl/acl_rt.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_kernel_load.h"
#include "plugin/device/ascend/hal/device/ascend_runtime_manager.h"
#include "plugin/device/ascend/hal/hardware/ascend_collective_comm_lib.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "plugin/device/ascend/hal/device/ascend_stream_assign.h"
#include "plugin/device/ascend/hal/device/ge_runtime/model_runner.h"
#include "plugin/device/ascend/hal/device/tasksink/task_generator.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/backend/optimizer/helper.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/session/kernel_build_client.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "plugin/device/ascend/kernel/tbe/op_impl_mode_config.h"
#include "kernel/oplib/op_info_utils.h"
#include "common/plugin/opp_so_manager.h"
#ifndef ENABLE_SECURITY
#include "toolchain/prof_api.h"
#include "include/backend/debug/profiler/profiling.h"
#include "plugin/device/ascend/hal/device/profiling/profiling_manager.h"
#include "plugin/device/ascend/hal/device/profiling/profiling_utils.h"
#endif
#include "plugin/device/ascend/hal/device/ascend_memory_manager.h"
#include "plugin/device/ascend/hal/device/ascend_event.h"
#ifndef ENABLE_SECURITY
#include "plugin/device/ascend/hal/device/dump/ascend_dump.h"
#include "include/backend/debug/data_dump/dump_json_parser.h"
#include "include/backend/debug/data_dump/e2e_dump.h"
#include "plugin/device/ascend/hal/device/dump/kernel_dumper.h"
#endif
#include "toolchain/adx_datadump_server.h"
#include "utils/trace_base.h"
#include "external/acl/error_codes/rt_error_codes.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/parallel_context.h"
#include "include/common/utils/comm_manager.h"
#ifdef MEM_REUSE_DEBUG
#include "backend/common/mem_reuse/mem_reuse_checker.h"
#include "include/common/debug/env_config_parser.h"
#endif
#include "include/common/utils/config_manager.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#include "plugin/device/ascend/hal/device/ascend_data_queue.h"
#ifdef ENABLE_DUMP_IR
#include "include/common/debug/rdr/recorder_manager.h"
#endif

#include "kernel/framework_utils.h"
#include "plugin/device/ascend/hal/common/platform_info_util.h"
#ifndef ENABLE_SECURITY
using mindspore::device::ascend::ProfilingManager;
using mindspore::device::ascend::ProfilingUtils;
#endif
using mindspore::device::ascend::tasksink::TaskGenerator;
using mindspore::ge::model_runner::ModelRunner;
using mindspore::kernel::tbe::TbeUtils;
using std::vector;

constexpr uint32_t kTupleTaskId = 0;
constexpr uint32_t kTupleStreamId = 1;
constexpr uint32_t kTupleArgs = 2;
constexpr uint32_t kTupleInfoId = 3;
constexpr uint32_t kProfilingMaxTaskIdInStream = 65531;
constexpr auto kModuleName = "MindSpore";
constexpr size_t kPathMax = 4096;

namespace mindspore::device::ascend {
namespace {
void WriteEvent(const CNodePtr &cnode, std::ofstream *ofs) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(ofs);
  if (common::AnfAlgo::HasNodeAttr(kAttrEventId, cnode)) {
    *ofs << common::AnfAlgo::GetNodeAttr<uint32_t>(cnode, kAttrEventId) << ", ";
  } else {
    *ofs << ", ";
  }
}

void WriteLabel(const CNodePtr &cnode, std::ofstream *ofs) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(ofs);
  if (common::AnfAlgo::HasNodeAttr(kAttrLabelIndex, cnode)) {
    *ofs << common::AnfAlgo::GetNodeAttr<uint32_t>(cnode, kAttrLabelIndex) << ", ";
  } else if (common::AnfAlgo::HasNodeAttr(kAttrLabelSwitchList, cnode)) {
    auto label_list = common::AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(cnode, kAttrLabelSwitchList);
    std::string label_str = "\"";
    for (size_t i = 0; i < label_list.size(); ++i) {
      label_str += std::to_string(label_list[i]) + (i + 1 < label_list.size() ? ", " : "\", ");
    }
    *ofs << label_str;
  } else {
    *ofs << ", ";
  }
}

void WriteActiveStream(const CNodePtr &cnode, std::ofstream *ofs) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(ofs);
  std::string active_stream_str;
  if (common::AnfAlgo::HasNodeAttr(kAttrActiveStreamList, cnode)) {
    auto stream_list = common::AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(cnode, kAttrActiveStreamList);
    active_stream_str = "\"";
    for (size_t j = 0; j < stream_list.size(); ++j) {
      active_stream_str += std::to_string(stream_list[j]) + (j + 1 < stream_list.size() ? ", " : "\", ");
    }
    *ofs << active_stream_str;
  } else {
    *ofs << ", ";
  }
}

void WriteGroup(const CNodePtr &cnode, std::ofstream *ofs) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(ofs);
  if (AnfAlgo::GetKernelType(cnode) == HCCL_KERNEL && common::AnfAlgo::HasNodeAttr(kAttrGroup, cnode)) {
    *ofs << common::AnfAlgo::GetNodeAttr<std::string>(cnode, kAttrGroup) << ", ";
  } else {
    *ofs << ", ";
  }
}
}  // namespace

void AscendRuntimeCore::InitCore() { kernel::OpImplModeConfig::GetInstance().Initialize(); }

bool AscendRuntimeCore::GenTask(const session::KernelGraph &graph) {
  SetContextForce();
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
  ModelRunner::Instance().SetModelStatus(graph.graph_id(), ge::model_runner::ModelStatus::RAW);
  if (!ret.second) {
    MS_LOG(EXCEPTION) << "Duplicate GraphId! Please check in ascend_session.";
  }
  MS_LOG(INFO) << "TaskGenerator GetTaskInfo end...";
  return true;
}

void AscendRuntimeCore::GenKernelEventsCore(const session::KernelGraph &graph) {
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

void AscendRuntimeCore::GetLastNodesOnStream(const std::vector<CNodePtr> &kernels,
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

void AscendRuntimeCore::ProcessBoundaryEvent(
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

bool AscendRuntimeCore::LoadTask(const session::KernelGraph &graph) {
  SetContextForce();
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
  if (ProfilingManager::GetInstance().IsProfilingInitialized()) {
    ProfilingUtils::GetGraphNodes(graph);
  }
#endif

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
  LaunchDataDump(graph.graph_id());
#endif

  ModelRunner::Instance().LoadModelComplete(model_iter->first);
  return true;
}

bool AscendRuntimeCore::RunTaskCore(const session::KernelGraph &graph) {
  if (ModelRunner::Instance().GetModelStatus(graph.graph_id()) == ge::model_runner::ModelStatus::UNLOADED) {
    if (!LoadTask(graph)) {
      return false;
    }
  }
  current_graph_ = &graph;
  SetContextForce();
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
          aclrtMemcpy(&value, sizeof(int32_t), device_address->ptr_, device_address->size_, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret == ACL_ERROR_NONE) {
          MS_LOG(INFO) << "Value = " << value;
        }
      }
    }
#ifndef ENABLE_SECURITY
    MS_LOG(INFO) << "Set default stream to null.";
    stream_ = nullptr;
    communication_stream_ = nullptr;
    PrintDebugInfoAndDumpFailNode(graph);
    DumpDebugInfoFile(graph);
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
  ModelRunner::Instance().SetModelStatus(graph.graph_id(), ge::model_runner::ModelStatus::EXECUTED);
  task_fail_infos_.clear();
  return true;
}

bool AscendRuntimeCore::LoadCore(const session::KernelGraph &graph, bool is_task_sink) {
  if (!is_task_sink) {
    MS_LOG(INFO) << "Graph mode with not task sink";
    GenKernelEventsCore(graph);
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

void *AscendRuntimeCore::GetModelStreamCore(uint32_t graph_id) const {
  return ModelRunner::Instance().GetModelStream(graph_id);
}

bool AscendRuntimeCore::GraphWithEmptyTaskList(const session::KernelGraph &graph) const {
  auto iter = task_map_.find(graph.graph_id());
  if (iter == task_map_.end()) {
    MS_LOG(EXCEPTION) << "Unknown graph ptr";
  }
  return iter->second.empty();
}

bool AscendRuntimeCore::CheckGraphIdValid(GraphId graph_id) const {
  return task_map_.find(graph_id) != task_map_.end() && graph_model_map_.find(graph_id) != graph_model_map_.end();
}

bool AscendRuntimeCore::LoadDataCore() {
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

void AscendRuntimeCore::RegTaskFailCallback(const bool &is_release) {
  // Set callback func when exception error
  auto rt_ret = is_release ? rtRegTaskFailCallbackByModule(kModuleName, nullptr)
                           : rtRegTaskFailCallbackByModule(kModuleName, TaskFailCallback);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Reg SetTaskFailCallback failed, error: " << rt_ret;
  }
}

void AscendRuntimeCore::TaskFailCallback(rtExceptionInfo *task_fail_info) {
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
    auto node = GetErrorNodeInfo(task_fail_info->streamid, task_fail_info->taskid).first;
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
    task_fail_infos_.push_back(*task_fail_info);
  }
}

std::pair<CNodePtr, std::string> AscendRuntimeCore::GetErrorNodeInfo(uint32_t streamid, uint32_t taskid) {
  if (current_graph_ == nullptr) {
    return {nullptr, ""};
  }
  auto runtime_info_map = ModelRunner::Instance().GetRuntimeInfoMap(current_graph_->graph_id());
  for (const auto &iter : runtime_info_map) {
    MS_EXCEPTION_IF_NULL(iter.second);
    uint32_t task_id = std::get<kTupleTaskId>(*iter.second);
    uint32_t stream_id = std::get<kTupleStreamId>(*iter.second);
    std::string task_info = std::get<kTupleInfoId>(*iter.second);
    if (task_id == taskid && stream_id == streamid) {
      MS_EXCEPTION_IF_NULL(current_graph_);
      auto &execute_node = current_graph_->execution_order();
      auto node = std::find_if(execute_node.begin(), execute_node.end(), [&iter](const auto &node) {
        MS_EXCEPTION_IF_NULL(node);
        return node->UniqueName() == iter.first;
      });
      if (node != execute_node.end()) {
        return {*node, task_info};
      }
    }
  }
  return {nullptr, ""};
}

std::string AscendRuntimeCore::GetDumpPath(const std::string &suffix) {
  std::string rank_id_str;
  auto inst = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(inst);
  if (inst->parallel_mode() != parallel::kStandalone) {
    uint32_t rank_id = 0;
    if (!CommManager::GetInstance().GetRankID(kHcclWorldGroup, &rank_id)) {
      MS_LOG(WARNING) << "Get rank id failed, now using the default value 0.";
    }
    rank_id_str = std::to_string(rank_id);
  } else {
    rank_id_str = common::GetEnv(kRankID);
    if (rank_id_str.empty()) {
      MS_LOG(WARNING) << "Environment variable 'RANK_ID' is empty, using the default value: 0";
      rank_id_str = "0";
    }
  }

  auto ms_om_path = common::GetEnv("MS_OM_PATH");
  std::string path;
  if (ms_om_path.empty()) {
    MS_LOG(WARNING) << "The environment variable 'MS_OM_PATH' is not set, the files will save to the process local "
                    << "path, as ./rank_id/" + suffix + "/...";
    path = "./rank_" + rank_id_str + "/" + suffix;
  } else {
    path = ms_om_path + "/rank_" + rank_id_str + "/" + suffix;
  }
  return path;
}

bool AscendRuntimeCore::DeleteDumpDir(const std::string &path) {
  string real_path = GetRealPath(path);
  if (DeleteDumpFile(real_path) == -1) {
    return false;
  }
  if (rmdir(real_path.c_str()) == -1) {
    MS_LOG(WARNING) << "Delete dir " << real_path << " failed!";
  }
  return true;
}

std::string AscendRuntimeCore::GetRealPath(const std::string &path) {
  char real_path_mem[kPathMax] = {0};
  char *real_path_ret = realpath(path.c_str(), real_path_mem);
  if (real_path_ret == nullptr) {
    return "";
  }
  return std::string(real_path_mem);
}

int AscendRuntimeCore::DeleteDumpFile(std::string path) {
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

#ifndef ENABLE_SECURITY
void AscendRuntimeCore::PrintDebugInfoAndDumpFailNode(const session::KernelGraph & /* graph */) {
  const std::string path = GetDumpPath("node_dump");
  if (access(path.c_str(), F_OK) == 0) {
    if (!DeleteDumpDir(path)) {
      MS_LOG(ERROR) << "Delete dump directory " << path << " failed";
    }
  }
  for (const auto &task_fail_info : task_fail_infos_) {
    MS_LOG(ERROR) << "Task fail infos, rt task_id: " << task_fail_info.taskid
                  << ", rt stream_id: " << task_fail_info.streamid << ", tid: " << task_fail_info.tid
                  << ", device_id: " << task_fail_info.deviceid << ", retcode: " << task_fail_info.retcode << " ("
                  << GetErrorMsg(task_fail_info.retcode) << ")";
    auto error_node_info = GetErrorNodeInfo(task_fail_info.streamid, task_fail_info.taskid);
    CNodePtr &node = error_node_info.first;
    std::string &task_info = error_node_info.second;
    // step1: print task info
    if (!task_info.empty()) {
      MS_LOG(ERROR) << "Task DebugString, " << task_info;
    }

    // step2: dump fail node
    // Dump error data in local path
    if (node == nullptr) {
      continue;
    }
    auto full_scope_name = node->fullname_with_scope();
    MS_LOG(WARNING) << "Dump task error infos (input/output's value) for node:[" << full_scope_name
                    << "], save path: " << path << ", " << trace::DumpSourceLines(node, false);

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
      MS_LOG(WARNING)
        << "    4) If it is a dynamic dataset, please set the input to dynamic through `set_inputs`, or set sink_size "
           "to 1. It is recommended to use the former, because the latter has poor performance.";
    }

    E2eDump::DumpInputData(node, false, path, &full_scope_name);
    E2eDump::DumpOutputData(node, false, path, &full_scope_name);
  }
}

void AscendRuntimeCore::DumpDebugInfoFile(const session::KernelGraph &graph) {
  auto file = GetDumpPath("exec_order") + "/" + graph.ToString() + ".csv";
  auto realpath = Common::CreatePrefixPath(file, true);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path failed. path=" << file;
    return;
  }
  file = realpath.value();
  std::ofstream ofs(file, std::ios::ate);  // raii object, no need to close
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open file [" << file << "] failed!";
    return;
  }
  const auto &runtime_info_map = ModelRunner::Instance().GetRuntimeInfoMap(graph.graph_id());

  ofs << "Index, Op Name, Stream ID (MindSpore), Stream ID (Runtime), Task ID (Runtime), Event ID (MindSpore), "
      << "Label ID (MindSpore), Active Stream ID (MindSpore), Group Name" << std::endl;
  for (size_t i = 0; i < graph.execution_order().size(); ++i) {
    const CNodePtr &cur_cnode_ptr = graph.execution_order()[i];
    MS_EXCEPTION_IF_NULL(cur_cnode_ptr);
    ofs << i << ", ";
    ofs << cur_cnode_ptr->UniqueName() << ", ";
    ofs << AnfAlgo::GetStreamId(cur_cnode_ptr) << ", ";
    if (auto iter = runtime_info_map.find(cur_cnode_ptr->UniqueName()); iter != runtime_info_map.end()) {
      ofs << std::get<kTupleStreamId>(*iter->second) << ", " << std::get<kTupleTaskId>(*iter->second) << ", ";
    } else if (cur_cnode_ptr->UniqueName().find(kEndGraph) != std::string::npos) {
      iter = runtime_info_map.find(kEndGraph);
      if (iter != runtime_info_map.end()) {
        ofs << std::get<kTupleStreamId>(*iter->second) << ", " << std::get<kTupleTaskId>(*iter->second) << ", ";
      } else {
        ofs << "NOT FOUND, NOT FOUND, ";
      }
    } else {
      ofs << "NOT FOUND, NOT FOUND, ";
    }
    WriteEvent(cur_cnode_ptr, &ofs);
    WriteLabel(cur_cnode_ptr, &ofs);
    WriteActiveStream(cur_cnode_ptr, &ofs);
    WriteGroup(cur_cnode_ptr, &ofs);

    ofs << std::endl;
  }
  MS_LOG(ERROR) << "Execute order has saved at " << file;
}

void AscendRuntimeCore::DistributeDebugTask(const session::KernelGraph &graph,
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

void AscendRuntimeCore::LaunchDataDump(GraphId graph_id) {
  if (!DumpJsonParser::GetInstance().async_dump_enabled()) {
    return;
  }
  MS_LOG(INFO) << "Start Launch Dump Data";
  auto runtime_info_map = ModelRunner::Instance().GetRuntimeInfoMap(graph_id);
  for (auto iter = runtime_info_map.begin(); iter != runtime_info_map.end();) {
    if (std::get<kTupleArgs>(*iter->second) == nullptr) {
      iter = runtime_info_map.erase(iter);
    } else {
      ++iter;
    }
  }
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

void AscendRuntimeCore::UnloadModelCore(uint32_t graph_id) {
  SetContextForce();
  if (graph_id != UINT32_MAX) {
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
    if (auto model_iter = graph_model_map_.find(graph_id); model_iter != graph_model_map_.end()) {
      MS_LOG(DEBUG) << "Ge UnloadModel " << graph_id;
      ModelRunner::Instance().UnloadModel(graph_id);
      (void)graph_model_map_.erase(model_iter);
    }
    rt_model_zero_copy_.Release(graph_id);
  } else {
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
    for (auto &iter : graph_model_map_) {
      MS_LOG(INFO) << "Ge UnloadModel " << iter.first;
      ModelRunner::Instance().UnloadModel(iter.first);
    }
    graph_model_map_.clear();
  }
}

bool AscendRuntimeCore::CheckAndUnloadModelInAdvance(uint32_t model_id) {
  if (ModelRunner::Instance().GetModelStatus(model_id) == ge::model_runner::ModelStatus::EXECUTED) {
    ModelRunner::Instance().UnloadModel(model_id);
    return true;
  }
  return false;
}

std::vector<rtExceptionInfo> AscendRuntimeCore::task_fail_infos_ = {};
std::map<std::string, uint32_t> AscendRuntimeCore::overflow_tasks_;
AscendRuntimeCore::~AscendRuntimeCore() { graph_model_map_.clear(); }
}  // namespace mindspore::device::ascend
