/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/hal/device/profiling/profiling_utils.h"
#include <sys/syscall.h>
#include <algorithm>
#include <mutex>
#include "kernel/kernel.h"
#include "ops/structure_op_name.h"
#include "plugin/device/ascend/hal/device/profiling/profiling_manager.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"
#include "nlohmann/json.hpp"
#include "include/backend/debug/profiler/profiling.h"
#include "plugin/device/ascend/kernel/ascend_kernel_mod.h"
#ifndef ASCEND_910B
#include "plugin/device/ascend/hal/device/ge_runtime/model_runner.h"
using mindspore::ge::model_runner::ModelRunner;
#endif

namespace mindspore {
namespace device {
namespace ascend {
constexpr auto kPatternOpaque = "Opaque";
constexpr char kFpStartNode[] = "fp_point";
constexpr char kBpEndNode[] = "bp_point";
constexpr uint64_t kProfilingFpStartLogId = 2;
constexpr uint64_t kProfilingBpEndLogId = 3;
constexpr uint64_t kProfilingIterEndLogId = 4;
constexpr auto kDouble = 2;
std::mutex ProfilingUtils::profiler_mutex;

const std::unordered_map<std::string, GeProfInfoType> kNamesToProfTypes = {
  {"ModelExecute", GeProfInfoType::kModelExecute},
  {"ModelLoad", GeProfInfoType::kModelLoad},
  {"InputCopy", GeProfInfoType::kInputCopy},
  {"OutputCopy", GeProfInfoType::kOutputCopy},
  {"InferShape", GeProfInfoType::kInferShape},
  {"CompatibleInferShape", GeProfInfoType::kCompatibleInferShape},
  {"Tiling", GeProfInfoType::kTiling},
  {"CompatibleTiling", GeProfInfoType::kCompatibleTiling},
  {"StreamSync", GeProfInfoType::kStreamSync},
  {"step_info", GeProfInfoType::kStepInfo},
  {"task_memory_info", GeProfInfoType::kTaskMemoryInfo}};

nlohmann::json GetContextProfilingOption() {
  auto profiler_manager = profiler::ProfilerManager::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_manager);
  const string prof_options_str = profiler_manager->GetProfilingOptions();
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(prof_options_str);
  } catch (nlohmann::json::parse_error &e) {
    MS_LOG(EXCEPTION) << "Parse profiling option json failed, error:" << e.what();
  }
  return j;
}

ProfilingTraceInfo ProfilingUtils::GenerateProfilingTrace(const session::KernelGraph &kernel_graph) {
  MS_LOG(INFO) << "Profiling graph:" << kernel_graph.graph_id() << " Start to get trace";
  custom_node_index_ = 5000;
  auto &cnode_exec_order = kernel_graph.execution_order();
  auto profiling_option = GetContextProfilingOption();

  ProfilingTraceInfo profiling_trace;
  if (cnode_exec_order.size() <= 1) {
    return profiling_trace;
  }
  GetTraceBegin(kernel_graph, profiling_option, &profiling_trace);
  GetTraceIterEnd(kernel_graph, &profiling_trace);
  GetTraceBpEnd(kernel_graph, profiling_option, &profiling_trace);
  GetTraceHccl(kernel_graph, NOT_NULL(&profiling_trace));

  auto set_string_converter = [](const std::set<std::string> &str_set) {
    std::ostringstream stream;
    stream << "(";
    (void)std::copy(str_set.begin(), str_set.end(), std::ostream_iterator<std::string>(stream, ","));
    stream << ")";
    return stream.str();
  };

  MS_LOG(INFO) << "Profiling graph:" << kernel_graph.graph_id() << " trace_begin:" << profiling_trace.trace_begin
               << " trace_bp_end:" << set_string_converter(profiling_trace.trace_bp_end)
               << " trace_iter_end:" << set_string_converter(profiling_trace.trace_iter_end);
  return profiling_trace;
}

void ProfilingUtils::GetTraceHccl(const session::KernelGraph &kernel_graph,
                                  NotNull<ProfilingTraceInfo *> profiling_trace) {
  for (const auto &node : kernel_graph.execution_order()) {
    if (common::AnfAlgo::IsCommunicationOp(node)) {
      MS_EXCEPTION_IF_NULL(node);
      (void)profiling_trace->trace_custom_node.insert(node->fullname_with_scope());
      MS_LOG(INFO) << "Profiling graph:" << kernel_graph.graph_id() << " Get hccl node:" << node->fullname_with_scope();
    }
  }
}

void ProfilingUtils::GetTraceBegin(const session::KernelGraph &kernel_graph, const nlohmann::json &option,
                                   ProfilingTraceInfo *trace_info) {
  MS_EXCEPTION_IF_NULL(trace_info);
  auto &execution_orders = kernel_graph.execution_order();
  auto iter = option.find(kFpStartNode);
  if (iter != option.end() && iter->is_string()) {
    std::string trace_begin_str = *iter;
    if (!trace_begin_str.empty()) {
      MS_LOG(INFO) << "Profiling graph:" << kernel_graph.graph_id()
                   << " Get fp_point from profiling_option:" << trace_begin_str;
      trace_info->trace_begin = trace_begin_str;
      return;
    }
  }

  std::string fp_start_str;
  std::set<std::string> getnext_outputs;
  GetCNodeOutputRealNode(kGetNextOpName, kernel_graph, NOT_NULL(&getnext_outputs));
  if (getnext_outputs.empty()) {
    auto first_node = execution_orders.front();
    MS_EXCEPTION_IF_NULL(first_node);
    fp_start_str = first_node->fullname_with_scope();
  } else {
    for (auto &cnode : execution_orders) {
      MS_EXCEPTION_IF_NULL(cnode);
      if (getnext_outputs.count(cnode->fullname_with_scope()) != 0) {
        fp_start_str = cnode->fullname_with_scope();
        break;
      }
    }
  }
  trace_info->trace_begin = fp_start_str;
}

void ProfilingUtils::GetCNodeOutputRealNode(const std::string &node_name, const session::KernelGraph &kernel_graph,
                                            NotNull<std::set<std::string> *> getnext_outputs) {
  for (const auto &cnode : kernel_graph.execution_order()) {
    MS_EXCEPTION_IF_NULL(cnode);
    for (const auto &input : cnode->inputs()) {
      auto prev_cnode = common::AnfAlgo::VisitKernel(input, 0);
      MS_EXCEPTION_IF_NULL(prev_cnode.first);
      if (!prev_cnode.first->isa<CNode>()) {
        continue;
      }
      if (common::AnfAlgo::GetCNodeName(prev_cnode.first) == node_name) {
        (void)getnext_outputs->insert(cnode->fullname_with_scope());
        MS_LOG(INFO) << "Profiling graph:" << kernel_graph.graph_id()
                     << " Find GetNext Output CNode:" << cnode->fullname_with_scope();
      }
    }
  }
  if (getnext_outputs->empty()) {
    MS_LOG(INFO) << "Profiling graph:" << kernel_graph.graph_id() << " GetNext not found";
  }
}

void ProfilingUtils::GetTraceBpEnd(const session::KernelGraph &kernel_graph, const nlohmann::json &option,
                                   ProfilingTraceInfo *trace_info) {
  MS_EXCEPTION_IF_NULL(trace_info);
  auto bp_point = option.find(kBpEndNode);
  if (bp_point != option.end() && bp_point->is_string()) {
    std::string bp_point_str = *bp_point;
    if (!bp_point_str.empty()) {
      MS_LOG(INFO) << "Profiling graph:" << kernel_graph.graph_id()
                   << " Get bp_point from profiling_option:" << bp_point_str;
      (void)trace_info->trace_bp_end.insert(bp_point_str);
      return;
    }
  }

  std::string bp_end_str;
  // Contain hccl kernel
  auto &execution_orders = kernel_graph.execution_order();
  auto iter = execution_orders.rbegin();
  while (iter != execution_orders.rend()) {
    if (common::AnfAlgo::IsCommunicationOp(*iter)) {
      // store communication op input nodes' name
      std::set<std::string> ar_input_node_names;
      size_t input_num = common::AnfAlgo::GetInputTensorNum(*iter);
      for (size_t i = 0; i < input_num; ++i) {
        auto input_node_with_index = common::AnfAlgo::GetPrevNodeOutput(*iter, i);
        auto input_node = input_node_with_index.first;
        MS_EXCEPTION_IF_NULL(input_node);
        (void)ar_input_node_names.insert(input_node->fullname_with_scope());
      }
      // start from previous node
      ++iter;
      // find input names in previous node
      while (iter != execution_orders.rend()) {
        MS_EXCEPTION_IF_NULL(*iter);
        if (ar_input_node_names.find((*iter)->fullname_with_scope()) != ar_input_node_names.end()) {
          bp_end_str = (*iter)->fullname_with_scope();
          break;
        }
        ++iter;
      }
      break;
    }
    ++iter;
  }

  if (bp_end_str.empty()) {
    trace_info->trace_bp_end = trace_info->trace_iter_end;
  } else {
    (void)trace_info->trace_bp_end.insert(bp_end_str);
  }
}

std::string ProfilingUtils::GetGraphLastKernelName(const session::KernelGraph &kernel_graph) {
  std::string last_tbe_kernel_name;
  auto &execution_order = kernel_graph.execution_order();
  // find last tbe_kernel
  for (auto iter = execution_order.rbegin(); iter != execution_order.rend(); ++iter) {
    MS_EXCEPTION_IF_NULL(*iter);
    if (AnfAlgo::GetKernelType(*iter) == TBE_KERNEL || AnfAlgo::GetKernelType(*iter) == AKG_KERNEL ||
        common::AnfAlgo::IsCommunicationOp(*iter)) {
      last_tbe_kernel_name = (*iter)->fullname_with_scope();
      break;
    }
  }
  if (last_tbe_kernel_name.empty()) {
    MS_LOG(WARNING) << "Profiling graph:" << kernel_graph.graph_id() << " No TBE or AKG or HCCL node found";
  }
  return last_tbe_kernel_name;
}

void ProfilingUtils::GetTraceIterEnd(const session::KernelGraph &kernel_graph, ProfilingTraceInfo *trace_info) {
  MS_EXCEPTION_IF_NULL(trace_info);
  // Find last execute node in control flow
  auto &execution_orders = kernel_graph.execution_order();
  for (auto &node : execution_orders) {
    if (common::AnfAlgo::HasNodeAttr(kAttrProfilingIterEnd, node)) {
      MS_LOG(INFO) << "Profiling graph:" << kernel_graph.graph_id()
                   << " Found PROFILING_ITER_END:" << node->fullname_with_scope();
      (void)trace_info->trace_iter_end.insert(node->fullname_with_scope());
    }
  }

  if (!trace_info->trace_iter_end.empty()) {
    return;
  }

  MS_LOG(WARNING) << "Profiling graph:" << kernel_graph.graph_id()
                  << " PROFILING_ITER_END not found. Found last TBE or Akg or Hccl op as PROFILING_ITER_END instead.";
  auto last_kernel_name = GetGraphLastKernelName(kernel_graph);
  if (last_kernel_name.empty()) {
    MS_LOG(WARNING) << "Profiling graph:" << kernel_graph.graph_id() << " No TBE or AKG or HCCL op found in graph";
  } else {
    (void)trace_info->trace_iter_end.insert(last_kernel_name);
  }
}

NotNull<CNodePtr> ProfilingUtils::CreateProfilingCNode(const ProfilingContent &profiling_content,
                                                       NotNull<session::KernelGraph *> graph_ptr) {
  kernel::KernelBuildInfo::KernelBuildInfoBuilder selected_kernel_builder;
  selected_kernel_builder.SetInputsFormat({kOpFormat_DEFAULT, kOpFormat_DEFAULT});
  selected_kernel_builder.SetInputsDeviceType({TypeId::kNumberTypeInt32, TypeId::kNumberTypeInt32});
  selected_kernel_builder.SetFusionType(kPatternOpaque);
  selected_kernel_builder.SetProcessor(kernel::Processor::AICORE);
  selected_kernel_builder.SetKernelType(KernelType::RT_KERNEL);
  abstract::AbstractBasePtr type_none_abstract = std::make_shared<abstract::AbstractNone>();
  auto primitive = std::make_shared<Primitive>(ProfilingUtils::kProfiling);
  std::vector<AnfNodePtr> inputs;
  (void)inputs.emplace_back(NewValueNode(primitive));
  CNodePtr cnode_ptr = graph_ptr->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(cnode_ptr);
  AnfAlgo::SetSelectKernelBuildInfo(selected_kernel_builder.Build(), cnode_ptr.get());
  cnode_ptr->set_abstract(type_none_abstract);
  // set attr
  ValuePtr notify_value = MakeValue(profiling_content.notify);
  ValuePtr trace_id_value = MakeValue(profiling_content.profiler_trace_id);
  ValuePtr flags_value = MakeValue(profiling_content.flags);
  common::AnfAlgo::SetNodeAttr(ProfilingUtils::kNotify, notify_value, cnode_ptr);
  common::AnfAlgo::SetNodeAttr(ProfilingUtils::kProfilerTraceId, trace_id_value, cnode_ptr);
  common::AnfAlgo::SetNodeAttr(ProfilingUtils::kFlags, flags_value, cnode_ptr);
  return NOT_NULL(cnode_ptr);
}

void ProfilingUtils::SaveProfilingPoint(uint32_t graph_id, const std::string &node_name, uint32_t point_id) {
  MS_LOG(INFO) << "Save profiling point, graph id" << graph_id << ", node name: " << node_name
               << ", point_id: " << point_id;
  std::shared_ptr<StepPointDesc> point_desc_ptr = std::make_shared<StepPointDesc>(node_name, point_id);
  auto iter = graph_point_.find(graph_id);
  if (iter == graph_point_.end()) {
    (void)graph_point_.emplace(graph_id, std::vector<std::shared_ptr<StepPointDesc>>{point_desc_ptr});
  } else {
    (void)iter->second.emplace_back(point_desc_ptr);
  }
}

void ProfilingUtils::InsertProfilingTraceFp(const mindspore::AnfNodePtr &anf_node,
                                            const ProfilingTraceInfo &profiling_trace_info,
                                            NotNull<session::KernelGraph *> graph_ptr,
                                            NotNull<std::vector<mindspore::CNodePtr> *> kernel_list) {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (profiling_trace_info.trace_begin == anf_node->fullname_with_scope()) {
    MS_LOG(INFO) << "Profiling graph:" << graph_ptr->graph_id()
                 << " Match FpStart:" << profiling_trace_info.trace_begin;
    InsertProfilingTraceJobId(anf_node, graph_ptr, kernel_list);
    ProfilingContent fp_profiling_content = {false, kProfilingFpStartLogId, 0};
    auto fp_profiling_node = CreateProfilingCNodeWithStream(anf_node, fp_profiling_content, graph_ptr);
    (void)kernel_list->emplace_back(fp_profiling_node);
    // insert ProfDesc
    SaveProfilingPoint(graph_ptr->graph_id(), anf_node->fullname_with_scope(), kProfilingFpStartLogId);
  }
}

void ProfilingUtils::InsertProfilingTraceJobId(const AnfNodePtr &anf_node, NotNull<session::KernelGraph *> graph_ptr,
                                               NotNull<std::vector<CNodePtr> *> kernel_list) {
  auto job_id = ProfilingManager::GetInstance().GetJobId();
  ProfilingContent job_profiling_context = {false, job_id, 0};
  auto job_profiling_node = CreateProfilingCNodeWithStream(anf_node, job_profiling_context, graph_ptr);
  (void)kernel_list->emplace_back(job_profiling_node);
}

CNodePtr ProfilingUtils::CreateProfilingCNodeWithStream(const mindspore::AnfNodePtr &anf_node,
                                                        const ProfilingContent &profiling_content,
                                                        NotNull<session::KernelGraph *> graph_ptr) {
  CNodePtr profiling_node = CreateProfilingCNode(profiling_content, graph_ptr);
  AnfAlgo::SetStreamDistinctionLabel(AnfAlgo::GetStreamDistinctionLabel(anf_node.get()), profiling_node.get());
  AnfAlgo::SetStreamId(AnfAlgo::GetStreamId(anf_node), profiling_node.get());
  return profiling_node;
}

void ProfilingUtils::InsertProfilingCustomOp(const AnfNodePtr &anf_node, const ProfilingTraceInfo &profiling_trace_info,
                                             NotNull<session::KernelGraph *> graph_ptr,
                                             NotNull<std::vector<CNodePtr> *> kernel_list) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto iter = profiling_trace_info.trace_custom_node.find(anf_node->fullname_with_scope());
  if (iter == profiling_trace_info.trace_custom_node.end()) {
    return;
  }
  MS_LOG(INFO) << "Profiling graph:" << graph_ptr->graph_id() << " Match CustomOp:" << anf_node->fullname_with_scope();
  // custom op profiling job start from 10000.
  auto custom_point_id = kDouble * custom_node_index_;
  ProfilingContent front_profiling_content = {false, custom_point_id, 0};
  CNodePtr front_node = CreateProfilingCNodeWithStream(anf_node, front_profiling_content, graph_ptr);
  (void)kernel_list->insert(kernel_list->end() - 1, front_node);
  SaveProfilingPoint(graph_ptr->graph_id(), anf_node->fullname_with_scope(), custom_point_id);

  ProfilingContent back_profiling_content = {false, custom_point_id + 1, 0};
  CNodePtr back_node = CreateProfilingCNodeWithStream(anf_node, back_profiling_content, graph_ptr);
  (void)kernel_list->insert(kernel_list->end(), back_node);
  SaveProfilingPoint(graph_ptr->graph_id(), anf_node->fullname_with_scope(), custom_point_id + 1);
  ++custom_node_index_;
}

void ProfilingUtils::InsertProfilingTraceBpEnd(const AnfNodePtr &anf_node,
                                               const ProfilingTraceInfo &profiling_trace_info,
                                               NotNull<session::KernelGraph *> graph_ptr,
                                               NotNull<std::vector<CNodePtr> *> kernel_list) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto node_name = anf_node->fullname_with_scope();
  if (profiling_trace_info.trace_bp_end.find(node_name) != profiling_trace_info.trace_bp_end.end()) {
    MS_LOG(INFO) << "Profiling graph:" << graph_ptr->graph_id() << " Match BpEnd:" << node_name;
    ProfilingContent bp_end_profiling_content = {false, kProfilingBpEndLogId, 0};
    CNodePtr bp_end_node = CreateProfilingCNodeWithStream(anf_node, bp_end_profiling_content, graph_ptr);
    (void)kernel_list->emplace_back(bp_end_node);
    SaveProfilingPoint(graph_ptr->graph_id(), node_name, kProfilingBpEndLogId);
  }
}

void ProfilingUtils::InsertProfilingTraceIterEnd(const AnfNodePtr &anf_node,
                                                 const ProfilingTraceInfo &profiling_trace_info,
                                                 NotNull<session::KernelGraph *> graph_ptr,
                                                 NotNull<std::vector<mindspore::CNodePtr> *> kernel_list) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto full_scope_name = anf_node->fullname_with_scope();
  if (profiling_trace_info.trace_iter_end.find(full_scope_name) != profiling_trace_info.trace_iter_end.end()) {
    MS_LOG(INFO) << "Profiling graph:" << graph_ptr->graph_id() << " Match IterEnd:" << full_scope_name;
    ProfilingContent iter_end_profiling_content = {true, kProfilingIterEndLogId, 0};
    auto iter_end_kernel_ptr = CreateProfilingCNodeWithStream(anf_node, iter_end_profiling_content, graph_ptr);
    (void)kernel_list->emplace_back(iter_end_kernel_ptr);
    SaveProfilingPoint(graph_ptr->graph_id(), full_scope_name, kProfilingIterEndLogId);
  }
}

void ProfilingUtils::SetGraphKernelName(uint32_t graph_id, const std::vector<std::string> &kernel_names) {
  auto ret = graph_kernel_name_.try_emplace(graph_id, kernel_names);
  if (!ret.second) {
    MS_LOG(WARNING) << "[profiling] graph " << graph_id << " kernel names already exist";
  }
}

void ProfilingUtils::SetGraphProfilingCNode(uint32_t graph_id, const std::vector<CNodePtr> &profiling_cnode_list) {
  auto ret = graph_profiling_cnode_.try_emplace(graph_id, profiling_cnode_list);
  if (!ret.second) {
    MS_LOG(WARNING) << "[profiling] graph " << graph_id << " profiling cnode list already exist";
  }
}

bool ProfilingUtils::ValidComputeGraph(const session::KernelGraph &kernel_graph) {
  for (const auto &node : kernel_graph.execution_order()) {
    auto kernel_type = AnfAlgo::GetKernelType(node);
    if (kernel_type == TBE_KERNEL || kernel_type == AKG_KERNEL || kernel_type == AICPU_KERNEL) {
      return true;
    }
  }
  return false;
}

void ProfilingUtils::ReportAllGraphProfilingData() {
  MS_LOG(INFO) << "report event: " << report_event_.size();
  for (auto data : report_event_) {
    auto ret = MsprofReportEvent(static_cast<uint32_t>(false), &data);
    if (ret != MSPROF_ERROR_NONE) {
      MS_LOG(ERROR) << "RecordModelLoad failed.";
    }
  }
  MS_LOG(INFO) << "report event: " << report_compact_info_.size();
  for (auto data : report_compact_info_) {
    auto compact_ret = MsprofReportCompactInfo(false, &data, sizeof(MsprofCompactInfo));
    if (compact_ret != MSPROF_ERROR_NONE) {
      MS_LOG(ERROR) << "MsprofReportCompactInfo failed.";
    }
  }

  MS_LOG(INFO) << "report event: " << report_additional_info_.size();
  for (auto data : report_additional_info_) {
    auto addition_ret = MsprofReportAdditionalInfo(false, &data, sizeof(MsprofAdditionalInfo));
    if (addition_ret != MSPROF_ERROR_NONE) {
      MS_LOG(ERROR) << "MsprofReportAdditionalInfo failed.";
    }
  }

  MS_LOG(INFO) << "report event: " << report_api_.size();
  for (auto data : report_api_) {
    auto api_ret = MsprofReportApi(false, &data);
    if (api_ret != MSPROF_ERROR_NONE) {
      MS_LOG(ERROR) << "MsprofReportAdditionalInfo failed.";
    }
  }
  report_event_.clear();
  report_compact_info_.clear();
  report_additional_info_.clear();
  report_api_.clear();
}

void ProfilingUtils::ReportProfilingData(const std::vector<uint32_t> &task_ids, const std::vector<uint32_t> &stream_ids,
                                         uint32_t graph_id, uint32_t rt_model_id) {
  auto ret = graph_profiling_cnode_.find(graph_id);
  if (ret == graph_profiling_cnode_.end()) {
    MS_LOG(WARNING) << "Graph id not found in graph_profiling_cnode_, graph id is " << graph_id
                    << ", will not report this graph profiling data.";
    return;
  }

  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);

  auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  ProfilingReporter reporter(device_id, graph_id, rt_model_id, ret->second, stream_ids, task_ids);
  reporter.ReportTasks();

  // Report profiling point
  auto point_iter = graph_point_.find(graph_id);
  if (point_iter == graph_point_.end()) {
    MS_LOG(WARNING) << "Graph id not found in graph_point, will not report this graph step point data, graph id is: "
                    << graph_id;
    return;
  }
  reporter.ReportStepPoint(point_iter->second);
}

void ProfilingUtils::SetReportProfilingData(const std::vector<uint32_t> &task_ids,
                                            const std::vector<uint32_t> &stream_ids, uint32_t graph_id,
                                            uint32_t rt_model_id) {
  GraphProfilingData report_data = {task_ids, stream_ids, graph_id, rt_model_id};
  (void)report_data_.emplace_back(report_data);
}

// Report MindSpore Framework data to Ascend Profiler
void ProfilingUtils::ReportMindSporeFrameworkData() {
  if (ProfilingManager::GetInstance().IsMsprofiling()) {
    return;
  }
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  uint32_t graph_id = 0;
  uint32_t rt_model_id = 0;
  std::vector<CNodePtr> cnode_list;
  std::vector<uint32_t> stream_ids;
  std::vector<uint32_t> task_ids;
  ProfilingReporter repoter(device_id, graph_id, rt_model_id, cnode_list, stream_ids, task_ids);
  MS_LOG(INFO) << "Start to report MindSpore Framework data to Ascend Profiler.";
  repoter.ReportParallelStrategy();
  repoter.ReportMDTraceData();
  MS_LOG(INFO) << "Stop to report MindSpore Framework data to Ascend Profiler.";
}

uint64_t ProfilingUtils::GetMsprofHashId(const std::string &info) {
  auto iter = msprof_hash_id_.find(info);
  if (iter == msprof_hash_id_.end()) {
    const char *hash_info = info.c_str();
    uint64_t hash_id = MsprofGetHashId(hash_info, info.length());
    auto ret = msprof_hash_id_.try_emplace(info, hash_id);
    if (!ret.second) {
      MS_LOG(WARNING) << "note " << info << " hash id already exist";
    }
    return hash_id;
  } else {
    return iter->second;
  }
}

void ProfilingUtils::BuildSingleTensorInfo(const CNodePtr &node, const uint64_t opName_hash_id, const size_t index,
                                           const uint32_t tensor_num, TensorInfoWrapper *tensor_info_wrapper) {
  MS_EXCEPTION_IF_NULL(tensor_info_wrapper);
  auto &tensor_info = tensor_info_wrapper->tensor_info;
  tensor_info.type = MSPROF_REPORT_NODE_TENSOR_INFO_TYPE;
  tensor_info.level = MSPROF_REPORT_NODE_LEVEL;
  tensor_info_wrapper->tensor_num = tensor_num;
  tensor_info.dataLen = kTensorInfoBytesWithCap + kTensorInfoBytes * (static_cast<uint32_t>(tensor_num) - 1U);
  auto prof_tensor_data = reinterpret_cast<MsprofTensorInfo *>(tensor_info.data);
  prof_tensor_data->opName = opName_hash_id;
  prof_tensor_data->tensorNum = tensor_num;
  for (size_t k = 0UL; k < static_cast<size_t>(tensor_num); ++k) {
    const size_t tensor_index = (index * static_cast<size_t>(MSPROF_GE_TENSOR_DATA_NUM)) + k;
    InitProfTensorData(node, tensor_index, k, prof_tensor_data);
  }
}

void ProfilingUtils::InitProfTensorData(const CNodePtr &node, const size_t index, const uint64_t offset_idx,
                                        MsprofTensorInfo *tensor_info) {
  const auto InitTensorDesc = [&tensor_info](const MsprofGeTensorType tensor_type, const ShapeVector &shape,
                                             const std::string &format, const uint32_t vm_data_type,
                                             const uint64_t offset_idx) {
    tensor_info->tensorData[offset_idx].tensorType = static_cast<uint32_t>(tensor_type);
    // when enum Format is changed, profiling analyze needs to be synchronized
    tensor_info->tensorData[offset_idx].format = OpFormat2Index[format] + MSPROF_DIFFERENCE;
    // when enum DataType is changed, profiling analyze needs to be synchronized
    tensor_info->tensorData[offset_idx].dataType = vm_data_type + MSPROF_DIFFERENCE;
    auto shape_size = std::min(static_cast<uint64_t>(MSPROF_GE_TENSOR_DATA_SHAPE_LEN), shape.size());
    (void)std::copy(shape.begin(), shape.begin() + shape_size, tensor_info->tensorData[offset_idx].shape);
  };

  const size_t input_size = common::AnfAlgo::GetInputTensorNum(node);
  if (index < input_size) {
    // when current index is smaller than input_size, build tensor by input tensor
    auto [input_node, input_index] = common::AnfAlgo::GetPrevNodeOutput(node, index);
    ShapeVector shape = AnfAlgo::GetOutputDeviceShape(input_node, input_index);
    std::string data_format = AnfAlgo::GetOutputFormat(input_node, input_index);
    uint32_t vm_data_type = static_cast<uint32_t>(AnfAlgo::GetOutputDeviceDataType(input_node, input_index));
    InitTensorDesc(MSPROF_GE_TENSOR_TYPE_INPUT, shape, data_format, vm_data_type, offset_idx);
  } else {
    // when current index is bigger than input_size, build tensor by output tensor, use index - input_size as
    // index of output tensor
    ShapeVector shape = AnfAlgo::GetOutputDeviceShape(node, index - input_size);
    std::string data_format = AnfAlgo::GetOutputFormat(node, index - input_size);
    uint32_t vm_data_type = static_cast<uint32_t>(AnfAlgo::GetOutputDeviceDataType(node, index - input_size));
    InitTensorDesc(MSPROF_GE_TENSOR_TYPE_OUTPUT, shape, data_format, vm_data_type, offset_idx);
  }
}

void ProfilingUtils::RecordModelLoad(const rtModel_t rt_model_handle) {
#ifndef ASCEND_910B
  uint32_t rt_model_id = 0;
  rtError_t rt_model_ret = rtModelGetId(rt_model_handle, &rt_model_id);
  if (rt_model_ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Call rt api rtModelGetId failed, ret: " << rt_model_ret;
    return;
  }
  MS_LOG(INFO) << "RecordModelLoad model_id: " << rt_model_id;

  const uint64_t prof_time = MsprofSysCycleTime();
  MsprofEvent model_load_event_{};
  model_load_event_.type = static_cast<uint32_t>(GeProfInfoType::kModelLoad);
  model_load_event_.itemId = rt_model_id;
  model_load_event_.level = MSPROF_REPORT_MODEL_LEVEL;
  model_load_event_.timeStamp = prof_time;
  model_load_event_.requestId = 0U;
  auto tid = syscall(SYS_gettid);
  model_load_event_.threadId = static_cast<uint32_t>(tid);
  if (ProfilingManager::GetInstance().IsProfilingStart()) {
    auto ret = MsprofReportEvent(static_cast<uint32_t>(false), &model_load_event_);
    if (ret != MSPROF_ERROR_NONE) {
      MS_LOG(ERROR) << "RecordModelLoad failed.";
    }
  } else {
    report_event_.emplace_back(model_load_event_);
  }
#endif
}

void ProfilingUtils::RecordModelExecute(const KernelGraphPtr kernel_graph) {
#ifndef ASCEND_910B
  uint32_t rt_model_id = 0;
  rtModel_t rt_model_handle = ModelRunner::Instance().GetModelHandle(kernel_graph->graph_id());
  rtError_t rt_model_ret = rtModelGetId(rt_model_handle, &rt_model_id);
  if (rt_model_ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Call rt api rtModelGetId failed, ret: " << rt_model_ret;
    return;
  }
  MS_LOG(INFO) << "RecordModelExecute model_id: " << rt_model_id;

  auto request_id = 0;

  MsprofEvent model_execute{};
  model_execute.level = MSPROF_REPORT_MODEL_LEVEL;
  model_execute.itemId = rt_model_id;
  auto tid = syscall(SYS_gettid);
  model_execute.threadId = static_cast<uint32_t>(tid);
  model_execute.type = static_cast<uint32_t>(GeProfInfoType::kModelExecute);
  const uint64_t prof_time = MsprofSysCycleTime();
  model_execute.timeStamp = prof_time;
  model_execute.requestId = static_cast<uint32_t>(request_id);
  if (ProfilingManager::GetInstance().IsProfilingStart()) {
    auto ret = MsprofReportEvent(static_cast<uint32_t>(false), &model_execute);
    if (ret != MSPROF_ERROR_NONE) {
      MS_LOG(ERROR) << "RecordModelLoad failed.";
    }
  } else {
    report_event_.emplace_back(model_execute);
  }
#endif
}

std::string ProfilingUtils::GetFullScopeName(const std::string &op_name, const bool is_op_name) {
  std::string full_scope_name;
  if (!is_op_name) {
    auto op_index = op_name.find("-op");
    if (op_index != std::string::npos) {
      full_scope_name = op_name.substr(0, op_name.find("_", op_index + 1));
    }
  } else {
    full_scope_name = op_name;
  }
  return full_scope_name;
}

void ProfilingUtils::RecordLaunchTaskBegin(const std::string &op_name, const bool is_op_name) {
  std::string full_scope_name = GetFullScopeName(op_name, is_op_name);
  auto iter = node_addition_info_.find(full_scope_name);
  if (iter == node_addition_info_.end()) {
    MS_LOG(WARNING) << "Do not find op info: " << full_scope_name;
    return;
  }
  ProfNodeAdditionInfo &addition_info = iter->second;
  addition_info.api.beginTime = MsprofSysCycleTime();
  MS_LOG(DEBUG) << "api Launch begin " << full_scope_name << ", " << addition_info.api.beginTime;
}

void ProfilingUtils::RegisterProfType() {
  if (is_prof_type_registered_) {
    return;
  }
  for (const auto &name_to_type : kNamesToProfTypes) {
    if (name_to_type.second < GeProfInfoType::kModelLevelEnd) {
      auto ret = MsprofRegTypeInfo(MSPROF_REPORT_MODEL_LEVEL, static_cast<uint32_t>(name_to_type.second),
                                   name_to_type.first.c_str());
      if (ret != MSPROF_ERROR_NONE) {
        MS_LOG(ERROR) << "MSPROF_REPORT_MODEL_LEVEL failed.";
      }
    } else {
      auto ret = MsprofRegTypeInfo(MSPROF_REPORT_NODE_LEVEL, static_cast<uint32_t>(name_to_type.second),
                                   name_to_type.first.c_str());
      if (ret != MSPROF_ERROR_NONE) {
        MS_LOG(ERROR) << "MSPROF_REPORT_NODE_LEVEL failed.";
      }
    }
  }
  is_prof_type_registered_ = true;
  return;
}

void ProfilingUtils::ReportTask(const std::string &op_name, const bool is_op_name) {
  std::string full_scope_name = GetFullScopeName(op_name, is_op_name);
  auto iter = node_addition_info_.find(full_scope_name);
  if (iter == node_addition_info_.end()) {
    MS_LOG(WARNING) << "Do not find op info: " << full_scope_name;
    return;
  }
  ProfNodeAdditionInfo &addition_info = iter->second;
  const uint64_t prof_time = MsprofSysCycleTime();
  addition_info.node_basic_info.timeStamp = prof_time;
  auto tid = syscall(SYS_gettid);
  addition_info.node_basic_info.threadId = static_cast<uint32_t>(tid);

  if (ProfilingManager::GetInstance().IsProfilingStart()) {
    auto compact_ret = MsprofReportCompactInfo(false, &addition_info.node_basic_info, sizeof(MsprofCompactInfo));
    if (compact_ret != MSPROF_ERROR_NONE) {
      MS_LOG(ERROR) << "MsprofReportCompactInfo failed.";
    }
  } else {
    report_compact_info_.emplace_back(addition_info.node_basic_info);
  }
  MS_LOG(DEBUG) << "MsprofReportCompactInfo：op_name: " << op_name
                << ", tensors: " << addition_info.tensor_info_wrappers.size();

  for (auto &tensor_info_wrapper : addition_info.tensor_info_wrappers) {
    tensor_info_wrapper.tensor_info.timeStamp = prof_time;
    tensor_info_wrapper.tensor_info.threadId = static_cast<uint32_t>(tid);
    if (ProfilingManager::GetInstance().IsProfilingStart()) {
      auto addition_ret =
        MsprofReportAdditionalInfo(false, &tensor_info_wrapper.tensor_info, sizeof(MsprofAdditionalInfo));
      if (addition_ret != MSPROF_ERROR_NONE) {
        MS_LOG(ERROR) << "MsprofReportAdditionalInfo failed.";
      }
    } else {
      report_additional_info_.emplace_back(tensor_info_wrapper.tensor_info);
    }
  }

  addition_info.api.endTime = prof_time;
  addition_info.api.threadId = static_cast<uint32_t>(tid);
  if (ProfilingManager::GetInstance().IsProfilingStart()) {
    auto api_ret = MsprofReportApi(false, &addition_info.api);
    if (api_ret != MSPROF_ERROR_NONE) {
      MS_LOG(ERROR) << "MsprofReportAdditionalInfo failed.";
    }
  } else {
    report_api_.emplace_back(addition_info.api);
  }
}

void ProfilingUtils::InitLaunchApi(const uint64_t name_hash, MsprofApi *api) {
  const auto kernel_type_hash = MSPROF_REPORT_NODE_LAUNCH_TYPE;
  api->type = kernel_type_hash;
  api->level = MSPROF_REPORT_NODE_LEVEL;
  api->itemId = name_hash;
}

uint32_t ProfilingUtils::GetBlockDim(const CNodePtr &node) {
  auto kernel_mod = AnfAlgo::GetKernelMod(node);
  auto ascend_kernel_mod = dynamic_cast<kernel::AscendKernelMod *>(kernel_mod);
  MS_EXCEPTION_IF_NULL(ascend_kernel_mod);
  return ascend_kernel_mod->block_dim();
}

void ProfilingUtils::InitReportNode(const CNodePtr &cnode, bool init_begin_time) {
  std::lock_guard<std::mutex> lock(profiler_mutex);
  MS_EXCEPTION_IF_NULL(cnode);
  ProfNodeAdditionInfo addition_info{};
  if (init_begin_time) {
    addition_info.api.beginTime = MsprofSysCycleTime();
  }
  MsprofCompactInfo &basic_info = addition_info.node_basic_info;
  basic_info.level = MSPROF_REPORT_NODE_LEVEL;
  basic_info.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
  auto &prof_node_basic_info = basic_info.data.nodeBasicInfo;
  uint64_t opName_hash_id = GetMsprofHashId(cnode->fullname_with_scope());
  prof_node_basic_info.opName = opName_hash_id;
  std::string opType = common::AnfAlgo::GetCNodeName(cnode);
  prof_node_basic_info.opType = GetMsprofHashId(opType);
  prof_node_basic_info.blockDim = GetBlockDim(cnode);
  KernelType kernel_type = AnfAlgo::GetKernelType(cnode);
  prof_node_basic_info.taskType = static_cast<uint32_t>(KernelType2TaskTypeEnum[kernel_type]);

  size_t total_size = common::AnfAlgo::GetInputTensorNum(cnode) + AnfAlgo::GetOutputTensorNum(cnode);
  const size_t batch_size = total_size / MSPROF_GE_TENSOR_DATA_NUM;
  for (size_t i = 0U; i < batch_size; i++) {
    TensorInfoWrapper tensor_info_wrapper{};
    BuildSingleTensorInfo(cnode, opName_hash_id, i, MSPROF_GE_TENSOR_DATA_NUM, &tensor_info_wrapper);
    addition_info.tensor_info_wrappers.emplace_back(tensor_info_wrapper);
  }

  const size_t remain_index = total_size % static_cast<size_t>(MSPROF_GE_TENSOR_DATA_NUM);
  if (remain_index != 0UL) {
    TensorInfoWrapper tensor_info_wrapper{};
    BuildSingleTensorInfo(cnode, opName_hash_id, batch_size, remain_index, &tensor_info_wrapper);
    (void)addition_info.tensor_info_wrappers.emplace_back(tensor_info_wrapper);
  }
  InitLaunchApi(opName_hash_id, &addition_info.api);

  auto ret = node_addition_info_.try_emplace(cnode->fullname_with_scope(), addition_info);
  if (!ret.second) {
    MS_LOG(DEBUG) << cnode->fullname_with_scope() << " node addition already exist";
  }
}

void ProfilingUtils::GetGraphNodes(const session::KernelGraph &kernel_graph) {
  RegisterProfType();
  for (const auto &cnode : kernel_graph.execution_order()) {
    InitReportNode(cnode);
  }
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
