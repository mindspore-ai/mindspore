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
#include "plugin/device/ascend/hal/device/profiling/profiling_utils.h"
#include <algorithm>
#include "kernel/kernel.h"
#include "plugin/device/ascend/hal/device/profiling/profiling_manager.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "utils/ms_utils.h"
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"
#include "nlohmann/json.hpp"
#include "mindspore/core/ops/core_ops.h"
#include "include/backend/debug/profiler/profiling.h"

namespace mindspore {
namespace device {
namespace ascend {
constexpr char kFpStartNode[] = "fp_point";
constexpr char kBpEndNode[] = "bp_point";
constexpr uint64_t kProfilingFpStartLogId = 2;
constexpr uint64_t kProfilingBpEndLogId = 3;
constexpr uint64_t kProfilingIterEndLogId = 4;
constexpr auto kDouble = 2;

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
  selected_kernel_builder.SetFusionType(kernel::kPatternOpaque);
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
  for (auto data : report_data_) {
    ReportProfilingData(data.task_ids_, data.stream_ids_, data.graph_id_, data.rt_model_id);
  }
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
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
