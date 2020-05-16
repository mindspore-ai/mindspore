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

#include <map>
#include "device/ascend/profiling/reporter/graph_desc_reporter.h"
#include "device/ascend/profiling/profiling_utils.h"
#include "kernel/kernel.h"
#include "device/ascend/profiling/profiling_manager.h"
#include "session/anf_runtime_algorithm.h"
#include "common/utils.h"
#include "utils/utils.h"
#include "device/ascend/profiling/reporter/task_desc_reporter.h"
#include "utils/context/ms_context.h"

namespace mindspore {
namespace device {
namespace ascend {
constexpr uint32_t kMaxProfilingNodeNum = 100;
constexpr char kCustomNode[] = "PROFILING_CUSTOM_";
constexpr char kFpStartNode[] = "PROFILING_FP_START";
constexpr char kBpEndNode[] = "PROFILING_BP_END";
constexpr char kIterEndNode[] = "PROFILING_ITER_END";
std::unordered_map<uint32_t, std::vector<CNodePtr>> ProfilingUtils::graph_profiling_cnode_;
std::unordered_map<uint32_t, std::vector<std::string>> ProfilingUtils::graph_kernel_name_;
uint32_t ProfilingUtils::custom_node_index_ = 1;

ProfilingTraceInfo ProfilingUtils::GetProfilingTraceFromEnv(NotNull<const session::KernelGraph *> graph_ptr) {
  MS_LOG(INFO) << "get env start";
  custom_node_index_ = 1;
  auto &cnode_exec_order = graph_ptr->execution_order();
  ProfilingTraceInfo profiling_trace;
  profiling_trace.trace_begin = GetTraceBegin(cnode_exec_order);
  profiling_trace.trace_bp_end = GetTraceBpEnd(cnode_exec_order);
  profiling_trace.trace_netoutput = GetTraceNetoutput(cnode_exec_order);

  for (uint32_t i = 1; i <= kMaxProfilingNodeNum; ++i) {
    std::string env_str = std::string(kCustomNode) + std::to_string(i);
    const char *node_full_name = std::getenv(env_str.c_str());
    if (node_full_name == nullptr) {
      break;
    }
    MS_LOG(INFO) << "Get profiling node:" << node_full_name;
    profiling_trace.trace_custom_node.insert(node_full_name);
  }
  MS_LOG(INFO) << "get env end";
  GetTraceHccl(cnode_exec_order, NOT_NULL(&profiling_trace));

  MS_LOG(INFO) << "[profiling]trace_begin:" << profiling_trace.trace_begin
               << " trace_bp_end:" << profiling_trace.trace_bp_end
               << " trace_netoutput:" << profiling_trace.trace_netoutput;
  return profiling_trace;
}

void ProfilingUtils::GetTraceHccl(const std::vector<CNodePtr> &cnode_exec_order,
                                  NotNull<ProfilingTraceInfo *> profiling_trace) {
  for (const auto &node : cnode_exec_order) {
    if (AnfAlgo::IsCommunicationOp(node)) {
      MS_EXCEPTION_IF_NULL(node);
      profiling_trace->trace_custom_node.insert(node->fullname_with_scope());
      MS_LOG(INFO) << "[profiling]Get hccl node:" << node->fullname_with_scope();
    }
  }
}

std::string ProfilingUtils::GetTraceBegin(const std::vector<CNodePtr> &cnode_exec_order) {
  const char *trace_begin = std::getenv(kFpStartNode);
  if (trace_begin != nullptr) {
    return std::string(trace_begin);
  }

  std::string fp_start_str;
  std::set<std::string> getnext_outputs;
  GetCNodeOutputRealNode(kGetNextOpName, cnode_exec_order, NOT_NULL(&getnext_outputs));
  if (getnext_outputs.empty()) {
    auto first_node = cnode_exec_order.front();
    MS_EXCEPTION_IF_NULL(first_node);
    fp_start_str = first_node->fullname_with_scope();
  } else {
    for (auto &cnode : cnode_exec_order) {
      if (getnext_outputs.count(cnode->fullname_with_scope()) != 0) {
        fp_start_str = cnode->fullname_with_scope();
        break;
      }
    }
  }
  return fp_start_str;
}

void ProfilingUtils::GetCNodeOutputRealNode(const std::string &node_name, const std::vector<CNodePtr> &cnode_exec_order,
                                            NotNull<std::set<std::string> *> getnext_outputs) {
  for (const auto &cnode : cnode_exec_order) {
    for (const auto &input : cnode->inputs()) {
      auto prev_cnode = AnfAlgo::VisitKernel(input, 0);
      if (!prev_cnode.first->isa<CNode>()) {
        continue;
      }
      if (AnfAlgo::GetCNodeName(prev_cnode.first) == node_name) {
        getnext_outputs->insert(cnode->fullname_with_scope());
        MS_LOG(INFO) << "Find GetNext Output CNode:" << cnode->fullname_with_scope();
      }
    }
  }
  if (getnext_outputs->empty()) {
    MS_LOG(WARNING) << "GetNext not found";
  }
}

std::string ProfilingUtils::GetTraceBpEnd(const std::vector<CNodePtr> &cnode_exec_order) {
  const char *trace_bp_end = std::getenv(kBpEndNode);

  if (trace_bp_end != nullptr) {
    return std::string(trace_bp_end);
  }
  std::string bp_end_str;
  // Contain hccl kernel
  auto iter = cnode_exec_order.rbegin();
  while (iter != cnode_exec_order.rend()) {
    if (AnfAlgo::IsCommunicationOp(*iter)) {
      // store communication op input nodes' name
      std::set<std::string> ar_input_node_names;
      for (size_t i = 0; i < AnfAlgo::GetInputTensorNum(*iter); ++i) {
        auto input_node_with_index = AnfAlgo::GetPrevNodeOutput(*iter, i);
        auto input_node = input_node_with_index.first;
        ar_input_node_names.insert(input_node->fullname_with_scope());
      }
      // start from previous node
      ++iter;
      // find input names in previous node
      while (iter != cnode_exec_order.rend()) {
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
    bp_end_str = GetGraphLastTbeKernelName(cnode_exec_order);
  }
  return bp_end_str;
}

std::string ProfilingUtils::GetGraphLastTbeKernelName(const std::vector<CNodePtr> &cnode_exec_order) {
  std::string last_tbe_kernel_name;
  // find last tbe_kernel
  for (auto iter = cnode_exec_order.rbegin(); iter != cnode_exec_order.rend(); ++iter) {
    if (AnfAlgo::GetKernelType(*iter) == TBE_KERNEL) {
      last_tbe_kernel_name = (*iter)->fullname_with_scope();
      break;
    }
  }
  if (last_tbe_kernel_name.empty()) {
    MS_LOG(WARNING) << "tbe kernel not found in graph";
  }
  return last_tbe_kernel_name;
}

std::string ProfilingUtils::GetTraceNetoutput(const std::vector<CNodePtr> &cnode_exec_order) {
  const char *trace_netoutput = std::getenv(kIterEndNode);
  return trace_netoutput == nullptr ? GetGraphLastTbeKernelName(cnode_exec_order) : std::string(trace_netoutput);
}

NotNull<CNodePtr> ProfilingUtils::CreateProfilingCNode(const ProfilingContent &profiling_content,
                                                       NotNull<session::KernelGraph *> graph_ptr) {
  kernel::KernelBuildInfo::KernelBuildInfoBuilder selected_kernel_builder;
  selected_kernel_builder.SetInputsFormat({kOpFormat_DEFAULT, kOpFormat_DEFAULT});
  selected_kernel_builder.SetInputsDeviceType({TypeId::kNumberTypeInt32, TypeId::kNumberTypeInt32});
  selected_kernel_builder.SetFusionType(kernel::FusionType::OPAQUE);
  selected_kernel_builder.SetProcessor(kernel::Processor::AICORE);
  selected_kernel_builder.SetKernelType(KernelType::RT_KERNEL);
  abstract::AbstractBasePtr type_none_abstract = std::make_shared<abstract::AbstractNone>();
  auto primitive = std::make_shared<Primitive>(ProfilingUtils::kProfiling);
  std::vector<AnfNodePtr> inputs;
  inputs.emplace_back(NewValueNode(primitive));
  CNodePtr cnode_ptr = graph_ptr->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(cnode_ptr);
  AnfAlgo::SetSelectKernelBuildInfo(selected_kernel_builder.Build(), cnode_ptr.get());
  cnode_ptr->set_abstract(type_none_abstract);
  // set attr
  ValuePtr notify_value = MakeValue(profiling_content.notify);
  ValuePtr trace_id_value = MakeValue(profiling_content.profiler_trace_id);
  ValuePtr flags_value = MakeValue(profiling_content.flags);
  AnfAlgo::SetNodeAttr(ProfilingUtils::kNotify, notify_value, cnode_ptr);
  AnfAlgo::SetNodeAttr(ProfilingUtils::kProfilerTraceId, trace_id_value, cnode_ptr);
  AnfAlgo::SetNodeAttr(ProfilingUtils::kFlags, flags_value, cnode_ptr);
  return NOT_NULL(cnode_ptr);
}

void ProfilingUtils::ProfilingTraceFpStart(const mindspore::AnfNodePtr &anf_node,
                                           const ProfilingTraceInfo &profiling_trace_info,
                                           NotNull<session::KernelGraph *> graph_ptr,
                                           NotNull<std::vector<mindspore::CNodePtr> *> kernel_list) {
  if (profiling_trace_info.trace_begin == anf_node->fullname_with_scope()) {
    MS_LOG(INFO) << "Profiling Match FpStart:" << profiling_trace_info.trace_begin;
    ProfilingTraceJobId(anf_node, graph_ptr, kernel_list);
    ProfilingContent fp_profiling_content = {false, kProfilingFpStartLogId, 0};
    auto fp_profiling_node = CreateProfilingCNodeWithStream(anf_node, fp_profiling_content, graph_ptr);
    kernel_list->emplace_back(fp_profiling_node);
  }
}

void ProfilingUtils::ProfilingTraceJobId(const AnfNodePtr &anf_node, NotNull<session::KernelGraph *> graph_ptr,
                                         NotNull<std::vector<CNodePtr> *> kernel_list) {
  MS_LOG(INFO) << "Profiling Match start";
  auto job_id = ProfilingManager::GetInstance().GetJobId();
  ProfilingContent job_profiling_context = {false, job_id, 0};
  auto job_profiling_node = CreateProfilingCNodeWithStream(anf_node, job_profiling_context, graph_ptr);
  kernel_list->emplace_back(job_profiling_node);
}

CNodePtr ProfilingUtils::CreateProfilingCNodeWithStream(const mindspore::AnfNodePtr &anf_node,
                                                        const ProfilingContent &profiling_content,
                                                        NotNull<session::KernelGraph *> graph_ptr) {
  CNodePtr profiling_node = CreateProfilingCNode(profiling_content, graph_ptr);
  AnfAlgo::SetStreamDistinctionLabel(AnfAlgo::GetStreamDistinctionLabel(anf_node.get()), profiling_node.get());
  AnfAlgo::SetStreamId(AnfAlgo::GetStreamId(anf_node), profiling_node.get());
  return profiling_node;
}

void ProfilingUtils::ProfilingCustomOp(const AnfNodePtr &anf_node, const ProfilingTraceInfo &profiling_trace_info,
                                       NotNull<session::KernelGraph *> graph_ptr,
                                       NotNull<std::vector<CNodePtr> *> kernel_list) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto iter = profiling_trace_info.trace_custom_node.find(anf_node->fullname_with_scope());
  if (iter == profiling_trace_info.trace_custom_node.end()) {
    return;
  }
  MS_LOG(INFO) << "Profiling Match CustomOp:" << anf_node->fullname_with_scope();
  // custom op profiling job start from 3.
  ProfilingContent front_profiling_content = {false, 2 * custom_node_index_ + 1, 0};
  CNodePtr front_node = CreateProfilingCNodeWithStream(anf_node, front_profiling_content, graph_ptr);
  kernel_list->insert(kernel_list->end() - 1, front_node);

  ProfilingContent back_profiling_content = {false, 2 * custom_node_index_ + 2, 0};
  CNodePtr back_node = CreateProfilingCNodeWithStream(anf_node, back_profiling_content, graph_ptr);
  kernel_list->insert(kernel_list->end(), back_node);
  ++custom_node_index_;
}

void ProfilingUtils::ProfilingTraceBpEnd(const AnfNodePtr &anf_node, const ProfilingTraceInfo &profiling_trace_info,
                                         NotNull<session::KernelGraph *> graph_ptr,
                                         NotNull<std::vector<CNodePtr> *> kernel_list) {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (profiling_trace_info.trace_bp_end == anf_node->fullname_with_scope()) {
    MS_LOG(INFO) << "Profiling Match BpEnd:" << profiling_trace_info.trace_bp_end;
    ProfilingContent bp_end_profiling_content = {false, kProfilingBpEndLogId, 0};
    CNodePtr bp_end_node = CreateProfilingCNodeWithStream(anf_node, bp_end_profiling_content, graph_ptr);
    kernel_list->emplace_back(bp_end_node);
  }
}

void ProfilingUtils::ProfilingTraceEnd(const AnfNodePtr &anf_node, const ProfilingTraceInfo &profiling_trace_info,
                                       NotNull<session::KernelGraph *> graph_ptr,
                                       NotNull<std::vector<mindspore::CNodePtr> *> kernel_list) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto full_scope_name = anf_node->fullname_with_scope();
  if (profiling_trace_info.trace_netoutput == full_scope_name) {
    MS_LOG(INFO) << "Profiling Match IterEnd:" << profiling_trace_info.trace_netoutput;
    ProfilingContent bp_end_profiling_content = {true, kProfilingIterEndLogId, 0};
    CNodePtr bp_kernel_ptr = CreateProfilingCNodeWithStream(anf_node, bp_end_profiling_content, graph_ptr);
    kernel_list->emplace_back(bp_kernel_ptr);
  }
}

void ProfilingUtils::SetGraphKernelName(uint32_t graph_id, const std::vector<std::string> &kernel_names) {
  auto ret = graph_kernel_name_.try_emplace(graph_id, kernel_names);
  if (!ret.second) {
    MS_LOG(ERROR) << "[profiling]graph " << graph_id << " kernel names already exist";
  }
}

void ProfilingUtils::SetGraphProfilingCNode(uint32_t graph_id, const std::vector<CNodePtr> &profiling_cnode_list) {
  auto ret = graph_profiling_cnode_.try_emplace(graph_id, profiling_cnode_list);
  if (!ret.second) {
    MS_LOG(ERROR) << "[profiling]graph " << graph_id << " profiling cnode list already exist";
  }
}

bool ProfilingUtils::ValidComputeGraph(NotNull<const session::KernelGraph *> graph_ptr) {
  for (const auto &node : graph_ptr->execution_order()) {
    if (AnfAlgo::GetKernelType(node) == TBE_KERNEL) {
      return true;
    }
  }
  return false;
}

void ProfilingUtils::ReportProfilingData(const std::vector<uint32_t> &task_ids,
                                         NotNull<const session::KernelGraph *> graph) {
  if (!ValidComputeGraph(graph)) {
    MS_LOG(WARNING) << "Not a valid compute graph:" << graph->graph_id();
    return;
  }

  auto ret = graph_profiling_cnode_.find(graph->graph_id());
  if (ret == graph_profiling_cnode_.end()) {
    MS_LOG(ERROR) << "Graph id not found";
    return;
  }

  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  TaskDescReporter task_reporter(context->device_id(), "vm.task_desc_info_" + std::to_string(graph->graph_id()),
                                 ret->second);
  task_reporter.set_task_ids(task_ids);
  task_reporter.ReportData();

  GraphDescReporter graph_reporter(context->device_id(), "vm.graph_desc_info_" + std::to_string(graph->graph_id()),
                                   ret->second);
  graph_profiling_cnode_.erase(ret);
  graph_reporter.ReportData();
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
