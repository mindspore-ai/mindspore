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

#include "device/ascend/profiling/profiling_utils.h"

#include <map>

#include "kernel/kernel.h"
#include "device/ascend/profiling/profiling_manager.h"
#include "session/anf_runtime_algorithm.h"
#include "common/utils.h"
#include "utils/utils.h"

namespace mindspore {
namespace device {
namespace ascend {
const char ProfilingUtils::kProfiling[] = "Profiling";
const char ProfilingUtils::kNotify[] = "notify";
const char ProfilingUtils::kProfilerTraceId[] = "profiler_trace_id";
const char ProfilingUtils::kFlags[] = "flags";
std::unordered_map<uint32_t, std::vector<std::string>> ProfilingUtils::graph_kernel_name_;
bool ProfilingUtils::GetProfilingTraceInfo(const std::shared_ptr<session::KernelGraph> &graph_ptr,
                                           ProfilingTraceInfo *profiling_trace_info) {
  MS_EXCEPTION_IF_NULL(profiling_trace_info);
  MS_EXCEPTION_IF_NULL(graph_ptr);
  bool find_begin = false;
  bool first_allreduce = true;
  for (const auto &anf_node : graph_ptr->execution_order()) {
    if (anf_node->isa<CNode>()) {
      const std::string kernel_name = AnfAlgo::GetCNodeName(anf_node);
      if ((kernel_name == "Cast" || kernel_name == "Four2Five") && !find_begin) {
        profiling_trace_info->profiling_trace_begin = anf_node->fullname_with_scope();
        find_begin = true;
      }
      if (kernel_name == "Conv2DBackpropFilter") {
        profiling_trace_info->profiling_trace_bp_end = anf_node->fullname_with_scope();
      }
      if (kernel_name == kFusedMulApplyMomentumOpName || kernel_name == kApplyMomentumOpName) {
        profiling_trace_info->profiling_trace_netoutput = anf_node->fullname_with_scope();
      }
      if (kernel_name == kAllReduceOpName) {
        if (first_allreduce) {
          profiling_trace_info->profiling_allreduce1_start = anf_node->fullname_with_scope();
          profiling_trace_info->profiling_allreduce1_end = anf_node->fullname_with_scope();
          first_allreduce = false;
        } else {
          profiling_trace_info->profiling_allreduce2_start = anf_node->fullname_with_scope();
          profiling_trace_info->profiling_allreduce2_end = anf_node->fullname_with_scope();
        }
      }
    }
  }
  MS_LOG(INFO) << "[profiling]begin:" << profiling_trace_info->profiling_trace_begin
               << ", net_output:" << profiling_trace_info->profiling_trace_netoutput
               << ", end:" << profiling_trace_info->profiling_trace_bp_end
               << ", allreduce1:" << profiling_trace_info->profiling_allreduce1_start
               << ", allreduce2:" << profiling_trace_info->profiling_allreduce2_start;
  return profiling_trace_info->IsValid();
}

bool ProfilingUtils::GetNetOutput(AnfNodePtr anf_node, std::string *profiling_trace_net_output) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(profiling_trace_net_output);
  MS_LOG(INFO) << "[profiling]Anf node's full name with scope:" << anf_node->fullname_with_scope();
  if (!profiling_trace_net_output->empty()) {
    MS_LOG(INFO) << "[profiling]Has got the net_output:" << profiling_trace_net_output->c_str();
    return true;
  }

  if (AnfAlgo::IsRealKernel(anf_node)) {
    *profiling_trace_net_output = anf_node->fullname_with_scope();
    return true;
  }

  auto cnode = anf_node->cast<CNodePtr>();
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "[profiling]Anf node should be a CNode";
    return false;
  }

  auto inputs = cnode->inputs();
  auto input_size = inputs.size();
  if (input_size < 2) {
    MS_LOG(ERROR) << "[profiling]Anf node' input size(" << input_size << ") < 2, don't support get apply kernel node.";
    return false;
  }
  return GetNetOutput(inputs[1], profiling_trace_net_output);
}

CNodePtr ProfilingUtils::CreateProfilingCNode(const std::shared_ptr<session::KernelGraph> &graph_ptr, bool notify,
                                              uint64_t profiler_trace_id, uint32_t flags) {
  MS_EXCEPTION_IF_NULL(graph_ptr);
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
  ValuePtr notify_value = MakeValue(notify);
  ValuePtr trace_id_value = MakeValue(profiler_trace_id);
  ValuePtr flags_value = MakeValue(flags);
  AnfAlgo::SetNodeAttr(ProfilingUtils::kNotify, notify_value, cnode_ptr);
  AnfAlgo::SetNodeAttr(ProfilingUtils::kProfilerTraceId, trace_id_value, cnode_ptr);
  AnfAlgo::SetNodeAttr(ProfilingUtils::kFlags, flags_value, cnode_ptr);
  return cnode_ptr;
}

void ProfilingUtils::ProfilingTraceFpStart(const std::shared_ptr<mindspore::session::KernelGraph> &graph_ptr,
                                           const mindspore::AnfNodePtr &anf_node,
                                           const mindspore::device::ascend::ProfilingTraceInfo &profiling_trace_info,
                                           std::vector<mindspore::CNodePtr> *kernel_list) {
  if (profiling_trace_info.IsValid() && profiling_trace_info.profiling_trace_begin == anf_node->fullname_with_scope()) {
    if (graph_ptr == nullptr || kernel_list == nullptr || anf_node == nullptr) {
      MS_LOG(ERROR) << "[profiling]input param invalid";
      return;
    }
    auto job_id = ProfilingManager::GetInstance().GetJobId();
    // job task info
    CNodePtr job_kernel_ptr = CreateProfilingCNode(graph_ptr, false, job_id, 0);
    AnfAlgo::SetStreamDistinctionLabel(AnfAlgo::GetStreamDistinctionLabel(anf_node.get()), job_kernel_ptr.get());
    AnfAlgo::SetStreamId(AnfAlgo::GetStreamId(anf_node), job_kernel_ptr.get());
    // fp task info
    CNodePtr start_kernel_ptr = CreateProfilingCNode(graph_ptr, false, kProfilingFpStartLogId, 0);
    AnfAlgo::SetStreamDistinctionLabel(AnfAlgo::GetStreamDistinctionLabel(anf_node.get()), start_kernel_ptr.get());
    AnfAlgo::SetStreamId(AnfAlgo::GetStreamId(anf_node), start_kernel_ptr.get());
    kernel_list->emplace_back(job_kernel_ptr);
    kernel_list->emplace_back(start_kernel_ptr);
  }
}

void ProfilingUtils::ProfilingAllReduce(const std::shared_ptr<session::KernelGraph> &graph_ptr,
                                        const AnfNodePtr &anf_node, int job_id, const std::string &profiling_node_name,
                                        std::vector<CNodePtr> *kernel_list) {
  MS_EXCEPTION_IF_NULL(graph_ptr);
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(kernel_list);
  auto full_scope_name = anf_node->fullname_with_scope();
  if (profiling_node_name == full_scope_name) {
    CNodePtr allreduce_kernel_ptr = CreateProfilingCNode(graph_ptr, false, job_id, 0);
    AnfAlgo::SetStreamDistinctionLabel(AnfAlgo::GetStreamDistinctionLabel(anf_node.get()), allreduce_kernel_ptr.get());
    AnfAlgo::SetStreamId(AnfAlgo::GetStreamId(anf_node), allreduce_kernel_ptr.get());
    kernel_list->emplace_back(allreduce_kernel_ptr);
  }
}

void ProfilingUtils::ProfilingTraceEnd(const std::shared_ptr<mindspore::session::KernelGraph> &graph_ptr,
                                       const mindspore::AnfNodePtr &anf_node,
                                       const mindspore::device::ascend::ProfilingTraceInfo &profiling_trace_info,
                                       std::vector<mindspore::CNodePtr> *kernel_list) {
  MS_EXCEPTION_IF_NULL(graph_ptr);
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(kernel_list);
  if (profiling_trace_info.IsValid()) {
    auto full_scope_name = anf_node->fullname_with_scope();
    if (profiling_trace_info.profiling_trace_netoutput == full_scope_name) {
      CNodePtr bp_kernel_ptr = CreateProfilingCNode(graph_ptr, true, kProfilingIterEndLogId, 0);
      AnfAlgo::SetStreamDistinctionLabel(AnfAlgo::GetStreamDistinctionLabel(anf_node.get()), bp_kernel_ptr.get());
      AnfAlgo::SetStreamId(AnfAlgo::GetStreamId(anf_node), bp_kernel_ptr.get());
      kernel_list->emplace_back(bp_kernel_ptr);
    }

    if (profiling_trace_info.profiling_trace_bp_end == full_scope_name) {
      CNodePtr end_task_info = CreateProfilingCNode(graph_ptr, false, kProfilingBpEndLogId, 0);
      AnfAlgo::SetStreamDistinctionLabel(AnfAlgo::GetStreamDistinctionLabel(anf_node.get()), end_task_info.get());
      AnfAlgo::SetStreamId(AnfAlgo::GetStreamId(anf_node), end_task_info.get());
      kernel_list->emplace_back(end_task_info);
    }
  }
}

void ProfilingUtils::SetGraphKernelName(uint32_t graph_id, const std::vector<std::string> &kernel_names) {
  auto iter = graph_kernel_name_.find(graph_id);
  if (iter == graph_kernel_name_.end()) {
    graph_kernel_name_[graph_id] = kernel_names;
  } else {
    MS_LOG(ERROR) << "[profiling]graph kernel names already exist";
  }
}

void ProfilingUtils::ReportProfilingData(uint32_t graph_id, const std::vector<uint32_t> &task_ids) {
  auto iter = graph_kernel_name_.find(graph_id);
  if (iter == graph_kernel_name_.end()) {
    MS_LOG(ERROR) << "[profiling]graph id " << graph_id << " not in graph_kernel_name_";
    return;
  }
  auto &kernel_names = iter->second;

  MS_LOG(INFO) << "kernel_names size:" << kernel_names.size() << ", task_ids size:" << task_ids.size();
  if (kernel_names.size() != task_ids.size()) {
    MS_LOG(ERROR) << "[profiling]kernel name and task id not match";
    return;
  }
  std::map<uint32_t, std::string> op_task_id_map;
  size_t size = kernel_names.size();
  for (size_t i = 0; i < size; ++i) {
    auto it = op_task_id_map.find(task_ids[i]);
    if (it != op_task_id_map.end()) {
      MS_LOG(WARNING) << "task_id " << task_ids[i] << " exist, " << kernel_names[i];
      continue;
    }
    op_task_id_map[task_ids[i]] = kernel_names[i];
  }
  if (!ProfilingManager::GetInstance().ReportProfilingData(op_task_id_map)) {
    MS_LOG(ERROR) << "ReportProfilingData failed";
  }
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
