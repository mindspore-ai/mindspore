/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/hal/profiler/gpu_profiling_utils.h"
#include "kernel/kernel.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "utils/ms_utils.h"
#include "utils/ms_context.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace profiler {
namespace gpu {
constexpr char kFpStartNode[] = "PROFILING_FP_START";
constexpr char kBpEndNode[] = "PROFILING_BP_END";
constexpr char kIterEndNode[] = "PROFILING_ITER_END";
constexpr auto kInitDatasetQueueOpName = "InitDataSetQueue";

bool ProfilingUtils::have_communication_op = false;
ProfilingTraceInfo ProfilingUtils::profiling_trace = {"", "", ""};
std::unordered_map<uint32_t, bool> ProfilingUtils::is_first_step_map_ = {};

ProfilingTraceInfo ProfilingUtils::GetProfilingTraceFromEnv(NotNull<const session::KernelGraph *> graph_ptr) {
  MS_LOG(INFO) << "get current subgraph op name start.";
  auto &cnode_exec_order = graph_ptr->execution_order();
  if (cnode_exec_order.empty()) {
    return profiling_trace;
  }

  ProfilingTraceInfo empty_info;
  ProfilingTraceInfo last_graph_profiling_trace = profiling_trace;
  profiling_trace = empty_info;
  SetTraceIterEnd(cnode_exec_order);
  SetTraceFpStart(cnode_exec_order);
  SetTraceBpEnd(cnode_exec_order);
  GetTraceHccl(cnode_exec_order);

  OutputStepTraceOpNameStatus();
  is_first_step_map_[graph_ptr->graph_id()] = false;

  // If current graph has only one node, the bp_end will be empty, so select the last graph node.
  if (profiling_trace.trace_bp_end != "") {
    return profiling_trace;
  } else {
    return last_graph_profiling_trace;
  }
}

void ProfilingUtils::OutputStepTraceOpNameStatus() {
  if (profiling_trace.IsValid()) {
    MS_LOG(INFO) << "Get all the step_trace op name.";
  }
  MS_LOG(INFO) << "[profiling]trace_fp_start: " << profiling_trace.trace_fp_start
               << "trace_bp_end: " << profiling_trace.trace_bp_end
               << "trace_iter_end: " << profiling_trace.trace_iter_end;
  MS_LOG(INFO) << "get step_trace op name end.";
}

void ProfilingUtils::GetTraceHccl(const std::vector<CNodePtr> &cnode_exec_order) {
  for (const auto &node : cnode_exec_order) {
    if (common::AnfAlgo::IsCommunicationOp(node)) {
      MS_EXCEPTION_IF_NULL(node);
      if (std::find(profiling_trace.trace_custom_node.begin(), profiling_trace.trace_custom_node.end(),
                    node->fullname_with_scope()) == profiling_trace.trace_custom_node.end()) {
        profiling_trace.trace_custom_node.push_back(node->fullname_with_scope());
      }
      MS_LOG(INFO) << "[profiling]Get hccl node:" << node->fullname_with_scope();
    }
  }
}

void ProfilingUtils::SetTraceFpStart(const std::vector<CNodePtr> &cnode_exec_order) {
  const char *trace_fp_start = std::getenv(kFpStartNode);
  if (trace_fp_start != nullptr) {
    profiling_trace.trace_fp_start = std::string(trace_fp_start);
    MS_LOG(INFO) << "Set the Fp Start Op Name from Environment Variable:" << profiling_trace.trace_fp_start;
    return;
  }

  auto first_node = cnode_exec_order.front();
  MS_EXCEPTION_IF_NULL(first_node);
  auto node_name = common::AnfAlgo::GetCNodeName(first_node);
  if (node_name == kInitDatasetQueueOpName) {
    return;
  }

  if (node_name == kGetNextOpName) {
    if (cnode_exec_order.size() > 1) {
      profiling_trace.trace_fp_start = cnode_exec_order.at(1)->fullname_with_scope();
    } else {
      MS_LOG(WARNING) << "No Op Behind the GetNext Op" << std::endl;
    }
  } else {
    profiling_trace.trace_fp_start = first_node->fullname_with_scope();
  }
}

void ProfilingUtils::SetTraceBpEnd(const std::vector<CNodePtr> &cnode_exec_order) {
  const char *trace_bp_end = std::getenv(kBpEndNode);
  if (trace_bp_end != nullptr) {
    profiling_trace.trace_bp_end = std::string(trace_bp_end);
    MS_LOG(INFO) << "Set the Bp End Op Name from Environment Variable:" << profiling_trace.trace_bp_end;
    return;
  }

  std::string bp_end_str;
  // Contain hccl kernel (try to find the last communication op)
  auto iter = cnode_exec_order.rbegin();
  while (iter != cnode_exec_order.rend()) {
    if (common::AnfAlgo::IsCommunicationOp(*iter)) {
      break;
    }
    ++iter;
  }
  // If find the communication op
  if (iter != cnode_exec_order.rend()) {
    // store communication op input nodes' name
    std::set<std::string> ar_input_node_names;
    size_t input_num = common::AnfAlgo::GetInputTensorNum(*iter);
    for (size_t i = 0; i < input_num; ++i) {
      auto input_node_with_index = common::AnfAlgo::GetPrevNodeOutput(*iter, i);
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
  }

  if (bp_end_str.empty() && !have_communication_op) {
    bp_end_str = GetGraphSecondLastKernelName(cnode_exec_order);
  }

  if (!bp_end_str.empty()) {
    profiling_trace.trace_bp_end = bp_end_str;
  }
}

void ProfilingUtils::SetTraceIterEnd(const std::vector<CNodePtr> &cnode_exec_order) {
  const char *trace_iter_end = std::getenv(kIterEndNode);
  if (trace_iter_end != nullptr) {
    profiling_trace.trace_iter_end = std::string(trace_iter_end);
    MS_LOG(INFO) << "Set the Iter End Op Name from Environment Variable:" << profiling_trace.trace_iter_end;
    return;
  }

  auto iter_end = cnode_exec_order.rbegin();
  profiling_trace.trace_iter_end = (*iter_end)->fullname_with_scope();
}

std::string ProfilingUtils::GetGraphSecondLastKernelName(const std::vector<CNodePtr> &cnode_exec_order) {
  std::string second_last_kernel_name;
  auto iter = cnode_exec_order.rbegin();
  ++iter;
  if (iter == cnode_exec_order.rend()) {
    --iter;
  }
  second_last_kernel_name = (*iter)->fullname_with_scope();
  return second_last_kernel_name;
}

bool ProfilingUtils::IsFirstStep(const uint32_t graph_id) {
  auto iter = is_first_step_map_.find(graph_id);
  if (iter == is_first_step_map_.end()) {
    is_first_step_map_[graph_id] = false;
    return true;
  }
  return is_first_step_map_[graph_id];
}
}  // namespace gpu
}  // namespace profiler
}  // namespace mindspore
