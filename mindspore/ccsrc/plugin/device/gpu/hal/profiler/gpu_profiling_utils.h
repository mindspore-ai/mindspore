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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_HAL_PROFILER_GPU_PROFILING_UTILS_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_HAL_PROFILER_GPU_PROFILING_UTILS_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <set>
#include <unordered_map>

#include "include/backend/kernel_graph.h"

namespace mindspore {
namespace profiler {
namespace gpu {
struct ProfilingTraceInfo {
  // support get all the op name from environment variable
  // fp start op is the first op in all subgraph except data related op
  std::string trace_fp_start;
  // bp end op is the input node op of the last communication op (if exist)
  std::string trace_bp_end;
  // iteration end op is the last executed op
  std::string trace_iter_end;

  // profiling specific op, such as AllReduce;
  std::vector<std::string> trace_custom_node;

  bool IsValid() const { return !(trace_fp_start.empty() || trace_bp_end.empty() || trace_iter_end.empty()); }
};

class ProfilingUtils {
 public:
  ProfilingUtils() = default;
  ~ProfilingUtils() = default;

  // Get profiling trace point from envs.
  // export PROFILING_FP_START='full name of the first cnode to execute'
  // export PROFILING_BP_END='full name of the last backpropagation cnode to execute'
  // export PROFILING_ITER_END='full name of last cnode in graph to execute'
  static ProfilingTraceInfo GetProfilingTraceFromEnv(NotNull<const session::KernelGraph *> graph_ptr);
  static void OutputStepTraceOpNameStatus();
  static bool IsFirstStep(const uint32_t graph_id);

  static bool have_communication_op;
  static ProfilingTraceInfo profiling_trace;

 private:
  static void SetTraceFpStart(const std::vector<CNodePtr> &cnode_exec_order);
  static void SetTraceBpEnd(const std::vector<CNodePtr> &cnode_exec_order);
  static void SetTraceIterEnd(const std::vector<CNodePtr> &cnode_exec_order);
  static std::string GetGraphSecondLastKernelName(const std::vector<CNodePtr> &cnode_exec_order);
  static void GetTraceHccl(const std::vector<CNodePtr> &cnode_exec_order);
  static std::unordered_map<uint32_t, bool> is_first_step_map_;
};
}  // namespace gpu
}  // namespace profiler
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_HAL_PROFILER_GPU_PROFILING_UTILS_H_
