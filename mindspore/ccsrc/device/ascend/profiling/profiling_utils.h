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
#ifndef MINDSPORE_MINDSPORE_CCSRC_DEVICE_ASCEND_PROFILING_PROFILING_UTILS_H_
#define MINDSPORE_MINDSPORE_CCSRC_DEVICE_ASCEND_PROFILING_PROFILING_UTILS_H_

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "session/kernel_graph.h"

namespace mindspore {
namespace device {
namespace ascend {
struct ProfilingTraceInfo {
  // execute order's first execute op(like: Cast or Four2Five ...), except tdt op(GetNext ...)
  std::string profiling_trace_begin;
  // get first net_output(apply kernel) from graph outputs: fp ->net_output<- bp
  std::string profiling_trace_bp_end;
  // execute order's end execute (like: Conv2DBackpropFilter)
  std::string profiling_trace_netoutput;

  std::string profiling_allreduce1_start;

  std::string profiling_allreduce1_end;

  std::string profiling_allreduce2_start;

  std::string profiling_allreduce2_end;

  // 1. insert profiling_trace_begin if profiling_trace_bp_end is not empty.
  // 2. op lanuch get task info with callback func.
  // 3. insert profiling_trace_bp_end.
  // 4. insert profiling_trace_net_output if profiling_trace_bp_end is not empty.

  bool IsValid() const { return !(profiling_trace_begin.empty() || profiling_trace_bp_end.empty()); }
};

class ProfilingUtils {
 public:
  ProfilingUtils() = default;
  ~ProfilingUtils() = default;
  static bool GetProfilingTraceInfo(const std::shared_ptr<session::KernelGraph> &graph_ptr,
                                    ProfilingTraceInfo *profiling_trace_info);
  static void ProfilingTraceFpStart(const std::shared_ptr<session::KernelGraph> &graph_ptr, const AnfNodePtr &anf_node,
                                    const ProfilingTraceInfo &profiling_trace_info, std::vector<CNodePtr> *kernel_list);
  static void ProfilingAllReduce(const std::shared_ptr<session::KernelGraph> &graph_ptr, const AnfNodePtr &anf_node,
                                 int job_id, const std::string &profiling_node_name,
                                 std::vector<CNodePtr> *kernel_list);
  static void ProfilingTraceEnd(const std::shared_ptr<session::KernelGraph> &graph_ptr, const AnfNodePtr &anf_node,
                                const ProfilingTraceInfo &profiling_trace_info, std::vector<CNodePtr> *kernel_list);
  static void SetGraphKernelName(uint32_t graph_id, const std::vector<std::string> &kernel_names);
  static void ReportProfilingData(uint32_t graph_id, const std::vector<uint32_t> &task_ids);

  static const char kProfiling[];
  static const char kNotify[];
  static const char kProfilerTraceId[];
  static const char kFlags[];

 private:
  static bool GetNetOutput(AnfNodePtr anf_node, std::string *profiling_trace_net_output);
  static CNodePtr CreateProfilingCNode(const std::shared_ptr<session::KernelGraph> &graph_ptr, bool notify,
                                       uint64_t profiler_trace_id, uint32_t flags);
  // graph id --> (kernel name list)
  static std::unordered_map<uint32_t, std::vector<std::string>> graph_kernel_name_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_DEVICE_ASCEND_PROFILING_PROFILING_UTILS_H_
