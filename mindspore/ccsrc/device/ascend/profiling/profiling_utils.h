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
#include <set>
#include <unordered_map>
#include "session/kernel_graph.h"
#include "utils/contract.h"

namespace mindspore {
namespace device {
namespace ascend {
struct ProfilingTraceInfo {
  // execute order's first execute op(like: Cast or Four2Five ...), except tdt op(GetNext ...)
  std::string trace_begin;
  // get first net_output(apply kernel) from graph outputs: fp ->net_output<- bp
  std::string trace_bp_end;
  // execute order's end execute (like: Conv2DBackpropFilter)
  std::string trace_netoutput;

  // profiling specific op, such as AllReduce;
  std::set<std::string> trace_custom_node;

  // 1. insert profiling_trace_begin if profiling_trace_bp_end is not empty.
  // 2. op lanuch get task info with callback func.
  // 3. insert profiling_trace_bp_end.
  // 4. insert profiling_trace_net_output if profiling_trace_bp_end is not empty.

  bool IsValid() const { return !(trace_begin.empty() || trace_netoutput.empty()); }
};

struct ProfilingContent {
  // true -send data from device to host and finish profiling
  bool notify;
  uint64_t profiler_trace_id;
  uint32_t flags;
};

class ProfilingUtils {
 public:
  ProfilingUtils() = default;
  ~ProfilingUtils() = default;

  // Insert job_id profiling node and fp_start profiling node.
  // Job_id is got from envs, which shound be a number greater than 255
  // Fp_start node should been inserted in the start of a network, and the log_id is hard code to 1.
  static void ProfilingTraceFpStart(const AnfNodePtr &anf_node, const ProfilingTraceInfo &profiling_trace_info,
                                    NotNull<session::KernelGraph *> graph_ptr,
                                    NotNull<std::vector<CNodePtr> *> kernel_list);

  static void ProfilingTraceJobId(const AnfNodePtr &anf_node, NotNull<session::KernelGraph *> graph_ptr,
                                  NotNull<std::vector<CNodePtr> *> kernel_list);

  // Insert net output profiling node, which tells the device to stop profiling.
  // The notify in struct ProfilingContent should be 'true', which tells the device to send data to host.
  static void ProfilingTraceEnd(const AnfNodePtr &anf_node, const ProfilingTraceInfo &profiling_trace_info,
                                NotNull<session::KernelGraph *> graph_ptr,
                                NotNull<std::vector<CNodePtr> *> kernel_list);

  // Insert bp_end profiling node, which should been inserted after the last backpropagation CNode in the network.
  static void ProfilingTraceBpEnd(const mindspore::AnfNodePtr &anf_node, const ProfilingTraceInfo &profiling_trace_info,
                                  NotNull<session::KernelGraph *> graph_ptr,
                                  NotNull<std::vector<mindspore::CNodePtr> *> kernel_list);

  // Mapping graph id and the kernels' name in the graph
  static void SetGraphKernelName(uint32_t graph_id, const std::vector<std::string> &kernel_names);

  // Mapping task_id and kernel name for device to generate the time cost of specific kernel.
  // Device calculate the time cost of the task which is marked by task id.
  // But we need data of (kernel name , time cost)
  static void ReportProfilingData(uint32_t graph_id, const std::vector<uint32_t> &task_ids);

  // Get profiling trace point from envs.
  // export PROFILING_FP_START='full name of the first cnode to execute'
  // export PROFILING_BP_END='full name of the last backpropagation cnode to execute'
  // export PROFILING_ITER_END='full name of last cnode in graph to execute'
  // And other cnode, like AllReduce, export PROFILING_CUSTOM_1='full name of AllReduce cnode'
  // GetNext, export PROFIFLING_CUSTOM_2='full name fo GetNext cnode'
  // The variable i in PROFILING_CUSTOM_i should start from 1 without interruption.
  static ProfilingTraceInfo GetProfilingTraceFromEnv(NotNull<session::KernelGraph *> graph_ptr);

  // Insert two profiling trace points, one in front and one behind
  static void ProfilingCustomOp(const mindspore::AnfNodePtr &anf_node, const ProfilingTraceInfo &profiling_trace_info,
                                NotNull<session::KernelGraph *> graph_ptr,
                                NotNull<std::vector<mindspore::CNodePtr> *> kernel_list);

  inline static constexpr char kProfiling[] = "Profiling";
  inline static constexpr char kNotify[] = "notify";
  inline static constexpr char kProfilerTraceId[] = "profiler_trace_id";
  inline static constexpr char kFlags[] = "flags";

 private:
  static NotNull<CNodePtr> CreateProfilingCNode(const ProfilingContent &profiling_content,
                                                NotNull<session::KernelGraph *> graph_ptr);
  static CNodePtr CreateProfilingCNodeWithStream(const AnfNodePtr &anf_node, const ProfilingContent &profiling_content,
                                                 NotNull<session::KernelGraph *> graph_ptr);
  static std::string GetTraceBegin(const std::vector<CNodePtr> &cnode_exec_order);
  static std::string GetTraceBpEnd(const std::vector<CNodePtr> &cnode_exec_order);
  static std::string GetTraceNetoutput(const std::vector<CNodePtr> &cnode_exec_order);
  static std::string GetGraphLastTbeKernelName(const std::vector<CNodePtr> &cnode_exec_order);
  static void GetTraceHccl(const std::vector<CNodePtr> &cnode_exec_order,
                           NotNull<ProfilingTraceInfo *> profiling_trace);
  static void GetCNodeOutputRealNode(const std::string &node_name, const std::vector<CNodePtr> &cnode_exec_order,
                                     NotNull<std::set<std::string> *> getnext_outputs);

  // graph id --> (kernel name list)
  static std::unordered_map<uint32_t, std::vector<std::string>> graph_kernel_name_;
  static uint32_t custom_node_index_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_DEVICE_ASCEND_PROFILING_PROFILING_UTILS_H_
