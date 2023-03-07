/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_PROFILING_PROFILING_UTILS_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_PROFILING_PROFILING_UTILS_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <set>
#include <unordered_map>
#include "include/backend/kernel_graph.h"
#include "include/common/utils/contract.h"
#include "plugin/device/ascend/hal/device/profiling/profiling_reporter.h"

namespace mindspore {
namespace device {
namespace ascend {
struct ProfilingTraceInfo {
  // (trace_begin) -> FP -> BP -> (trace_bp_end) -> OPTIMIZER -> (trace_iter_end)
  std::string trace_begin;
  std::set<std::string> trace_bp_end;
  std::set<std::string> trace_iter_end;

  // profiling specific op, such as AllReduce;
  std::set<std::string> trace_custom_node;

  bool IsValid() const { return !(trace_begin.empty() || trace_iter_end.empty()); }
};

struct ProfilingContent {
  // true -send data from device to host and finish profiling
  bool notify;
  uint64_t profiler_trace_id;
  uint32_t flags;
};

struct GraphProfilingData {
  std::vector<uint32_t> task_ids_;
  std::vector<uint32_t> stream_ids_;
  uint32_t graph_id_;
  uint32_t rt_model_id;
};

class ProfilingUtils {
 public:
  ProfilingUtils() = default;
  ~ProfilingUtils() = default;

  static void InsertProfilingTraceFp(const AnfNodePtr &anf_node, const ProfilingTraceInfo &profiling_trace_info,
                                     NotNull<session::KernelGraph *> graph_ptr,
                                     NotNull<std::vector<CNodePtr> *> kernel_list);
  static void InsertProfilingTraceJobId(const AnfNodePtr &anf_node, NotNull<session::KernelGraph *> graph_ptr,
                                        NotNull<std::vector<CNodePtr> *> kernel_list);
  static void InsertProfilingTraceIterEnd(const AnfNodePtr &anf_node, const ProfilingTraceInfo &profiling_trace_info,
                                          NotNull<session::KernelGraph *> graph_ptr,
                                          NotNull<std::vector<CNodePtr> *> kernel_list);
  static void InsertProfilingTraceBpEnd(const mindspore::AnfNodePtr &anf_node,
                                        const ProfilingTraceInfo &profiling_trace_info,
                                        NotNull<session::KernelGraph *> graph_ptr,
                                        NotNull<std::vector<mindspore::CNodePtr> *> kernel_list);
  static void SetGraphProfilingCNode(uint32_t graph_id, const std::vector<CNodePtr> &profiling_cnode_list);
  static void SetGraphKernelName(uint32_t graph_id, const std::vector<std::string> &kernel_names);
  // Save graph information to Framework file
  static void ReportProfilingData(const std::vector<uint32_t> &task_ids, const std::vector<uint32_t> &stream_ids,
                                  uint32_t graph_id, uint32_t rt_model_id);
  // Report MindSpore Framework data to Ascend Profiler
  static void ReportMindSporeFrameworkData();
  // Generate profiling trace
  static ProfilingTraceInfo GenerateProfilingTrace(const session::KernelGraph &kernel_graph);

  // Insert two profiling trace points, one in front and one behind
  static void InsertProfilingCustomOp(const mindspore::AnfNodePtr &anf_node,
                                      const ProfilingTraceInfo &profiling_trace_info,
                                      NotNull<session::KernelGraph *> graph_ptr,
                                      NotNull<std::vector<mindspore::CNodePtr> *> kernel_list);

  static std::map<uint32_t, std::vector<std::string>> graph_kernel_name() { return graph_kernel_name_; }

  static void SetReportProfilingData(const std::vector<uint32_t> &task_ids, const std::vector<uint32_t> &stream_ids,
                                     uint32_t graph_id, uint32_t rt_model_id);
  static void ReportAllGraphProfilingData();
  static bool ValidComputeGraph(const session::KernelGraph &kernel_graph);

  inline static constexpr char kProfiling[] = "Profiling";
  inline static constexpr char kNotify[] = "notify";
  inline static constexpr char kProfilerTraceId[] = "profiler_trace_id";
  inline static constexpr char kFlags[] = "flags";

 private:
  static NotNull<CNodePtr> CreateProfilingCNode(const ProfilingContent &profiling_content,
                                                NotNull<session::KernelGraph *> graph_ptr);
  static CNodePtr CreateProfilingCNodeWithStream(const AnfNodePtr &anf_node, const ProfilingContent &profiling_content,
                                                 NotNull<session::KernelGraph *> graph_ptr);
  static void GetTraceBegin(const session::KernelGraph &kernel_graph, const nlohmann::json &option,
                            ProfilingTraceInfo *trace_info);
  static void GetTraceBpEnd(const session::KernelGraph &kernel_graph, const nlohmann::json &option,
                            ProfilingTraceInfo *trace_info);
  static void GetTraceIterEnd(const session::KernelGraph &kernel_graph, ProfilingTraceInfo *trace_info);
  static std::string GetGraphLastKernelName(const session::KernelGraph &kernel_graph);
  static void GetTraceHccl(const session::KernelGraph &kernel_graph, NotNull<ProfilingTraceInfo *> profiling_trace);
  static void GetCNodeOutputRealNode(const std::string &node_name, const session::KernelGraph &kernel_graph,
                                     NotNull<std::set<std::string> *> getnext_outputs);

  static void SaveProfilingPoint(uint32_t graph_id, const std::string &node_name, uint32_t point_id);

  // graph id --> (kernel name list)
  inline static std::map<uint32_t, std::vector<CNodePtr>> graph_profiling_cnode_;
  inline static std::map<uint32_t, std::vector<std::string>> graph_kernel_name_;
  inline static std::map<uint32_t, std::vector<std::shared_ptr<StepPointDesc>>> graph_point_;
  inline static uint32_t custom_node_index_;
  inline static std::vector<GraphProfilingData> report_data_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEN_D_PROFILING_PROFILING_UTILS_H_
