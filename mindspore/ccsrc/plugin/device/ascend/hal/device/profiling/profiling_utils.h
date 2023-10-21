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
#include "toolchain/prof_api.h"
#include "runtime/rt_model.h"

namespace mindspore {
namespace device {
namespace ascend {
struct ProfilingTraceInfo {
  // (trace_begin) -> FP -> BP -> (trace_bp_end) -> OPTIMIZER -> (trace_iter_end)
  std::string trace_begin;
  std::set<std::string> trace_bp_end;
  std::set<std::string> trace_iter_end;

  // profiling specific op, such as AllReduce
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

struct TensorInfoWrapper {
  MsprofAdditionalInfo tensor_info;
  uint64_t tensor_num;
};

struct ProfNodeAdditionInfo {
  MsprofCompactInfo node_basic_info;
  std::vector<TensorInfoWrapper> tensor_info_wrappers;
  MsprofApi api;
};

enum class GeProfInfoType {
  // model level
  kModelExecute = MSPROF_REPORT_MODEL_GRAPH_ID_MAP_TYPE + 1,
  kModelLoad,
  kInputCopy,
  kOutputCopy,
  kModelLevelEnd,
  // node level
  kInferShape = MSPROF_REPORT_NODE_GE_API_BASE_TYPE + 1,
  kCompatibleInferShape,
  kTiling,
  kCompatibleTiling,
  kStreamSync,
  kStepInfo,
  kTaskMemoryInfo,
  kEnd
};

constexpr uint32_t kTensorInfoBytes = 44UL;
constexpr uint32_t kTensorInfoBytesWithCap = 56U;

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

  static uint64_t GetMsprofHashId(const std::string &info);
  static void GetGraphNodes(const session::KernelGraph &kernel_graph);
  static void BuildSingleTensorInfo(const CNodePtr &node, const uint64_t opName_hash_id, const size_t index,
                                    const uint32_t tensor_num, TensorInfoWrapper *tensor_info_wrapper);
  static void InitProfTensorData(const CNodePtr &node, const size_t index, const uint64_t offset_idx,
                                 MsprofTensorInfo *tensor_info);
  static void ReportTask(const std::string &op_name, const bool is_op_name);
  static uint32_t GetBlockDim(const CNodePtr &node);
  static void RecordLaunchTaskBegin(const std::string &op_name, const bool is_op_name);
  static void InitLaunchApi(const uint64_t name_hash, MsprofApi *api);
  static void RecordModelLoad(const rtModel_t rt_model_handle);
  static void RecordModelExecute(const KernelGraphPtr kernel_graph);
  static void RegisterProfType();
  static void InitReportNode(const CNodePtr &cnode, bool init_begin_time = false);

  inline static constexpr char kProfiling[] = "Profiling";
  inline static constexpr char kNotify[] = "notify";
  inline static constexpr char kProfilerTraceId[] = "profiler_trace_id";
  inline static constexpr char kFlags[] = "flags";
  static std::mutex profiler_mutex;

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
  static std::string GetFullScopeName(const std::string &op_name, const bool is_op_name);

  // graph id --> (kernel name list)
  inline static std::map<uint32_t, std::vector<CNodePtr>> graph_profiling_cnode_;
  inline static std::map<uint32_t, std::vector<std::string>> graph_kernel_name_;
  inline static std::map<uint32_t, std::vector<std::shared_ptr<StepPointDesc>>> graph_point_;
  inline static uint32_t custom_node_index_;
  inline static std::vector<GraphProfilingData> report_data_;

  inline static std::map<std::string, uint64_t> msprof_hash_id_;
  inline static std::map<std::string, ProfNodeAdditionInfo> node_addition_info_;
  inline static std::map<std::string, uint64_t> task_launch_begin_;
  inline static bool is_prof_type_registered_ = False;

  inline static std::vector<MsprofEvent> report_event_;
  inline static std::vector<MsprofCompactInfo> report_compact_info_;
  inline static std::vector<MsprofAdditionalInfo> report_additional_info_;
  inline static std::vector<MsprofApi> report_api_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEN_D_PROFILING_PROFILING_UTILS_H_
