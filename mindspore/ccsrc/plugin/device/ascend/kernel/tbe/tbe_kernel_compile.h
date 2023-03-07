/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_TBE_KERNEL_COMPILE_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_TBE_KERNEL_COMPILE_H_
#include <string>
#include <map>
#include <tuple>
#include <set>
#include <memory>
#include <vector>
#include <utility>
#include "ir/anf.h"
#include "kernel/kernel.h"
#include "kernel/kernel_fusion.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_build.h"
#include "include/backend/kernel_graph.h"

namespace mindspore {
namespace kernel {
namespace ascend {
using JsonNameMap = std::map<int64_t, std::string>;
struct TargetJobStatus {
  int target_job_id;
  std::string job_status;
  std::string except_msg;
  std::string json_name;
};

struct TaskInfo {
  std::string json_name;
  std::string full_name;
  nlohmann::json build_json;
  int task_id;
  bool is_dynamic = false;
  int64_t scope_id;
};

struct PreBuildResult {
  std::string core_type;
  std::string json_name;
  std::string fusion_type;
  nlohmann::json output_data_desc;
};

struct KernelIOSizeInfo {
  std::string json_name;
  std::vector<size_t> input_size_list;
  std::vector<size_t> output_size_list;
};

class TbeKernelCompileManager {
 public:
  static TbeKernelCompileManager &GetInstance() {
    static TbeKernelCompileManager instance;
    instance.TbeInitialize();
    return instance;
  }
  void TbeFinalize();
  void TbeInitialize();
  // check support
  bool TbeOpCheckSupported(const CNodePtr &node, nlohmann::json *kernel_json) const;
  // kernel select
  std::string TbeOpSelectFormat(const CNodePtr &node) const;
  // pre build
  void TbePreBuild(const KernelGraphPtr &kernel_graph);
  // single op compile
  std::pair<std::vector<CNodePtr>, std::vector<CNodePtr>> TbeSingleOpCompile(const std::vector<CNodePtr> &node_list);
  // fusion op compile
  JsonNameMap TbeFusionOpCompile(const std::vector<FusionScopeInfo> &fusion_scopes);
  void ClearFailedLog() { failed_log_.clear(); }
  std::string failed_log() { return failed_log_; }

 private:
  TbeKernelCompileManager() = default;
  ~TbeKernelCompileManager();
  // tbe kernel build client interface
  std::string DispatchCompileTask(const nlohmann::json &kernel_json) const;
  // save all build task: pre-build, single-build, fusion-build
  void SaveTaskInfo(const bool is_dynamic, const nlohmann::json &json, const std::string &json_name,
                    const std::string &full_name, int task_id, int64_t scope_id);
  // after job build, save some result info
  // for 'pre-build', save op-pattern and output_data_desc;
  // for 'dynamic', save compile_res
  void SaveSucceedTaskCompileResult(int task_id, const std::string &compile_info, const std::string &job_type);
  // save not support op
  void SaveFailedTaskCompileResult(int task_id);
  // save op-pattern and output_data_desc;
  void SavePreBuildResult(const std::string &json_name, const std::string &pre_build_result);
  // load tbe prebuild result from cached json file
  void LoadPreBuildResult();
  // load not support op
  void LoadNotSupportFusionOp();
  // query all build task
  void Query(const std::string &type);
  // single op build/pre-build
  void QueryProcess(const std::string &type, const std::string &job_result, std::vector<int> *success_job,
                    std::vector<int> *failed_job);
  void GetAllTbeNodes(const std::shared_ptr<session::KernelGraph> &kernel_graph,
                      std::vector<CNodePtr> *tbe_nodes) const;
  void PrintProcessLog(const nlohmann::json &json, int adjust_log_level) const;
  void JsonAssemble(const std::string &job_type, const nlohmann::json &src_json, nlohmann::json *dst_json) const;
  void PrintInitResult(const nlohmann::json &json) const;
  void PrintCompileResult(const nlohmann::json &json);
  std::string ParseSelectAndCheckResult(const nlohmann::json &json, const CNodePtr &node) const;
  void ParseTargetJobStatus(const nlohmann::json &json, TargetJobStatus *target_status) const;
  std::string ParseOpPattern(const std::string &json_str) const;
  nlohmann::json TurnStrToJson(const std::string &string) const;
  void SaveIOSizeInfo(const nlohmann::json &json, const std::string &json_name,
                      const std::vector<AnfNodePtr> &output_nodes = {});
  void ClearOldTask();
  void UpdateFusionTypeAndOutputDataDesc(const std::vector<CNodePtr> &nodes);
  JsonNameMap GetAllSuccessFusion();
  std::pair<std::vector<CNodePtr>, std::vector<CNodePtr>> GenKernelMod(const std::vector<CNodePtr> &node_list);
  void DistributeCompileTask(const std::vector<CNodePtr> &node_list, const std::string &job_type);
  void DistributePreBuildTask(const std::vector<CNodePtr> &node_list);

  // init flag
  static bool tbe_init_flag_;
  // tune flag
  static bool is_tune_flag_;
  // single op had build
  std::set<std::string> single_processed_kernels_;
  // single op had pre build
  std::set<std::string> pre_build_single_processed_kernels_;
  // fusion op had build
  std::set<std::string> fusion_processed_kernels_;
  // if op_debug_level is one of [1, 2, 4], skip tbe compile cache and rebuild again.
  std::string op_debug_level_;
  // if op_debug_config is not empty, skip tbe compile cache and rebuild again.
  std::string op_debug_config_;
  // id_node pair for node trace
  std::map<int, CNodePtr> job_id_to_node_;
  // id_task, all build jobs
  std::map<int, TaskInfo> task_map_;
  // pre build result
  std::map<std::string, PreBuildResult> prebuild_res_map_;
  // not support fusion op
  std::set<std::string> not_support_fusion_ops_;
  // using full_name to find json_name when update fusion type and out data desc
  std::map<std::string, std::string> pre_build_full_name_to_json_name_;
  // save io size for kernel mod
  std::map<std::string, KernelIOSizeInfo> kernel_io_size_info_;
  // using full_name to find json_name when gen kernel mod
  std::map<std::string, std::string> full_name_to_json_name_;
  // for fusion op
  JsonNameMap success_fusion_ops_;
  JsonNameMap all_fusion_ops_;
  // build failed log
  std::string failed_log_;
};
}  // namespace ascend
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_TBE_KERNEL_COMPILE_H_
