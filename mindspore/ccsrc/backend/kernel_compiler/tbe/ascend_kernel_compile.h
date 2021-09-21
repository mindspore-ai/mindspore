/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_ASCEND_KERNEL_COMPILE_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_ASCEND_KERNEL_COMPILE_H_
#include <string>
#include <map>
#include <tuple>
#include <set>
#include <memory>
#include <vector>
#include <utility>
#include "ir/anf.h"
#include "backend/kernel_compiler/kernel.h"
#include "backend/kernel_compiler/kernel_fusion.h"
#include "backend/kernel_compiler/tbe/tbe_kernel_build.h"
#include "backend/kernel_compiler/tbe/tbe_kernel_parallel_build.h"
#include "backend/session/kernel_graph.h"

namespace mindspore {
namespace kernel {
namespace ascend {
using KernelModMap = std::map<int64_t, KernelModPtr>;
struct TargetJobStatus {
  int target_job_id;
  std::string job_status;
  std::string except_msg;
  std::string json_name;
};

class AscendKernelCompileManager {
 public:
  static AscendKernelCompileManager &GetInstance() {
    static AscendKernelCompileManager instance;
    if (!instance.tbe_init_flag_) {
      instance.TbeInitialize();
    }
    return instance;
  }
  void TbeInitialize();
  void TbeFinalize();
  // kernel select
  std::string AscendOpSelectFormat(const AnfNodePtr &node);
  bool AscendOpCheckSupported(const AnfNodePtr &node);
  // pre build
  void AscendPreBuild(const std::shared_ptr<session::KernelGraph> &kernel_graph);
  // single op compile
  bool AscendSingleOpCompile(const std::vector<AnfNodePtr> &anf_nodes);
  // fusion op compile
  KernelModMap AscendFusionOpCompile(const std::vector<FusionScopeInfo> &fusion_scopes);
  // clear prev job's cache
  void ResetOldTask();

 private:
  AscendKernelCompileManager() = default;
  ~AscendKernelCompileManager();
  void GetAllAscendNodes(const std::shared_ptr<session::KernelGraph> &kernel_graph, std::vector<AnfNodePtr> *tbe_nodes);
  void QueryFinishJob(const std::string &type);
  void ParseTargetJobStatus(const std::string &type, const std::string &job_result, std::vector<int> *success_job);
  void QueryPreBuildFinishJob();
  void QueryFusionFinishJob(KernelModMap *kernel_mode_ret);
  void PrintProcessLog(const nlohmann::json &json, int adjust_log_level);
  void JsonAssemble(const std::string &job_type, const nlohmann::json &src_json, nlohmann::json *dst_json);
  void PrintInitResult(const nlohmann::json &json);
  void PrintCompileResult(const nlohmann::json &json);
  std::string OpSelectAndCheckResultProcess(const nlohmann::json &json, const AnfNodePtr &node);
  void QueryResultProcess(const nlohmann::json &json, TargetJobStatus *task_info);
  nlohmann::json TurnStrToJson(const std::string &str) const;

  static bool tbe_init_flag_;
  static bool is_tune_flag_;
  std::set<std::string> single_processed_kernels_;
  std::set<std::string> fusion_processed_kernels_;
  std::string op_debug_level_;  // if op_debug_level is "1", skip tbe compile cache and rebuild again.
  std::shared_ptr<ParallelBuildManager> build_manager_ = nullptr;
  std::map<int, nlohmann::json> job_list_;
  std::map<int, AnfNodePtr> job_id_to_node_;
  std::map<int, std::string> fusion_op_names_;
};
}  // namespace ascend
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_ASCEND_KERNEL_COMPILE_H_
