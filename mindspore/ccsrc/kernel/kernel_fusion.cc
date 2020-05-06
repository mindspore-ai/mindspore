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

#include "kernel/kernel_fusion.h"

#include <map>
#include <string>
#include <memory>
#include <utility>

#include "common/utils.h"
#include "kernel/tbe/tbe_kernel_build.h"
#include "kernel/tbe/tbe_kernel_parallel_build.h"
#include "kernel/tbe/tbe_utils.h"
#include "kernel/tbe/tbe_convert_utils.h"

namespace mindspore {
namespace kernel {
using mindspore::kernel::tbe::TbeUtils;
static bool GenPreBuildKernelJson(const std::vector<AnfNodePtr> &compute_nodes,
                                  std::vector<nlohmann::json> *prebuild_op_list) {
  MS_EXCEPTION_IF_NULL(prebuild_op_list);
  TbeKernelJsonCreator creator(PREBUILD);
  for (const auto &anf_node : compute_nodes) {
    nlohmann::json prebuild;
    if (!creator.GenTbeSingleKernelJson(anf_node, &prebuild)) {
      MS_LOG(ERROR) << "GenTbeSingleKernelJson failed";
      return false;
    }
    (*prebuild_op_list).push_back(prebuild);
  }
  return true;
}

std::map<int32_t, KernelModPtr> KernelFusion(const std::vector<FusionScopeInfo> &fusion_scopes) {
  MS_LOG(INFO) << "kernel fusion build start, scope size:" << fusion_scopes.size();
  std::map<int32_t, KernelModPtr> kernel_mod_ret;
  auto build_manger = std::make_shared<ParallelBuildManager>();
  MS_EXCEPTION_IF_NULL(build_manger);
  for (const auto &fusion_scope_iter : fusion_scopes) {
    auto scope_id = fusion_scope_iter.scope_id;
    nlohmann::json fusion_op;
    string fusion_kernel = "te_fusion";
    if (!TbeKernelBuild::GenFusionScopeJson(fusion_scope_iter.input_nodes, fusion_scope_iter.compute_nodes, &fusion_op,
                                            &fusion_kernel)) {
      continue;
    }
    // gen kernel_name & check cache
    std::string json_str = fusion_op.dump();
    size_t hash_id = std::hash<std::string>()(json_str);
    auto json_name = fusion_kernel.append("_").append(std::to_string(hash_id));
    fusion_op["fusion_op_name"] = json_name;
    // gen json for prebuild
    std::vector<nlohmann::json> prebuild_op_list;
    if (!GenPreBuildKernelJson(fusion_scope_iter.compute_nodes, &prebuild_op_list)) {
      continue;
    }
    // get io size
    std::vector<size_t> input_size_list;
    std::vector<size_t> output_size_list;
    if (!TbeKernelBuild::GetIOSize(fusion_op["op_list"], fusion_scope_iter.output_nodes, &input_size_list,
                                   &output_size_list)) {
      continue;
    }
    // search cache
    auto kernel_pack = TbeUtils::SearchCache(json_name, tbe::kProcessorAiCore);
    if (kernel_pack != nullptr) {
      MS_LOG(INFO) << "Use cached kernel, kernel json name: " << json_name;
      auto kernel_mod =
        build_manger->GenKernelMod(json_name, tbe::kProcessorAiCore, input_size_list, output_size_list, kernel_pack);
      if (kernel_mod != nullptr) {
        kernel_mod_ret[scope_id] = kernel_mod;
        continue;
      }
    }
    // fusion build
    nlohmann::json fusion_json;
    fusion_json["fusion_op"] = fusion_op;
    fusion_json["prebuild_ops"] = prebuild_op_list;
    auto task_id = build_manger->StartCompileOp(fusion_json);
    TbeUtils::SaveJsonInfo(json_name, fusion_json.dump());
    if (task_id < 0) {
      MS_EXCEPTION(ArgumentError) << "start compile failed.";
    }
    build_manger->SaveTaskInfo(task_id, nullptr, json_name, input_size_list, output_size_list, scope_id);
  }

  int build_failed_num = 0;
  while (!build_manger->IsAllTaskFinish()) {
    int task_id = -1;
    char *task_result = nullptr;
    auto ret = build_manger->WaitOne(&task_id, &task_result);
    if (!ret) {
      MS_EXCEPTION(ArgumentError) << "Build Failed. wait one ret:" << ret << ", task id:" << task_id;
    }

    if ((task_result != nullptr) && (strcmp(task_result, "Success") != 0)) {
      MS_LOG(INFO) << "Fusion warning: Fuison op build failed, err log: " << task_result
                   << "  change to single op build.";
      build_failed_num++;
    }
    auto kernel_mod_item = build_manger->TaskFinishProcess(task_id, false);
    if (kernel_mod_item.second != nullptr) {
      (void)kernel_mod_ret.emplace(kernel_mod_item);
    }
  }
  MS_LOG(INFO) << "Build Fusion Kernel Failed Num: " << build_failed_num;
  return kernel_mod_ret;
}
}  // namespace kernel
}  // namespace mindspore
