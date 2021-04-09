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

#include "backend/kernel_compiler/kernel_fusion.h"

#include <map>
#include <set>
#include <string>
#include <memory>
#include "backend/kernel_compiler/tbe/tbe_kernel_build.h"
#include "backend/kernel_compiler/tbe/tbe_kernel_parallel_build.h"
#include "backend/kernel_compiler/tbe/tbe_utils.h"
#include "backend/kernel_compiler/tbe/tbe_convert_utils.h"
#include "utils/ms_context.h"

namespace mindspore::kernel {
using mindspore::kernel::tbe::TbeUtils;

static size_t GenFusionJsonHash(const nlohmann::json &fusion_json) {
  // get an copy
  nlohmann::json fusion_json_copy = fusion_json;
  auto &op_lists = fusion_json_copy["op_list"];
  for (auto &op : op_lists) {
    op.erase("name");
    for (auto &output_desc : op["output_desc"]) {
      output_desc.erase("name");
    }
    if (op["type"] != "Data") {
      for (auto &input_desc : op["input_desc"]) {
        input_desc.erase("name");
      }
      for (auto &list_arg : op["prebuild_outs_attrs"]["list_args"]) {
        if (list_arg.is_object() && list_arg.find("name") != list_arg.end()) {
          list_arg.erase("name");
        }
      }
    }
  }
  return std::hash<std::string>()(fusion_json_copy.dump());
}

std::map<int64_t, KernelModPtr> KernelFusion(const std::vector<FusionScopeInfo> &fusion_scopes) {
  std::map<int64_t, KernelModPtr> kernel_mod_ret;
  static std::set<std::string> processed_fusion_kernel;
  auto build_manger = std::make_shared<ParallelBuildManager>();
  MS_EXCEPTION_IF_NULL(build_manger);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto device_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto tune_mode = context_ptr->get_param<std::string>(MS_CTX_TUNE_MODE);
  std::string offline_tune = common::GetEnv("ENABLE_TUNE_DUMP");
  if (!offline_tune.empty()) {
    for (size_t j = 0; j < offline_tune.length(); j++) {
      offline_tune[j] = tolower(offline_tune[j]);
    }
    if (!(offline_tune == "true" || offline_tune == "false")) {
      MS_LOG(EXCEPTION) << "The value of ENABLE_TUNE_DUMP must be 'true' or 'false'";
    }
  }

  for (const auto &fusion_scope_iter : fusion_scopes) {
    string fusion_kernel_name;
    nlohmann::json fusion_op;
    if (!TbeKernelBuild::GenFusionScopeJson(fusion_scope_iter.input_nodes, fusion_scope_iter.compute_nodes, &fusion_op,
                                            &fusion_kernel_name)) {
      continue;
    }
    // gen kernel_name & check cache
    size_t hash_id = GenFusionJsonHash(fusion_op);
    auto json_name =
      fusion_kernel_name.append("_").append(std::to_string(hash_id)).append("_").append(std::to_string(device_id));
    fusion_op["fusion_op_name"] = json_name;
    fusion_op["full_name"] = fusion_scope_iter.full_name;
    // get io size
    std::vector<size_t> input_size_list;
    std::vector<size_t> output_size_list;
    if (!TbeKernelBuild::GetIOSize(fusion_op["op_list"], fusion_scope_iter.output_nodes, &input_size_list,
                                   &output_size_list)) {
      continue;
    }
    // search cache
    auto kernel_pack = TbeUtils::SearchCache(json_name, tbe::kProcessorAiCore);
    if (kernel_pack != nullptr && ((!offline_tune.empty() && offline_tune != "true") || tune_mode == "NO_TUNE")) {
      auto kernel_mod =
        build_manger->GenKernelMod(json_name, tbe::kProcessorAiCore, input_size_list, output_size_list, kernel_pack);
      if (kernel_mod != nullptr) {
        kernel_mod_ret[fusion_scope_iter.scope_id] = kernel_mod;
        continue;
      }
    }
    // same op not need build, but need wait build finish to set kernel mode
    if (processed_fusion_kernel.find(json_name) != processed_fusion_kernel.end()) {
      build_manger->SaveSameFusionOpInfo(fusion_scope_iter.scope_id, json_name, tbe::kProcessorAiCore, input_size_list,
                                         output_size_list);
      continue;
    }
    (void)processed_fusion_kernel.insert(json_name);
    // generate soc info json
    nlohmann::json soc_info_json;
    TbeUtils::GenSocInfo(&soc_info_json);
    soc_info_json["autoTilingMode"] = tune_mode;
    auto soc_version = TbeKernelJsonCreator::GetSocVersion();
    soc_info_json["socVersion"] = soc_version;
    // fusion build
    nlohmann::json fusion_json;
    fusion_json["fusion_op"] = fusion_op;
    fusion_json["SocInfo"] = soc_info_json;
    auto task_id = build_manger->StartCompileOp(fusion_json);
    TbeUtils::SaveJsonInfo(json_name, fusion_json.dump());
    if (task_id < 0) {
      MS_EXCEPTION(ArgumentError) << "start compile failed.";
    }
    build_manger->SaveTaskInfo(task_id, nullptr, json_name, input_size_list, output_size_list,
                               fusion_scope_iter.scope_id);
  }

  int build_failed_num = 0;
  while (!build_manger->IsAllTaskFinish()) {
    int task_id = -1;
    std::string task_result;
    std::string build_result;
    auto ret = build_manger->WaitOne(&task_id, &task_result, &build_result);
    if (!ret) {
      MS_EXCEPTION(ArgumentError) << "Build Failed. wait one ret:" << ret << ", task id:" << task_id;
    }

    if (task_result != "Success") {
      MS_LOG(INFO) << "Fusion warning: Fuison op build failed, err log: " << task_result
                   << "  change to single op build.";
      build_failed_num++;
    }
    auto kernel_mod_item = build_manger->TaskFinishProcess(task_id, build_result, false);
    if (kernel_mod_item.second != nullptr) {
      (void)kernel_mod_ret.emplace(kernel_mod_item);
    }
  }
  bool ret = build_manger->GenSameFusionOpKernelMod(&kernel_mod_ret);
  if (!ret) {
    MS_LOG(INFO) << "Fusion warning: Fuison op has cache failed.";
  }

  MS_LOG(INFO) << "Build Fusion Kernel Failed Num: " << build_failed_num;
  return kernel_mod_ret;
}
}  // namespace mindspore::kernel
