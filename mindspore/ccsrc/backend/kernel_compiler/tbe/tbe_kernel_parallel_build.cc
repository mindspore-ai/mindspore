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

#include "backend/kernel_compiler/tbe/tbe_kernel_parallel_build.h"
#include <memory>
#include <set>
#include <algorithm>
#include <vector>
#include <string>
#include "utils/ms_context.h"
#include "backend/kernel_compiler/common_utils.h"
#include "backend/kernel_compiler/tbe/tbe_adapter.h"
#include "backend/kernel_compiler/tbe/tbe_kernel_build.h"
#include "backend/kernel_compiler/tbe/tbe_kernel_mod.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/kernel_compiler/tbe/tbe_convert_utils.h"
#include "backend/kernel_compiler/tbe/tbe_utils.h"
#include "backend/kernel_compiler/tbe/tbe_dynaminc_shape_util.h"
#include "utils/trace_base.h"
#include "utils/json_operation_utils.h"

namespace mindspore {
namespace kernel {
using mindspore::kernel::tbe::TbeUtils;
bool TbeOpParallelBuild(const std::vector<AnfNodePtr> &anf_nodes) {
  auto build_manger = std::make_shared<ParallelBuildManager>();
  MS_EXCEPTION_IF_NULL(build_manger);
  static std::set<std::string> processed_kernel = {};
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto tune_mode = context_ptr->get_param<std::string>(MS_CTX_TUNE_MODE);
  std::string offline_tune = common::GetEnv("ENABLE_TUNE_DUMP");
  if (!offline_tune.empty()) {
    for (size_t j = 0; j < offline_tune.length(); j++) {
      offline_tune[j] = tolower(offline_tune[j]);
    }
    if (!(offline_tune == "true" || offline_tune == "false")) {
      MS_LOG(ERROR) << "Invalid environment variable 'ENABLE_TUNE_DUMP', it should be 'true' or 'false', but got "
                    << tune_mode;
      return false;
    }
  }

  for (const auto &anf_node : anf_nodes) {
    // gen kernel json
    if (AnfAlgo::GetKernelMod(anf_node) != nullptr) {
      continue;
    }
    nlohmann::json kernel_json;
    TbeKernelJsonCreator creator(SINGLE_BUILD);
    if (!creator.GenTbeSingleKernelJson(anf_node, &kernel_json)) {
      MS_LOG(ERROR) << "GenTbeSingleKernelJson failed";
      TbeUtils::SaveJsonInfo(kernel_json["op_info"]["kernel_name"], kernel_json.dump());
      return false;
    }
    // get size
    std::vector<size_t> input_size_list;
    std::vector<size_t> output_size_list;
    (void)TbeKernelBuild::GetIOSize(kernel_json, &input_size_list, &output_size_list);
    // search cache
    const std::string &json_name = creator.json_name();
    if (build_manger->SearchInCache(json_name, input_size_list, output_size_list, anf_node.get()) &&
        ((!offline_tune.empty() && offline_tune != "true") || tune_mode == "NO_TUNE")) {
      continue;
    }
    // same op not need build, but need wait build finish to set kernel mode
    if (processed_kernel.find(json_name) != processed_kernel.end()) {
      build_manger->SaveSameOpInfo(anf_node, json_name, input_size_list, output_size_list);
      continue;
    }
    (void)processed_kernel.insert(json_name);
    // op build
    TbeUtils::SaveJsonInfo(kernel_json["op_info"]["kernel_name"], kernel_json.dump());
    auto task_id = ParallelBuildManager::StartCompileOp(kernel_json);
    build_manger->SaveTaskInfo(task_id, anf_node, json_name, input_size_list, output_size_list);
  }
  while (!build_manger->IsAllTaskFinish()) {
    int task_id = -1;
    std::string task_result;
    std::string build_result;
    auto ret = ParallelBuildManager::WaitOne(&task_id, &task_result, &build_result);
    if (!ret) {
      MS_EXCEPTION(ArgumentError) << "Build Failed. wait one ret:" << ret << ", task id:" << task_id
                                  << " trace: " << trace::DumpSourceLines(build_manger->GetAnfNodeByTaskID(task_id));
    }

    if (task_result != "Success") {
      MS_EXCEPTION(ArgumentError) << "task compile Failed, task id:" << task_id << ", cause:" << task_result
                                  << " trace: " << trace::DumpSourceLines(build_manger->GetAnfNodeByTaskID(task_id));
    }
    (void)build_manger->TaskFinishProcess(task_id, build_result);
  }
  return build_manger->GenSameOpKernelMod();
}

ParallelBuildManager::~ParallelBuildManager() { ResetTaskInfo(); }

void ParallelBuildManager::SavePreBuildTaskInfo(int32_t task_id, const AnfNodePtr &anf_node,
                                                const std::string &json_name) {
  MS_LOG(DEBUG) << "SavePreBuildTaskInfo, task id: " << task_id;
  struct KernelBuildTaskInfo task_info;
  task_info.node = anf_node;
  task_info.json_name = json_name;
  if (anf_node == nullptr) {
    task_info.processor = tbe::kProcessorAiCore;
  } else {
    task_info.processor = tbe::GetProcessor(anf_node);
  }
  task_info.scope_id = 0;
  pre_build_task_map_[task_id] = task_info;
}

void ParallelBuildManager::SaveTaskInfo(int32_t task_id, const mindspore::AnfNodePtr &anf_node,
                                        const std::string &json_name, const std::vector<size_t> &input_size_list,
                                        const std::vector<size_t> &output_size_list, int64_t scope_id) {
  MS_LOG(DEBUG) << "SaveTaskInfo, task id: " << task_id;
  struct KernelBuildTaskInfo task_info;
  task_info.node = anf_node;
  task_info.json_name = json_name;
  if (anf_node == nullptr) {
    task_info.processor = tbe::kProcessorAiCore;
  } else {
    task_info.processor = tbe::GetProcessor(anf_node);
  }
  task_info.input_size_list.assign(input_size_list.begin(), input_size_list.end());
  task_info.output_size_list.assign(output_size_list.begin(), output_size_list.end());
  task_info.scope_id = scope_id;
  task_map_[task_id] = task_info;
}

bool ParallelBuildManager::IsAllTaskFinish() const {
  MS_LOG(INFO) << "wait process task_num: " << task_map_.size();
  return task_map_.empty();
}

void ParallelBuildManager::PreTaskFinishProcess(int32_t task_id, const std::string &pre_build_result) {
  MS_LOG(DEBUG) << "can find pre task_id : " << task_id << " result:" << pre_build_result;
  auto task_iter = pre_build_task_map_.find(task_id);
  if (task_iter == pre_build_task_map_.end()) {
    MS_EXCEPTION(ArgumentError) << "can find pre task_id:" << task_id;
  }
  nlohmann::json result;
  if (!ParseJson(pre_build_result, &result)) {
    MS_LOG(EXCEPTION) << "Parse prebuild result error.";
  }
  auto fusion_name = GetJsonValue<std::string>(result, "op_pattern");
  auto fusion_type = kernel::GetFusionTypeByName(fusion_name);
  auto output_data_desc = GetJsonValue<nlohmann::json>(result, "op_params");

  auto node = task_iter->second.node;
  AnfAlgo::SetFusionType(node, fusion_type);
  AnfAlgo::SetOutputDataDesc(node, {output_data_desc});
  (void)pre_build_task_map_.erase(task_iter);
}

std::pair<int32_t, KernelModPtr> ParallelBuildManager::TaskFinishProcess(int32_t task_id, const std::string &build_ret,
                                                                         bool set_kernel_mod) {
  auto task_iter = task_map_.find(task_id);
  if (task_iter == task_map_.end()) {
    MS_EXCEPTION(ArgumentError) << "can find task_id:" << task_id;
  }
  auto json_name = task_iter->second.json_name;
  auto processor = task_iter->second.processor;
  auto kernel_pack = TbeUtils::InsertCache(json_name, processor);
  if (kernel_pack == nullptr) {
    if (set_kernel_mod) {
      MS_EXCEPTION(ArgumentError) << "Can not find .json file or the binary .o file for op "
                                  << task_iter->second.json_name << ", go check the cache files in kernel_meta/";
    } else {
      MS_LOG(INFO) << "fusion build kernel name:" << task_iter->second.json_name << "failed.";
      auto fusion_kernel_mod = std::make_pair(task_iter->second.scope_id, nullptr);
      (void)task_map_.erase(task_iter);
      return fusion_kernel_mod;
    }
  }
  auto kernel_mod = GenKernelMod(task_iter->second.input_size_list, task_iter->second.output_size_list, kernel_pack);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  if (set_kernel_mod) {
    auto cur_node = task_iter->second.node;
    MS_EXCEPTION_IF_NULL(cur_node);
    if (AnfAlgo::IsDynamicShape(cur_node) && (build_ret.empty() || build_ret.find("vars") == std::string::npos)) {
      MS_LOG(EXCEPTION) << "Build failed. The build result of dynamic shape op [" << AnfAlgo::GetCNodeName(cur_node)
                        << "] should not be empty, or can not find key ['vars'] in the result. build_res:[" << build_ret
                        << "].";
    }
    AnfAlgo::SetKernelMod(kernel_mod, cur_node.get());
    MS_LOG(INFO) << json_name << ": save compile info to json file, compile_info:" << build_ret;
    std::string old_build = common::GetEnv("MS_OLD_BUILD_PROCESS");
    if (!old_build.empty()) {
      AnfAlgo::SetNodeAttr(kAttrCompileInfo, MakeValue(build_ret), cur_node);
    } else {
      bool save_flag = true;
      TbeUtils::SaveCompileInfo(json_name, build_ret, &save_flag);
      if (!save_flag) {
        MS_LOG(EXCEPTION) << "Save json file failed, compile_info:" << build_ret;
      }
    }
  }
  auto ret = std::make_pair(task_iter->second.scope_id, kernel_mod);
  (void)task_map_.erase(task_iter);
  MS_LOG(INFO) << "wait process remain task_num:" << task_map_.size();
  return ret;
}

void ParallelBuildManager::SaveSameOpInfo(const mindspore::AnfNodePtr &anf_node, const std::string &json_name,
                                          const std::vector<size_t> &input_size_list,
                                          const std::vector<size_t> &output_size_list) {
  struct KernelBuildTaskInfo task_info;
  task_info.node = anf_node;
  task_info.json_name = json_name;
  task_info.processor = tbe::GetProcessor(anf_node);
  task_info.input_size_list.assign(input_size_list.begin(), input_size_list.end());
  task_info.output_size_list.assign(output_size_list.begin(), output_size_list.end());
  same_op_list_.push_back(task_info);
}

void ParallelBuildManager::SaveSameFusionOpInfo(const int64_t scope_id, const std::string &json_name,
                                                const std::string &processor,
                                                const std::vector<size_t> &input_size_list,
                                                const std::vector<size_t> &output_size_list) {
  struct KernelBuildTaskInfo task_info;
  task_info.scope_id = scope_id;
  task_info.json_name = json_name;
  task_info.processor = processor;
  task_info.input_size_list.assign(input_size_list.begin(), input_size_list.end());
  task_info.output_size_list.assign(output_size_list.begin(), output_size_list.end());
  same_op_list_.push_back(task_info);
}

bool ParallelBuildManager::GenSameOpKernelMod() const {
  for (const auto &task_info : same_op_list_) {
    bool ret =
      SearchInCache(task_info.json_name, task_info.input_size_list, task_info.output_size_list, task_info.node.get());
    if (!ret) {
      MS_LOG(INFO) << "can't find " << task_info.json_name << " in cache.";
      return false;
    }
  }
  return true;
}

bool ParallelBuildManager::GenSameFusionOpKernelMod(std::map<int64_t, KernelModPtr> *kernel_mode_ret) const {
  MS_EXCEPTION_IF_NULL(kernel_mode_ret);
  bool ret = true;
  for (const auto &task_info : same_op_list_) {
    auto kernel_pack = TbeUtils::SearchCache(task_info.json_name);
    if (kernel_pack != nullptr) {
      auto kernel_mode = GenKernelMod(task_info.input_size_list, task_info.output_size_list, kernel_pack);
      if (kernel_mode != nullptr) {
        (*kernel_mode_ret)[task_info.scope_id] = kernel_mode;
        continue;
      }
    }
    MS_LOG(INFO) << "can't find " << task_info.json_name << " in cache.";
    ret = false;
  }
  return ret;
}

bool ParallelBuildManager::SearchInCache(const std::string &json_name, const std::vector<size_t> &input_size_list,
                                         const std::vector<size_t> &output_size_list, mindspore::AnfNode *node) const {
  auto cached_kernel_pack = TbeUtils::SearchCache(json_name);
  if (cached_kernel_pack != nullptr) {
    auto kernel_mod_ptr = GenKernelMod(input_size_list, output_size_list, cached_kernel_pack);
    MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
    AnfAlgo::SetKernelMod(kernel_mod_ptr, node);
    return true;
  } else {
    return false;
  }
}

KernelModPtr ParallelBuildManager::GenKernelMod(const std::vector<size_t> &input_size_list,
                                                const std::vector<size_t> &output_size_list,
                                                const mindspore::kernel::KernelPackPtr &kernel_pack) const {
  MS_EXCEPTION_IF_NULL(kernel_pack);
  auto kernel_json_info = kernel_pack->kernel_json_info();
  auto kernel_mod_ptr = std::make_shared<TbeKernelMod>(kernel_pack);
  MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
  kernel_mod_ptr->SetInputSizeList(input_size_list);
  kernel_mod_ptr->SetOutputSizeList(output_size_list);
  kernel_mod_ptr->SetWorkspaceSizeList(kernel_json_info.workspaces);
  return kernel_mod_ptr;
}

int ParallelBuildManager::StartCompileOp(const nlohmann::json &kernel_json) {
  auto tune_mode = kernel_json["SocInfo"]["autoTilingMode"];
  return AscendKernelBuildClient::Instance().TbeStart(kernel_json.dump(), tune_mode);
}

std::string ParallelBuildManager::ProcessTbeJob(const nlohmann::json &kernel_json) {
  return AscendKernelBuildClient::Instance().TbeSendJob(kernel_json.dump());
}

bool ParallelBuildManager::WaitOne(int *task_id, std::string *task_result, std::string *pre_build_result) {
  MS_EXCEPTION_IF_NULL(task_id);
  return AscendKernelBuildClient::Instance().TbeWait(task_id, task_result, pre_build_result);
}

void ParallelBuildManager::ResetTaskInfo() noexcept {
  task_map_.clear();
  same_op_list_.clear();
  pre_build_task_map_.clear();
}

AnfNodePtr ParallelBuildManager::GetAnfNodeByTaskID(int32_t task_id) {
  auto find_iter = task_map_.find(task_id);
  if (find_iter != task_map_.end()) {
    return find_iter->second.node;
  }
  return nullptr;
}
}  // namespace kernel
}  // namespace mindspore
