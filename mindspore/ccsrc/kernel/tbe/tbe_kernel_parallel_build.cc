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

#include "kernel/tbe/tbe_kernel_parallel_build.h"

#include <memory>
#include <map>
#include <set>
#include <algorithm>
#include <vector>
#include <string>

#include "utils/context/ms_context.h"
#include "kernel/tbe/tbe_adapter.h"
#include "kernel/tbe/tbe_kernel_build.h"
#include "kernel/tbe/tbe_kernel_mod.h"
#include "session/anf_runtime_algorithm.h"
#include "./common.h"
#include "kernel/tbe/tbe_python_funcs.h"
#include "kernel/tbe/tbe_convert_utils.h"
#include "kernel/tbe/tbe_utils.h"

namespace mindspore {
namespace kernel {
using mindspore::kernel::tbe::TbeUtils;
constexpr auto kParallelCompileModule = "mindspore._extends.parallel_compile.tbe_compiler.tbe_process";
constexpr auto kCreateParallelCompiler = "create_tbe_parallel_compiler";
constexpr auto kStartCompileOp = "start_compile_op";
constexpr auto kWaitOne = "wait_one";
constexpr auto kResetTaskInfo = "reset_task_info";

bool TbeOpParallelPreBuild(const std::vector<AnfNodePtr> &anf_nodes) {
  auto build_manger = std::make_shared<ParallelBuildManager>();
  MS_EXCEPTION_IF_NULL(build_manger);
  for (const auto &anf_node : anf_nodes) {
    // gen kernel json
    MS_EXCEPTION_IF_NULL(anf_node);
    nlohmann::json kernel_json;
    TbeKernelJsonCreator creator(OP_PRE_COMPILE);
    if (!creator.GenTbeSingleKernelJson(anf_node, &kernel_json)) {
      MS_LOG(ERROR) << "GenTbeSingleKernelJson failed";
      return false;
    }
    kernel_json["compile_type"] = "pre_build";
    // op build
    auto task_id = build_manger->StartCompileOp(kernel_json);
    build_manger->SavePreTaskInfo(task_id, anf_node);
  }
  while (!build_manger->IsAllPreTaskFinish()) {
    int task_id = -1;
    char *task_result = nullptr;
    char *pre_build_result = nullptr;
    auto ret = build_manger->WaitOne(&task_id, &task_result, &pre_build_result);
    if (!ret) {
      MS_EXCEPTION(ArgumentError) << "Pre Build Failed. wait one ret:" << ret << ", task id:" << task_id;
    }

    if ((task_result != nullptr) && (strcmp(task_result, "Success") != 0)) {
      MS_EXCEPTION(ArgumentError) << "task pre compile Failed, task id:" << task_id << ", cause:" << task_result;
    }

    build_manger->PreTaskFinishProcess(task_id, pre_build_result);
  }
  return true;
}

bool TbeOpParallelBuild(std::vector<AnfNodePtr> anf_nodes) {
  auto build_manger = std::make_shared<ParallelBuildManager>();
  MS_EXCEPTION_IF_NULL(build_manger);
  set<std::string> processed_kernel;
  for (const auto &anf_node : anf_nodes) {
    // gen kernel json
    tbe::TbeAdapter::SetTbeAttrsForTransDataOp(anf_node);
    if (AnfAlgo::GetKernelMod(anf_node) != nullptr) {
      continue;
    }
    const std::string &processor = tbe::GetProcessor(anf_node);
    nlohmann::json kernel_json;
    TbeKernelJsonCreator creator(SINGLE_BUILD);
    if (!creator.GenTbeSingleKernelJson(anf_node, &kernel_json)) {
      MS_LOG(ERROR) << "GenTbeSingleKernelJson failed";
      return false;
    }
    // get size
    std::vector<size_t> input_size_list;
    std::vector<size_t> output_size_list;
    (void)TbeKernelBuild::GetIOSize(kernel_json, &input_size_list, &output_size_list);
    // search cache
    const std::string &json_name = creator.json_name();
    if (build_manger->SearchInCache(json_name, processor, input_size_list, output_size_list, anf_node.get())) {
      MS_LOG(INFO) << "Use cached kernel, kernel json name:." << json_name;
      continue;
    }
    // same op not need build, but need wait build finish to set kernel mode
    if (processed_kernel.find(json_name) != processed_kernel.end()) {
      build_manger->SaveSameOpInfo(anf_node, json_name, input_size_list, output_size_list);
      continue;
    }
    (void)processed_kernel.insert(json_name);
    // op build
    auto task_id = build_manger->StartCompileOp(kernel_json);
    build_manger->SaveTaskInfo(task_id, anf_node, json_name, input_size_list, output_size_list);
  }
  while (!build_manger->IsAllTaskFinish()) {
    int task_id = -1;
    char *task_result = nullptr;
    char *pre_build_result = nullptr;
    auto ret = build_manger->WaitOne(&task_id, &task_result, &pre_build_result);
    if (!ret) {
      MS_EXCEPTION(ArgumentError) << "Build Failed. wait one ret:" << ret << ", task id:" << task_id;
    }

    if ((task_result != nullptr) && (strcmp(task_result, "Success") != 0)) {
      MS_EXCEPTION(ArgumentError) << "task compile Failed, task id:" << task_id << ", cause:" << task_result;
    }
    (void)build_manger->TaskFinishProcess(task_id);
  }
  return build_manger->GenSameOpKernelMod();
}

ParallelBuildManager::ParallelBuildManager() { tbe_parallel_compiler_ = TbePythonFuncs::TbeParallelCompiler(); }

ParallelBuildManager::~ParallelBuildManager() { ResetTaskInfo(); }

int32_t ParallelBuildManager::StartCompileOp(const nlohmann::json &kernel_json) const {
  PyObject *pRes = nullptr;
  PyObject *pArgs = PyTuple_New(1);
  std::string json_str = kernel_json.dump();
  PyObject *arg1 = Py_BuildValue("s", json_str.c_str());
  (void)PyTuple_SetItem(pArgs, 0, arg1);
  pRes = PyObject_CallMethod(tbe_parallel_compiler_, kStartCompileOp, "O", pArgs);
  if (pRes == nullptr) {
    PyErr_Print();
    MS_EXCEPTION(ArgumentError) << "Failed to call function start_compile_op";
  }
  int task_id;
  (void)PyArg_Parse(pRes, "i", &task_id);
  MS_LOG(INFO) << "start compile , task id:" << task_id;
  return task_id;
}

bool ParallelBuildManager::WaitOne(int *task_id, char **task_result, char **pre_build_result) const {
  MS_LOG(INFO) << "wait task start.";
  MS_EXCEPTION_IF_NULL(task_id);
  MS_EXCEPTION_IF_NULL(task_result);
  PyObject *pRes = nullptr;
  PyObject *pArg = Py_BuildValue("()");
  pRes = PyObject_CallMethod(tbe_parallel_compiler_, kWaitOne, "O", pArg);
  if (pRes == nullptr) {
    PyErr_Print();
    MS_EXCEPTION(ArgumentError) << "Failed to call function wait_one";
    return false;
  }
  (void)PyArg_ParseTuple(pRes, "iss", task_id, task_result, pre_build_result);
  return true;
}

void ParallelBuildManager::SavePreTaskInfo(int32_t task_id, const mindspore::AnfNodePtr &anf_node) {
  MS_LOG(INFO) << "SavePreTaskInfo, task id: " << task_id;
  pre_task_map_[task_id] = anf_node;
}

void ParallelBuildManager::SaveTaskInfo(int32_t task_id, const mindspore::AnfNodePtr &anf_node,
                                        const std::string &json_name, const std::vector<size_t> &input_size_list,
                                        const std::vector<size_t> &output_size_list, int32_t scope_id) {
  MS_LOG(INFO) << "SaveTaskInfo, task id: " << task_id;
  struct KernelBuildTaskInfo task_info;
  task_info.node = anf_node.get();
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

bool ParallelBuildManager::IsAllPreTaskFinish() const {
  MS_LOG(INFO) << "wait pre build process task_num: " << pre_task_map_.size();
  return pre_task_map_.empty();
}

bool ParallelBuildManager::IsAllTaskFinish() const {
  MS_LOG(INFO) << "wait process task_num: " << task_map_.size();
  return task_map_.empty();
}

void ParallelBuildManager::PreTaskFinishProcess(int32_t task_id, const std::string &pre_build_result) {
  auto task_iter = pre_task_map_.find(task_id);
  if (task_iter == pre_task_map_.end()) {
    MS_EXCEPTION(ArgumentError) << "can find pre task_id:" << task_id;
  }
  auto node = task_iter->second;
  auto builder =
    std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(AnfAlgo::GetSelectKernelBuildInfo(node));
  std::string start_flag = "fusion_pattern_start";
  std::string end_flag = "fusion_pattern_end";
  int start = pre_build_result.find(start_flag);
  int end = pre_build_result.find(end_flag);
  if (start != -1 && end != -1) {
    std::string result = pre_build_result.substr(start + start_flag.size(), end - start - start_flag.size());
    transform(result.begin(), result.end(), result.begin(), ::toupper);
    FusionType fusion_type = tbe::GetFusionType(result);
    builder->SetFusionType(fusion_type);
    AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), node.get());
  }
  (void)pre_task_map_.erase(task_iter);
}

std::pair<int32_t, KernelModPtr> ParallelBuildManager::TaskFinishProcess(int32_t task_id, bool set_kernel_mod) {
  auto task_iter = task_map_.find(task_id);
  if (task_iter == task_map_.end()) {
    MS_EXCEPTION(ArgumentError) << "can find task_id:" << task_id;
  }
  auto json_name = task_iter->second.json_name;
  auto processor = task_iter->second.processor;
  auto kernel_pack = TbeUtils::InsertCache(json_name, processor);
  if (kernel_pack == nullptr) {
    if (set_kernel_mod) {
      MS_EXCEPTION(ArgumentError) << "build kernel name:" << task_iter->second.json_name << " failed.";
    } else {
      MS_LOG(INFO) << "fusion build kernel name:" << task_iter->second.json_name << "failed.";
      auto ret = std::make_pair(task_iter->second.scope_id, nullptr);
      (void)task_map_.erase(task_iter);
      return ret;
    }
  }
  auto kernel_mod = GenKernelMod(json_name, processor, task_iter->second.input_size_list,
                                 task_iter->second.output_size_list, kernel_pack);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  if (set_kernel_mod) {
    AnfAlgo::SetKernelMod(kernel_mod, task_iter->second.node);
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
  task_info.node = anf_node.get();
  task_info.json_name = json_name;
  task_info.processor = tbe::GetProcessor(anf_node);
  task_info.input_size_list.assign(input_size_list.begin(), input_size_list.end());
  task_info.output_size_list.assign(output_size_list.begin(), output_size_list.end());
  same_op_list_.push_back(task_info);
}

bool ParallelBuildManager::GenSameOpKernelMod() const {
  for (const auto &task_info : same_op_list_) {
    bool ret = SearchInCache(task_info.json_name, task_info.processor, task_info.input_size_list,
                             task_info.output_size_list, task_info.node);
    if (!ret) {
      MS_LOG(INFO) << "can't find " << task_info.json_name << " in cache.";
      return false;
    }
  }
  return true;
}

bool ParallelBuildManager::SearchInCache(const std::string &json_name, const std::string &processor,
                                         const std::vector<size_t> &input_size_list,
                                         const std::vector<size_t> &output_size_list, mindspore::AnfNode *node) const {
  auto cached_kernel_pack = TbeUtils::SearchCache(json_name, processor);
  if (cached_kernel_pack != nullptr) {
    MS_LOG(INFO) << "Find cached kernel, kernel json name" << json_name;
    auto kernel_mod_ptr = GenKernelMod(json_name, processor, input_size_list, output_size_list, cached_kernel_pack);
    MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
    AnfAlgo::SetKernelMod(kernel_mod_ptr, node);
    return true;
  } else {
    return false;
  }
}

KernelModPtr ParallelBuildManager::GenKernelMod(const string &json_name, const string &processor,
                                                const vector<size_t> &input_size_list,
                                                const vector<size_t> &output_size_list,
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

void ParallelBuildManager::ResetTaskInfo() {
  if (task_map_.empty()) {
    MS_LOG(INFO) << "All tasks are compiled success.";
    return;
  }
  task_map_.clear();
  same_op_list_.clear();
  if (tbe_parallel_compiler_ != nullptr) {
    PyObject *pArg = Py_BuildValue("()");
    (void)PyObject_CallMethod(tbe_parallel_compiler_, kResetTaskInfo, "O", pArg);
  }
}
}  // namespace kernel
}  // namespace mindspore
