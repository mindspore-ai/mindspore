/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "tools/graph_kernel/converter/akg/utils.h"

#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <algorithm>

#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "thread/threadpool.h"
#include "tools/common/tensor_util.h"
#include "utils/anf_utils.h"
#include "utils/file_utils.h"
#include "utils/log_adapter.h"
#include "utils/system/env.h"
#include "mindspore/ccsrc/include/common/debug/common.h"

namespace mindspore::graphkernel {
bool SaveJsonInfo(const std::string &json_name, const std::string &info) {
  std::string path = json_name + ".info";
  std::ofstream filewrite(path);
  if (!filewrite.is_open()) {
    MS_LOG(ERROR) << "Open file '" << path << "' failed!";
    return false;
  }
  filewrite << info << std::endl;
  filewrite.close();
  return true;
}

std::string SaveNodesInfo(const AnfNodePtrList &nodes, const std::string &dir, const DumpOption &option,
                          std::map<AnfNodePtr, std::string> *node_kernel, std::set<std::string> *kernel_names) {
  auto dir_path = FileUtils::CreateNotExistDirs(dir);
  if (!dir_path.has_value()) {
    MS_LOG(ERROR) << "Failed to CreateNotExistDirs: " << dir;
    return "";
  }
  std::set<std::string> unique_kernel_name;
  for (const auto &node : nodes) {
    graphkernel::GraphKernelJsonGenerator graph_kernel_json_generator(option);
    auto fg = GetCNodeFuncGraph(node);
    MS_EXCEPTION_IF_NULL(fg);
    auto mng = fg->manager();
    if (mng == nullptr) {
      mng = Manage(fg, true);
      fg->set_manager(mng);
    }
    std::vector<AnfNodePtr> node_list, input_list, output_list;
    GkUtils::GetValidKernelNodes(fg, &node_list, &input_list, &output_list);
    graph_kernel_json_generator.CollectFusedJson(node_list, input_list, output_list);
    auto json_kernel_name = graph_kernel_json_generator.kernel_name();
    if (node_kernel != nullptr) {
      (*node_kernel)[node] = json_kernel_name;
    }
    if (!unique_kernel_name.insert(json_kernel_name).second) {
      continue;
    }
    if (!SaveJsonInfo(dir_path.value() + "/" + json_kernel_name, graph_kernel_json_generator.kernel_json_str())) {
      return "";
    }
  }
  if (kernel_names != nullptr) {
    *kernel_names = std::move(unique_kernel_name);
  }
  return dir_path.value();
}

std::string GetCNodeDynamicInputIndex(const CNodePtr &cnode) {
  std::string dynamic_input_index;
  auto cb = Callback::Instance();
  for (size_t i = 1; i < cnode->inputs().size(); i++) {
    if (cnode->input(i)->isa<CNode>() || cnode->input(i)->isa<Parameter>()) {
      auto input_shape = cb->GetInputShape(cnode, i - 1);
      if (input_shape.size() <= 0 || input_shape[0] != 1) {
        MS_LOG(EXCEPTION) << "Dynamic inputs' batch size should be 1";
      }
      dynamic_input_index += std::to_string(i - 1) + ",";
    }
  }
  return dynamic_input_index;
}

std::string GetCNodeInputShapeStr(const CNodePtr &cnode) {
  std::string input_shape_str;
  auto cb = Callback::Instance();
  for (size_t i = 1; i < cnode->inputs().size(); i++) {
    auto input_shape = cb->GetInputShape(cnode, i - 1);
    input_shape_str += std::to_string(input_shape.size()) + ",";
    for (auto &v : input_shape) {
      input_shape_str += std::to_string(v) + ",";
    }
  }
  return input_shape_str;
}

std::string GetCNodeOutputShapeStr(const CNodePtr &cnode) {
  std::string output_shape_str;
  auto output_num = AnfUtils::GetOutputTensorNum(cnode);
  auto cb = Callback::Instance();
  for (size_t i = 0; i < output_num; i++) {
    auto output_shape = cb->GetOutputShape(cnode, i);
    output_shape_str += std::to_string(output_shape.size()) + ",";
    for (auto &v : output_shape) {
      output_shape_str += std::to_string(v) + ",";
    }
  }
  return output_shape_str;
}

std::string GetCNodeOutputTypeStr(const CNodePtr &cnode) {
  std::string output_type_str;
  auto output_num = AnfUtils::GetOutputTensorNum(cnode);
  auto cb = Callback::Instance();
  for (size_t i = 0; i < output_num; i++) {
    auto output_type = cb->GetOutputType(cnode, i);
    output_type_str += std::to_string(static_cast<int>(output_type)) + ",";
  }
  return output_type_str;
}

std::string GetCNodeOutputFormatStr(const CNodePtr &cnode) {
  std::string output_format_str;
  auto output_num = AnfUtils::GetOutputTensorNum(cnode);
  auto cb = Callback::Instance();
  for (size_t i = 0; i < output_num; i++) {
    auto output_format = cb->GetOutputFormat(cnode, i);
    if (output_format == kOpFormat_NHWC) {
      output_format_str += "1,";
    } else {  // default, NCHW
      output_format_str += "0,";
    }
  }
  return output_format_str;
}

ParameterPtr CreateAkgKernelParameter(const FuncGraphPtr &func_graph, const std::string &path,
                                      const std::string &kernel_name) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  auto param_node = func_graph->add_parameter();
  MS_CHECK_TRUE_RET(param_node != nullptr, nullptr);
  param_node->set_name(kernel_name);
  if (path.empty()) {
    return nullptr;
  }
  if (!Common::FileExists(path)) {
    return nullptr;
  }
  auto akg_fd = open(path.c_str(), O_RDONLY);
  struct stat sb;
  if (akg_fd < 0) {
    MS_LOG(ERROR) << "open " << path << " failed.";
    return nullptr;
  }
  if (fstat(akg_fd, &sb) == -1) {
    MS_LOG(ERROR) << "fstat " << path << " failed.";
    return nullptr;
  }
  auto akg_mmap = mmap(NULL, sb.st_size, PROT_READ, MAP_SHARED, akg_fd, 0);
  if (akg_mmap == nullptr) {
    MS_LOG(ERROR) << "mmap " << path << " failed.";
    return nullptr;
  }
  (void)close(akg_fd);
  auto tensor_info = lite::CreateTensorInfo(akg_mmap, sb.st_size, {sb.st_size}, kNumberTypeUInt8);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "Create tensor info failed";
    return nullptr;
  }
  (void)munmap(akg_mmap, sb.st_size);
  auto status = lite::InitParameterFromTensorInfo(param_node, tensor_info);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed";
    return nullptr;
  }
  return param_node;
}

bool CompileSingleJson(const std::string &json_name) {
  std::string attrs = "None";
  std::ostringstream py_cmd;
  py_cmd << kAddMSLiteAkg;
  py_cmd << "from akg.ms import compilewithjsonname\n";
  py_cmd << "if not compilewithjsonname(\'" << json_name << "\', " << attrs << "):\n";
  py_cmd << "    raise RuntimeError(\'Compile fail for json: " << json_name << "\')";
  std::string cmd = "python -c \"" + py_cmd.str() + "\"";
  auto ret = std::system(cmd.c_str());
  if (!WIFEXITED(ret)) {
    MS_LOG(ERROR) << "Python process start fail! process content is as follows:\n" << cmd;
    return false;
  }
  if (WEXITSTATUS(ret) != 0) {
    MS_LOG(ERROR) << "Failed to compile json: " << json_name;
    return false;
  }
  return true;
}

bool RetStatus(const int status) {
  if (WIFEXITED(status)) {
    if (WEXITSTATUS(status) == 0) {
      MS_LOG(INFO) << "compile all pass for subprocess!";
      return true;
    } else {
      MS_LOG(ERROR) << "Some jsons compile fail, please check log!";
    }
  } else if (WIFSIGNALED(status)) {
    MS_LOG(ERROR) << "compile stopped by signal, maybe cost too long time!";
  } else if (WSTOPSIG(status)) {
    MS_LOG(ERROR) << "compile process is stopped by others!";
  } else {
    MS_LOG(ERROR) << "unknown error in compiling!";
  }
  return false;
}

bool CompileJsonsInList(const std::string &dir_path, const std::vector<std::string> &json_list) {
  auto json_list_size = static_cast<int>(json_list.size());
  auto thread_num = std::min(PROCESS_LIMIT, json_list_size);
  if (thread_num == 0) {
    return true;
  }
  auto func = [&](void *cdata, int task_id, float lhs_scale, float rhs_scale) -> int {
    bool all_pass{true};
    for (int j = task_id; j < json_list_size; j += PROCESS_LIMIT) {
      auto res = CompileSingleJson(dir_path + "/" + json_list[j] + ".info");
      if (!res) {
        all_pass = false;
      }
    }
    if (!all_pass) {
      MS_LOG(ERROR) << "Some task failed.";
      return lite::RET_ERROR;
    }
    return lite::RET_OK;
  };
  auto *pool = ThreadPool::CreateThreadPool(thread_num);
  if (pool && pool->ParallelLaunch(func, nullptr, thread_num) == lite::RET_OK) {
    return true;
  }
  return false;
}

void ExcludeTunedObj(const std::string &dir_path, std::set<std::string> *kernel_names,
                     std::map<AnfNodePtr, std::string> *node_kernel) {
  auto fs = system::Env::GetFileSystem();
  std::map<std::string, std::string> tuned_obj_map;  // < tuned_signature, best split object name >
  for (auto &iter : *node_kernel) {
    auto fg = GetCNodeFuncGraph(iter.first);
    MS_EXCEPTION_IF_NULL(fg);
    auto tuned_sign = fg->has_attr(kTunedSign) ? GetValue<std::string>(fg->get_attr(kTunedSign)) : "";
    if (tuned_sign == iter.second) {
      // the kernel name is the same as signature, find cache.
      auto cache = tuned_obj_map.find(tuned_sign);
      if (cache != tuned_obj_map.end()) {
        iter.second = cache->second;
      }
      if (!fg->has_attr(kAttrNodeName)) {
        continue;
      }
      auto best_split_kernel = std::string("best_split_") + GetValue<std::string>(fg->get_attr(kAttrNodeName));
      auto best_split_file = dir_path + "/" + best_split_kernel + ".o";
      if (!fs->FileExist(best_split_file)) {
        continue;
      }
      // the cache file exists, use it.
      tuned_obj_map[tuned_sign] = best_split_kernel;
      iter.second = best_split_kernel;
      (void)kernel_names->erase(tuned_sign);
      MS_LOG(INFO) << "Reuse the object file " << best_split_file;
    } else {
      if (!tuned_sign.empty()) {
        MS_LOG(INFO) << "The kernel_name of " << iter.first->fullname_with_scope() << " mismatch its signature. "
                     << "kernel_name is " << iter.second << ", and tuned_signature is " << tuned_sign;
      }
    }
  }
}
}  // namespace mindspore::graphkernel
