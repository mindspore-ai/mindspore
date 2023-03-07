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

#include "tools/graph_kernel/converter/akg/akg_build.h"

#include <sys/wait.h>
#include <dirent.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <algorithm>

#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "utils/anf_utils.h"
#include "utils/file_utils.h"
#include "utils/log_adapter.h"
#include "utils/system/env.h"

namespace mindspore::graphkernel {
bool CompileSingleJson(const std::string &json_name) {
  std::string attrs = "None";
  std::ostringstream py_cmd;
  py_cmd << kAddAkgPath;
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
  auto process_num = std::min(PROCESS_LIMIT, json_list.size());
  if (process_num == 0) {
    return true;
  }
  size_t i;
  pid_t pid;
  std::vector<pid_t> child_process;
  for (i = 0; i < process_num; ++i) {
    pid = fork();
    if (pid < 0) {
      MS_LOG(ERROR) << "fork error";
      return false;
    } else if (pid == 0) {
      break;
    } else {
      child_process.emplace_back(pid);
    }
  }
  if (pid == 0) {
    setpgrp();
    (void)alarm(TIME_OUT);
    bool all_pass{true};
    for (size_t j = i; j < json_list.size(); j += PROCESS_LIMIT) {
      auto res = CompileSingleJson(dir_path + "/" + json_list[j] + ".info");
      if (!res) {
        all_pass = false;
      }
    }
    if (all_pass) {
      exit(0);
    } else {
      exit(1);
    }
  } else {
    bool all_process_pass{true};
    for (size_t j = 0; j < process_num; ++j) {
      int status = 0;
      waitpid(child_process[j], &status, 0);
      // kill child process of child process if overtime
      kill(-child_process[j], SIGTERM);
      all_process_pass = RetStatus(status) && all_process_pass;
    }
    if (all_process_pass) {
      return true;
    }
  }
  return false;
}

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
    graphkernel::AkgKernelJsonGenerator akg_kernel_json_generator(option);
    auto fg = GetCNodeFuncGraph(node);
    MS_EXCEPTION_IF_NULL(fg);
    auto mng = fg->manager();
    if (mng == nullptr) {
      mng = Manage(fg, true);
      fg->set_manager(mng);
    }
    std::vector<AnfNodePtr> node_list, input_list, output_list;
    GkUtils::GetValidKernelNodes(fg, &node_list, &input_list, &output_list);
    akg_kernel_json_generator.CollectFusedJson(node_list, input_list, output_list);
    auto json_kernel_name = akg_kernel_json_generator.kernel_name();
    if (node_kernel != nullptr) {
      (*node_kernel)[node] = json_kernel_name;
    }
    if (!unique_kernel_name.insert(json_kernel_name).second) {
      continue;
    }
    if (!SaveJsonInfo(dir_path.value() + "/" + json_kernel_name, akg_kernel_json_generator.kernel_json_str())) {
      return "";
    }
  }
  if (kernel_names != nullptr) {
    *kernel_names = std::move(unique_kernel_name);
  }
  return dir_path.value();
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

bool AkgKernelBuilder::CompileJsonsInAnfnodes(const AnfNodePtrList &node_list) {
  std::map<AnfNodePtr, std::string> node_info_map;
  std::set<std::string> uniq_info_names;
  auto dir_path =
    SaveNodesInfo(node_list, "./akg_kernel_meta", AkgKernelBuilder::json_option(), &node_info_map, &uniq_info_names);
  if (dir_path.empty()) {
    return false;
  }
  ExcludeTunedObj(dir_path, &uniq_info_names, &node_info_map);
  auto res = CompileJsonsInList(dir_path, std::vector<std::string>(uniq_info_names.begin(), uniq_info_names.end()));
  if (res) {
    std::set<std::string> obj_files;
    std::ostringstream objs;
    for (const auto &iter : node_info_map) {
      AnfUtils::SetNodeAttr("kernel_name", MakeValue(iter.second + "_kernel"), iter.first);
      if (obj_files.insert(iter.second).second) {
        objs << dir_path << "/" << iter.second << ".o ";
      }
    }
    auto cmd = "g++ -fPIC -shared -o akgkernels.so " + objs.str();
    if (std::system(cmd.c_str()) == 0) {
      return true;
    }
  }
  return false;
}
}  // namespace mindspore::graphkernel
