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

#include "common/graph_kernel/lite_adapter/akg_build.h"

#include <sys/wait.h>
#include <dirent.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#include "kernel/akg/akg_kernel_json_generator.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "utils/anf_utils.h"
#include "utils/file_utils.h"
#include "utils/log_adapter.h"

namespace mindspore::graphkernel {
bool CompileSingleJson(const std::string &json_name) {
  std::string attrs = "None";
  std::ostringstream py_cmd;
  py_cmd << "from mindspore._extends.parallel_compile.akg_compiler.get_file_path import get_akg_path\n";
  py_cmd << "import sys\n";
  py_cmd << "sys.path.insert(0, get_akg_path())\n";
  py_cmd << "from akg.ms import compilewithjsonname\n";
  py_cmd << "if not compilewithjsonname(\'" << json_name << "\', " << attrs << "):\n";
  py_cmd << "    raise RuntimeError(\'Compile fail for json: " << json_name << "\')";
  std::string cmd = "unset LD_LIBRARY_PATH;python -c \"" + py_cmd.str() + "\"";
  auto ret = system(cmd.c_str());
  if (!WIFEXITED(ret)) {
    MS_LOG(ERROR) << "python process start fail!";
    return false;
  }
  if (WEXITSTATUS(ret) != 0) {
    MS_LOG(ERROR) << "Error json file is: " << json_name;
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
  size_t i;
  pid_t pid;
  std::vector<pid_t> child_process;
  for (i = 0; i < PROCESS_LIMIT; ++i) {
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
    for (size_t j = 0; j < PROCESS_LIMIT; ++j) {
      int status = 0;
      waitpid(child_process[j], &status, 0);
      // kill child process of child process if overtime
      kill(-child_process[j], SIGTERM);
      return RetStatus(status);
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

void GetValidKernelNodes(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *node_list,
                         std::vector<AnfNodePtr> *input_list, std::vector<AnfNodePtr> *output_list) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node_list);
  MS_EXCEPTION_IF_NULL(input_list);
  std::vector<AnfNodePtr> node_lists = TopoSort(func_graph->get_return());
  for (auto const &node : node_lists) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr || !AnfUtils::IsRealKernel(node)) {
      continue;
    }
    node_list->push_back(node);
  }
  auto parameters = func_graph->parameters();
  input_list->insert(input_list->begin(), parameters.begin(), parameters.end());
  if (IsPrimitiveCNode(func_graph->output(), prim::kPrimMakeTuple)) {
    auto fg_output = func_graph->output()->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(fg_output);
    output_list->assign(fg_output->inputs().begin() + 1, fg_output->inputs().end());
  } else {
    output_list->push_back(func_graph->output());
  }
}

bool AkgKernelBuilder::CompileJsonsInAnfnodes(const AnfNodePtrList &node_list) {
  auto dir_path = FileUtils::CreateNotExistDirs(std::string("./kernel_meta"));
  if (!dir_path.has_value()) {
    MS_LOG(ERROR) << "Failed to CreateNotExistDirs: ./kernel_meta";
    return false;
  }
  std::vector<std::string> json_list;
  std::string kernels_name = "";
  for (const auto &node : node_list) {
    graphkernel::DumpOption option;
    option.get_compute_capability = true;
    graphkernel::AkgKernelJsonGenerator akg_kernel_json_generator(option);
    auto fg = GetCNodeFuncGraph(node);
    MS_EXCEPTION_IF_NULL(fg);
    auto mng = fg->manager();
    if (mng == nullptr) {
      mng = Manage(fg, true);
      fg->set_manager(mng);
    }
    std::vector<AnfNodePtr> node_list, input_list, output_list;
    GetValidKernelNodes(fg, &node_list, &input_list, &output_list);
    akg_kernel_json_generator.CollectFusedJson(node_list, input_list, output_list);
    auto json_kernel_name = akg_kernel_json_generator.kernel_name();
    AnfUtils::SetNodeAttr("kernel_name", MakeValue(json_kernel_name + "_kernel"), node->cast<CNodePtr>());
    if (find(json_list.begin(), json_list.end(), json_kernel_name) != json_list.end()) {
      continue;
    }
    json_list.push_back(json_kernel_name);
    kernels_name += dir_path.value() + "/" + json_kernel_name + ".o ";
    if (!SaveJsonInfo(dir_path.value() + "/" + json_kernel_name, akg_kernel_json_generator.kernel_json_str())) {
      return false;
    }
  }
  auto res = CompileJsonsInList(dir_path.value(), json_list);
  if (res) {
    auto cmd = "g++ -fPIC -shared " + kernels_name + " -o " + dir_path.value() + "/akgkernels.so";
    if (system(cmd.c_str()) == 0) {
      return true;
    }
  }
  return false;
}
}  // namespace mindspore::graphkernel
