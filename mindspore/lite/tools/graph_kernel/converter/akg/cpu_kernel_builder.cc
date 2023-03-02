/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "tools/graph_kernel/converter/akg/cpu_kernel_builder.h"

#include <sys/wait.h>
#include <dirent.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <algorithm>
#include <set>
#include <map>
#include <string>
#include <vector>
#include <memory>

#include "ir/anf.h"
#include "ir/func_graph.h"
#include "utils/anf_utils.h"
#include "utils/file_utils.h"
#include "utils/log_adapter.h"
#include "utils/system/env.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"

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

namespace {
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
}  // namespace

AnfNodePtr CpuKernelBuilder::CreateCustomOp(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  auto op = std::make_shared<ops::Custom>();
  op->set_type("GraphKernel");
  std::map<std::string, std::vector<uint8_t>> custom_attrs;
  auto fg = GetCNodeFuncGraph(cnode);
  MS_EXCEPTION_IF_NULL(fg);
  auto kernel_name = GetValue<std::string>(fg->get_attr("kernel_name"));
  std::vector<uint8_t> kernel_name_str(kernel_name.begin(), kernel_name.end());
  custom_attrs["kernel_name"] = kernel_name_str;
  if (GraphKernelFlags::GetInstance().enable_dynamic_batch && fg->has_attr("dynamic_input_index")) {
    std::string dynamic_input_index = GetValue<std::string>(fg->get_attr("dynamic_input_index"));
    custom_attrs["dynamic_input_index"] = std::vector<uint8_t>(dynamic_input_index.begin(), dynamic_input_index.end());
  }
  std::string input_shape_str = GetCNodeInputShapeStr(cnode);
  std::string output_shape_str = GetCNodeOutputShapeStr(cnode);
  std::string output_format_str = GetCNodeOutputFormatStr(cnode);
  std::string output_type_str = GetCNodeOutputTypeStr(cnode);
  custom_attrs["inputs_shape"] = std::vector<uint8_t>(input_shape_str.begin(), input_shape_str.end());
  custom_attrs["outputs_shape"] = std::vector<uint8_t>(output_shape_str.begin(), output_shape_str.end());
  custom_attrs["outputs_format"] = std::vector<uint8_t>(output_format_str.begin(), output_format_str.end());
  custom_attrs["outputs_type"] = std::vector<uint8_t>(output_type_str.begin(), output_type_str.end());
  op->set_attr(custom_attrs);
  auto inputs = cnode->inputs();
  inputs[0] = NewValueNode(op->GetPrim());
  auto custom_cnode = func_graph->NewCNode(inputs);
  custom_cnode->set_fullname_with_scope(cnode->fullname_with_scope());
  custom_cnode->set_abstract(cnode->abstract()->Clone());
  return custom_cnode;
}

bool CpuKernelBuilder::CompileJsonsInAnfnodes(const AnfNodePtrList &node_list) {
  if (GraphKernelFlags::GetInstance().enable_dynamic_batch) {
    for (auto &node : node_list) {
      auto gk_fg = GetCNodeFuncGraph(node);
      MS_EXCEPTION_IF_NULL(gk_fg);
      std::string dynamic_input_index = GetCNodeDynamicInputIndex(node->cast<CNodePtr>());
      if (!dynamic_input_index.empty()) {
        gk_fg->set_attr("dynamic_input_index", MakeValue(dynamic_input_index));
      }
    }
  }
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
