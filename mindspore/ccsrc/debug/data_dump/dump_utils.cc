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
#include "include/backend/debug/data_dump/dump_utils.h"
#include <dirent.h>
#ifdef ENABLE_DEBUGGER
#include <sys/stat.h>
#endif
#include <map>
#include <vector>
#include <stack>
#include <queue>
#include <algorithm>

#include "runtime/device/ms_device_shape_transfer.h"
#include "utils/ms_context.h"
#include "pipeline/jit/debug/anf_ir_utils.h"
#include "include/backend/debug/data_dump/dump_json_parser.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "include/common/utils/utils.h"
#include "include/common/debug/common.h"
#include "runtime/graph_scheduler/device_tensor_store.h"
#include "mindspore/core/utils/file_utils.h"

using mindspore::runtime::DeviceTensorStore;

namespace mindspore {
static std::vector<std::string> g_overflow_operators;

uint32_t ConvertPhysicalDeviceId(uint32_t device_id) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto device_target = context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto kernel_runtime = device::KernelRuntimeManager::Instance().GetSingleKernelRuntime(device_target, device_id);
  MS_EXCEPTION_IF_NULL(kernel_runtime);
  return kernel_runtime->device_id();
}

std::string GenerateDumpPath(uint32_t graph_id, uint32_t rank_id, bool is_cst) {
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  std::string net_name = dump_json_parser.net_name();
  std::string iterator = std::to_string(dump_json_parser.cur_dump_iter());
  std::string dump_path = dump_json_parser.path();
  if (dump_path.back() != '/') {
    dump_path += "/";
  }
  if (is_cst) {
    dump_path += ("rank_" + std::to_string(rank_id) + "/" + net_name + "/" + std::to_string(graph_id) + "/constants/");
  } else {
    dump_path +=
      ("rank_" + std::to_string(rank_id) + "/" + net_name + "/" + std::to_string(graph_id) + "/" + iterator + "/");
  }
  return dump_path;
}

void GetFileKernelName(NotNull<std::string *> kernel_name) {
  const std::string strsrc = "/";
  const std::string strdst = "--";
  std::string::size_type pos = 0;
  std::string::size_type srclen = strsrc.size();
  std::string::size_type dstlen = strdst.size();
  while ((pos = kernel_name->find(strsrc, pos)) != std::string::npos) {
    kernel_name->replace(pos, srclen, strdst);
    pos += dstlen;
  }
}

void GetDumpIntShape(const AnfNodePtr &node, size_t index, NotNull<ShapeVector *> const int_shapes, bool trans_flag) {
  if (trans_flag) {
    *int_shapes = trans::GetRuntimePaddingShape(node, index);
  } else {
    *int_shapes = AnfAlgo::GetOutputDeviceShape(node, index);
  }
}

const DeviceTensorPtr GetParameterInfo(const AnfNodePtr &node, NotNull<ShapeVector *> const int_shapes,
                                       NotNull<TypeId *> const host_type, NotNull<TypeId *> const device_type) {
  const auto &device_tensors = DeviceTensorStore::GetInstance().Fetch(node.get());
  if (device_tensors.size() < 1) {
    return nullptr;
  }
  auto device_addr = device_tensors[0];
  MS_EXCEPTION_IF_NULL(device_addr);
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  bool trans_flag = dump_json_parser.trans_flag();
  auto ref_node = device_addr->GetNodeIndex().first;
  MS_EXCEPTION_IF_NULL(ref_node);
  GetDumpIntShape(ref_node, kParameterOutputIndex, int_shapes, trans_flag);
  *host_type = common::AnfAlgo::GetOutputInferDataType(ref_node, kParameterOutputIndex);
  *device_type = AnfAlgo::GetOutputDeviceDataType(ref_node, kParameterOutputIndex);
  return device_addr;
}

void DumpMemToFile(const std::string &file_path, const device::DeviceAddress &addr, const ShapeVector &int_shapes,
                   const TypeId &type, bool trans_flag) {
  auto format = kOpFormat_DEFAULT;
  auto ret = addr.DumpMemToFile(file_path, format, int_shapes, type, trans_flag);
  if (!ret) {
    MS_LOG(ERROR) << "DumpMemToFile Failed: flag:" << trans_flag << ", path:" << file_path << ", host_format:" << format
                  << ".!";
  }
}

std::string GetOpNameWithoutScope(const std::string &fullname_with_scope, const std::string &separator) {
  std::size_t found = fullname_with_scope.rfind(separator);
  std::string op_name;
  if (found != std::string::npos) {
    op_name = fullname_with_scope.substr(found + separator.length());
  }
  return op_name;
}

void DumpToFile(const std::string &file_name, const std::string &dump_str) {
  if (dump_str.empty()) {
    MS_LOG(ERROR) << "Failed to dump empty tensor data.";
    return;
  }

  auto real_path = Common::CreatePrefixPath(file_name);
  if (!real_path.has_value()) {
    MS_LOG(ERROR) << "CreatePrefixPath failed.";
    return;
  }
  std::string real_path_str = real_path.value();
  ChangeFileMode(real_path_str, S_IWUSR);
  std::ofstream file(real_path_str, std::ofstream::out | std::ofstream::trunc);
  if (!file.is_open()) {
    MS_LOG(EXCEPTION) << "Open file " << real_path_str << "failed: " << ErrnoToString(errno);
  }
  file << dump_str;
  if (file.bad()) {
    file.close();
    MS_LOG(EXCEPTION) << "Dump string to file " << real_path_str << " failed: " << ErrnoToString(errno);
  }
  file.close();
  ChangeFileMode(real_path_str, S_IRUSR);
}

#ifdef ENABLE_DEBUGGER
bool IsFolder(const std::string &file_path) {
  struct stat st;
  if (lstat(file_path.c_str(), &st) != 0) {
    return false;
  }
  return S_ISDIR(st.st_mode);
}

bool IsEmptyFolder(const std::string &dir_path) {
  int dir_count = 0;
  DIR *d = opendir(dir_path.c_str());
  struct dirent *dir = nullptr;
  while ((dir = readdir(d)) != nullptr) {
    std::string name = dir->d_name;
    if (name == "." || name == "..") {
      continue;
    } else {
      dir_count++;
    }
  }
  (void)closedir(d);
  if (dir_count == 0) {
    return true;
  }
  return false;
}

void RemoveEmptyDir(const std::string &dir_path) {
  if (!IsFolder(dir_path)) {
    MS_LOG(WARNING) << "the path = " << dir_path.c_str() << "is not a folder";
    return;
  }
  std::stack<std::string> dirs_stack;
  std::queue<std::string> dirs_queue;
  dirs_queue.push(dir_path);
  while (!dirs_queue.empty()) {
    std::string &folder = dirs_queue.front();
    std::unique_ptr<DIR, int (*)(DIR *)> dir(opendir(folder.c_str()), &closedir);
    if (!dir) {
      MS_LOG(WARNING) << "the path = " << dir_path.c_str() << "is not exist";
      return;
    }
    dirent *dt;
    while ((dt = readdir(dir.get())) != nullptr) {
      std::string name = dt->d_name;
      if (name == "." || name == "..") {
        continue;
      }
      std::string sub_path = folder + "/" + name;
      if (IsFolder(sub_path)) {
        dirs_queue.push(sub_path);
      }
    }
    dirs_stack.push(folder);
    dirs_queue.pop();
  }

  while (!dirs_stack.empty()) {
    std::string &folder_stack = dirs_stack.top();
    if (!IsEmptyFolder(folder_stack)) {
      MS_LOG(INFO) << "the folder path in dirs_stack: " << folder_stack.c_str() << " is not empty folder";
    } else {
      if (remove(folder_stack.c_str()) != 0) {
        MS_LOG(WARNING) << "delete folder path in dirs_stack: " << folder_stack.c_str() << " is failed.";
      }
    }
    dirs_stack.pop();
  }
}

std::vector<std::string> Split(const std::string &input, const std::string &pattern) {
  std::string str = input;
  std::string::size_type pos;
  std::vector<std::string> result;
  str += pattern;
  size_t len = str.size();

  for (size_t i = 0; i < len; i++) {
    pos = str.find(pattern, i);
    if (pos < len) {
      std::string sub_s = str.substr(i, pos - i);
      result.push_back(sub_s);
      i = pos + pattern.size() - 1;
    }
  }
  return result;
}

void SaveOverflowOperator(const std::string &iterator, const std::string &dump_rank_path) {
  const std::string overflow_dump_dir = "debug_files";
  const std::string overflow_file_prefix = "Opdebug.Node_OpDebug.";
  const std::string cur_step_overflow_path = dump_rank_path + "/" + overflow_dump_dir + "/" + iterator;
  DIR *d = opendir(cur_step_overflow_path.c_str());
  g_overflow_operators.clear();
  if (d == nullptr) {
    MS_LOG(INFO) << "Overflow file directory does not exist!";
  } else {
    struct dirent *dir = nullptr;
    while ((dir = readdir(d)) != nullptr) {
      std::string filename = dir->d_name;
      if (filename.find(overflow_file_prefix) != std::string::npos) {
        const int kNumDots = 2;
        const int first_dot = 0;
        int dots_count = 0;
        int pos_start = overflow_file_prefix.size() - 1;
        std::string filename_substr = filename.substr(pos_start);
        size_t third_dot = filename_substr.find(".");
        while (dots_count != kNumDots && third_dot != std::string::npos) {
          third_dot = filename_substr.find(".", third_dot + 1);
          dots_count++;
        }
        std::string stream_task_name = filename_substr.substr(first_dot, third_dot + 1);
        g_overflow_operators.emplace_back(stream_task_name);
      }
    }
    (void)closedir(d);
  }
}

void DeleteNoOverflowFile(uint32_t rank_id, uint32_t graph_id) {
  auto &json_parser = DumpJsonParser::GetInstance();
  if (!(json_parser.async_dump_enabled() || json_parser.e2e_dump_enabled())) {
    return;
  }
  const int max_task_id = 65536;
  const int least_dots_num = 3;
  const int two_dots_num = 2;
  const int one_dots_num = 1;
  std::string cur_dump_path = json_parser.path() + "/rank_" + std::to_string(rank_id);
  std::string net_name_ = json_parser.net_name();
  std::string iterator = std::to_string(json_parser.cur_dump_iter());
  SaveOverflowOperator(iterator, cur_dump_path);
  std::string overflow_operator_dump_path =
    cur_dump_path + "/" + net_name_ + "/" + std::to_string(graph_id) + "/" + iterator;
  DIR *d = opendir(overflow_operator_dump_path.c_str());
  if (d == nullptr) {
    MS_LOG(INFO) << "Overflow iterator file directory does not exist!";
  } else {
    struct dirent *dir = nullptr;
    while ((dir = readdir(d)) != nullptr) {
      std::string filename = dir->d_name;
      if (filename == "." || filename == "..") {
        continue;
      }
      const std::string tmp_filename = filename;
      auto filename_splits = Split(filename, ".");
      int split_len = filename_splits.size();
      if (split_len < least_dots_num) {
        MS_LOG(WARNING) << "Overflow operator file format is incorrect";
        continue;
      }
      auto task_id = static_cast<uint32_t>(std::stoi(filename_splits.at(split_len - least_dots_num)));
      if (task_id >= max_task_id) {
        auto mod_val = task_id % max_task_id;
        filename = "." + std::to_string(mod_val) + "." + filename_splits.at(split_len - two_dots_num) + "." +
                   filename_splits.at(split_len - one_dots_num);
      }
      bool is_exist =
        std::any_of(std::begin(g_overflow_operators), std::end(g_overflow_operators),
                    [&](std::string stream_task_str) { return filename.find(stream_task_str) != std::string::npos; });
      if (!is_exist) {
        auto ret = remove((overflow_operator_dump_path + "/" + tmp_filename).c_str());
        if (ret == 0) {
          MS_LOG(INFO) << "Delete file successfully, filename is:" << tmp_filename.c_str();
        }
      }
    }
    (void)closedir(d);
    std::string overflow_file_graph_path = cur_dump_path + "/" + net_name_ + "/" + std::to_string(graph_id);
    RemoveEmptyDir(overflow_file_graph_path);
  }
}
#endif
}  // namespace mindspore
