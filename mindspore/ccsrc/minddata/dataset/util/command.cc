/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/util/command.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdlib.h>
#if !defined(_WIN32) && !defined(_WIN64)
#include <sys/types.h>
#include <sys/wait.h>
#endif
#include "utils/os.h"

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/util/log_adapter.h"

namespace mindspore {
namespace dataset {
#if !defined(_WIN32) && !defined(_WIN64)
void MonitorLoop(std::mutex *monitor_mtx, std::condition_variable *monitor_cv, bool *monitor_exit_flag,
                 const int32_t &worker_pid, const std::string &op_type) {
  if (monitor_mtx == nullptr || monitor_cv == nullptr || monitor_exit_flag == nullptr) {
    MS_LOG(WARNING) << "The input parameters monitor_mtx, monitor_cv and monitor_exit_flag should not be nullptr.";
    return;
  }
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  auto check_interval = cfg->multiprocessing_timeout_interval();
  auto start_time = std::chrono::system_clock::now();

  while (true) {
    // Waiting for the condition variable to be triggered or times out
    std::unique_lock<std::mutex> lock(*monitor_mtx);
    (*monitor_cv).wait_for(lock, std::chrono::seconds(check_interval), [&] { return *monitor_exit_flag; });
    if (*monitor_exit_flag) {
      return;
    }

    auto cost_time =
      std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - start_time).count();

    std::string low_op_type = op_type;
    for (char &c : low_op_type) {
      c = std::tolower(c);
    }

    MS_LOG(WARNING) << "It has been waiting for " << cost_time << "s because the sub-process worker of the "
                    << low_op_type
                    << " operation is hanging. "
                       "Check whether the user defined data transform is too slow or the "
                       "output data is too large. You can also set the timeout interval by "
                       "ds.config.set_multiprocessing_timeout_interval to adjust the output frequency "
                       "of this log.";
    MS_LOG(WARNING) << op_type << " worker subprocess ID " << worker_pid << " is stuck.";

    int status = 0;
    if (waitpid(worker_pid, &status, WNOHANG) != 0) {
      MS_LOG(WARNING) << op_type << " worker subprocess ID " << worker_pid << " has exited.";
      return;
    }

    // get the worker process stack
    PrintPythonStack(worker_pid, op_type);
  }
}

const int32_t output_stack_len = 4096;
const int32_t buf_len = 1024;
const int32_t cmd_len = 1024;
std::atomic<uint32_t> inc_id_cmd(0);

std::mutex mtx;

void PrintPythonStack(const int32_t &worker_id, const std::string &op_type) {
  std::string input_cmd = "py-spy dump -p " + std::to_string(worker_id);

  // gen the output file
  std::stringstream ss;
  ss << std::this_thread::get_id();
  std::string output_filename =
    "/tmp/" + std::to_string(getpid()) + "_" + ss.str() + "_" + std::to_string(inc_id_cmd++);

  // combine the whold cmd
  std::string whole_cmd = input_cmd + " >" + output_filename + " 2>&1;";

  // copy the input to local variable
  char cmd[cmd_len] = {0};
  if (strcpy_s(cmd, cmd_len, whole_cmd.c_str()) != EOK) {
    MS_LOG(WARNING) << "ExecuteCMD strcpy_s failed.";
    return;
  }

  {
    // execute the cmd
    std::unique_lock<std::mutex> lock(mtx);
    if (system(cmd) == -1) {
      MS_LOG(WARNING) << "ExecuteCMD system(\"" << cmd << "\", \"r\") failed.";
      return;
    }
  }

  // check the output file
  char canonical_path[PATH_MAX] = {0x00};
  if (realpath(output_filename.c_str(), canonical_path) == nullptr) {
    MS_LOG(WARNING) << "ExecuteCMD check the output file failed.";
    return;
  }

  // read the output file
  std::ifstream ifs(canonical_path, std::ios::in);
  if (!ifs.is_open()) {
    MS_LOG(WARNING) << "ExecuteCMD read file: " << canonical_path << " failed.";
    return;
  }

  int32_t offset = 0;
  int32_t reserved = 2;
  char output_stack[output_stack_len] = {0};
  char c = ifs.get();
  while (ifs.good()) {
    output_stack[offset] = c;
    if (offset >= output_stack_len - reserved) {
      break;
    }
    offset += 1;
    c = ifs.get();
  }
  ifs.close();

  std::string output_stack_str(output_stack);
  if (output_stack_str.find("command not found") != std::string::npos ||
      output_stack_str.find("No such file or directory") != std::string::npos) {
    MS_LOG(WARNING) << "Please `pip install py-spy` to get the stacks of the stuck process.";
  } else {
    MS_LOG(WARNING) << op_type << " worker subprocess stack:\n" << output_stack_str;
  }

  // remove the output file
  if (remove(canonical_path) != 0) {
    MS_LOG(WARNING) << "ExecuteCMD remove file: " << canonical_path << " failed.";
  }
}
#endif
}  // namespace dataset
}  // namespace mindspore
