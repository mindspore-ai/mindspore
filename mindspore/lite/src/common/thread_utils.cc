/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifdef __linux__
#include "src/common/thread_utils.h"
#include <sys/stat.h>
#include <wait.h>
#include "src/common/log_adapter.h"

namespace mindspore {
namespace lite {
namespace {
constexpr int kDefaultFolders = 2;
}  // namespace
Status CheckPidStatus(pid_t pid) {
  int status;
  pid_t w = waitpid(pid, &status, WUNTRACED | WCONTINUED);
  if (w < 0) {
    return kLiteError;
  }
  if (WIFEXITED(status)) {                         // normal exit
    if (WEXITSTATUS(status) != kProcessSuccess) {  // caused by asan
      return kLiteGraphFileError;
    } else {
      MS_LOG(INFO) << "Pre build and run for death test passed.";
      return kSuccess;
    }
  }
  return kLiteGraphFileError;  // abnormal exit, such as process killed or other signal, WIFSTOPPED, WIFCONTINUED
}

int GetNumThreads() {
  struct stat task_stat {};
  if (stat("/proc/self/task", &task_stat) != 0) {
    MS_LOG(ERROR) << "stat \"/proc/self/task\" failed.";
    return -1;
  }
  return static_cast<int>(task_stat.st_nlink) - kDefaultFolders;
}
}  // namespace lite
}  // namespace mindspore
#endif
