/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "cxx_api/akg_kernel_register.h"
#include <dlfcn.h>
#include <mutex>
#include <memory>
#include <string>
#include <fstream>
#include "backend/kernel_compiler/oplib/oplib.h"

static std::mutex init_mutex;
static bool Initialized = false;

namespace mindspore {
static bool RegAllOpFromFile() {
  Dl_info info;
  int dl_ret = dladdr(reinterpret_cast<void *>(RegAllOpFromFile), &info);
  if (dl_ret == 0) {
    MS_LOG(INFO) << "Get dladdr failed, skip.";
    return false;
  }
  std::string dir(info.dli_fname);
  MS_LOG(INFO) << "Get library path is " << dir;

  auto split_pos = dir.find_last_of('/');
  if (dir.empty() || split_pos == std::string::npos) {
    MS_LOG(INFO) << "Missing op config file, skip.";
    return false;
  }

  dir = dir.substr(0, split_pos) + "/../config/op_info.config";
  if (dir.size() >= PATH_MAX) {
    MS_LOG(ERROR) << "Op info path is invalid: " << dir;
    return false;
  }

  char real_path_mem[PATH_MAX] = {0};
  if (realpath(common::SafeCStr(dir), real_path_mem) == nullptr) {
    MS_LOG(ERROR) << "Op info path is invalid: " << dir;
    return false;
  }
  std::string real_path(real_path_mem);

  MS_LOG(INFO) << "Start to read op info from local file " << real_path;
  std::ifstream file(real_path);
  if (!file.is_open()) {
    MS_LOG(ERROR) << "Find op info file failed.";
    return false;
  }
  std::string line;
  while (getline(file, line)) {
    if (!line.empty()) {
      (void)kernel::OpLib::RegOp(line, "");
    }
  }
  MS_LOG(INFO) << "End";
  return true;
}

void RegAllOp() {
  std::lock_guard<std::mutex> lock(init_mutex);
  if (Initialized) {
    return;
  }
  bool ret = RegAllOpFromFile();
  if (!ret) {
    MS_LOG(ERROR) << "Register operators failed. The package may damaged or file is missing.";
    return;
  }

  Initialized = true;
}
}  // namespace mindspore
