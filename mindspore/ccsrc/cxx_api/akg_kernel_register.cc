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
#ifdef _MSC_VER
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <mutex>
#include <memory>
#include <string>
#include <fstream>
#include "kernel/oplib/oplib.h"

static std::mutex init_mutex;
static bool Initialized = false;

namespace mindspore {
static bool RegAllOpFromFile() {
  std::string dir;
#ifndef _MSC_VER
  Dl_info info;
  int dl_ret = dladdr(reinterpret_cast<void *>(RegAllOpFromFile), &info);
  if (dl_ret == 0) {
    MS_LOG(INFO) << "Get dladdr failed, skip.";
    return false;
  }
  dir = info.dli_fname;
#else
  HMODULE hModule = nullptr;
  if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                        (LPCSTR)RegAllOpFromFile, &hModule) != 0) {
    char szPath[MAX_PATH];
    if (GetModuleFileName(hModule, szPath, sizeof(szPath)) != 0) {
      dir = std::string(szPath);
    }
  } else {
    MS_LOG(INFO) << "Get GetModuleHandleEx failed, skip.";
    return false;
  }
#endif
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
#ifdef _MSC_VER
  if (_fullpath(real_path_mem, common::SafeCStr(dir), PATH_MAX) == nullptr) {
    MS_LOG(ERROR) << "Op info path is invalid: " << dir;
    return false;
  }
#else
  if (realpath(common::SafeCStr(dir), real_path_mem) == nullptr) {
    MS_LOG(ERROR) << "Op info path is invalid: " << dir;
    return false;
  }
#endif
  std::string real_path(real_path_mem);

  MS_LOG(INFO) << "Start to read op info from local file " << real_path;
  std::ifstream file(real_path);
  if (!file.is_open()) {
    MS_LOG(ERROR) << "Find op info file failed.";
    return false;
  }
  kernel::OpLib::GetOpInfoMap().clear();
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
