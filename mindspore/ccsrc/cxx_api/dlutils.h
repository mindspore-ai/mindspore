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
#ifndef MINDSPORE_CCSRC_CXX_API_DLUTILS_H_
#define MINDSPORE_CCSRC_CXX_API_DLUTILS_H_
#if !defined(_WIN32) && !defined(_WIN64)
#include <dlfcn.h>
#include <dirent.h>
#include <memory>
#include <string>
#include <fstream>
#include "utils/file_utils.h"

namespace mindspore {
inline Status DLSoPath(std::string *so_path) {
  if (so_path == nullptr) {
    return Status(kMEFailed, "Input so_path can not be nullptr.");
  }
  Dl_info dl_info;
  dladdr(reinterpret_cast<void *>(DLSoPath), &dl_info);
  std::string libmindspore_so = dl_info.dli_fname;

  auto pos = libmindspore_so.find("libmindspore.so");
  if (pos == std::string::npos) {
    return Status(kMEFailed, "Could not find libmindspore.so, check path.");
  }

  std::string parent_dir = libmindspore_so.substr(0, pos) + "../";
  std::string c_dataengine_so;

  DIR *dir = opendir(parent_dir.c_str());
  if (dir != nullptr) {
    // access all the files and directories within directory
    dirent *ent = readdir(dir);
    while (ent != nullptr) {
      if (std::string(ent->d_name).find("_c_dataengine") != std::string::npos) {
        c_dataengine_so = std::string(ent->d_name);
        break;
      }
      ent = readdir(dir);
    }
    closedir(dir);
  } else {
    return Status(kMEFailed, "Could not open directory: " + parent_dir);
  }

  std::string unreal_path = parent_dir + c_dataengine_so;
  auto realpath = FileUtils::GetRealPath(unreal_path.c_str());
  if (!realpath.has_value()) {
    return Status(kMEFailed, "Get c_dataengine_so real path failed, path: " + unreal_path);
  }

  *so_path = realpath.value();
  return kSuccess;
}

inline Status DLSoOpen(const std::string &dl_path, const std::string &func_name, void **handle, void **function) {
  // do dlopen and export functions from c_dataengine
  *handle = dlopen(dl_path.c_str(), RTLD_LAZY | RTLD_LOCAL);

  if (*handle == nullptr) {
    return Status(kMEFailed, "dlopen failed, the pointer[handle] is null.");
  }

  *function = dlsym(*handle, func_name.c_str());
  if (*function == nullptr) {
    return Status(kMEFailed, "Could not find " + func_name + " in " + dl_path);
  }
  return kSuccess;
}

inline void DLSoClose(void *handle) {
  if (handle != nullptr) {
    (void)dlclose(handle);
  }
}

#define CHECK_FAIL_AND_RELEASE(_s, _handle, _e) \
  do {                                          \
    Status __rc = (_s);                         \
    if (__rc.IsError()) {                       \
      MS_LOG(ERROR) << (_e);                    \
      DLSoClose((_handle));                     \
      return __rc;                              \
    }                                           \
  } while (false)

}  // namespace mindspore
#endif
#endif  // MINDSPORE_CCSRC_CXX_API_DLUTILS_H_
