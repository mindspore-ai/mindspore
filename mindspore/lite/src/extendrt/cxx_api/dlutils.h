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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_DLUTILS_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_DLUTILS_H_
#include <string>
#if !defined(_WIN32) && !defined(_WIN64)
#include <dlfcn.h>
#include <dirent.h>
#include <memory>
#include <fstream>
#include "utils/file_utils.h"
#include "include/api/status.h"

namespace mindspore {
inline Status DLSoPath(const std::string &benchmark_so, const std::string &target_so, std::string *target_so_path) {
  if (target_so_path == nullptr) {
    return Status(kMEFailed, "Input so_path can not be nullptr.");
  }
  Dl_info dl_info;
  dladdr(reinterpret_cast<void *>(DLSoPath), &dl_info);
  std::string cur_so_path = dl_info.dli_fname;

  auto pos = cur_so_path.find(benchmark_so);
  if (pos == std::string::npos) {
    return Status(kMEFailed, "Could not find benchmark so " + benchmark_so + " check path " + cur_so_path);
  }
  std::string parent_dir = cur_so_path.substr(0, pos);
  std::string found_target_so;

  DIR *dir = opendir(parent_dir.c_str());
  if (dir != nullptr) {
    // access all the files and directories within directory
    dirent *ent = readdir(dir);
    while (ent != nullptr) {
      if (std::string(ent->d_name).find(target_so) != std::string::npos) {
        found_target_so = std::string(ent->d_name);
        break;
      }
      ent = readdir(dir);
    }
    (void)closedir(dir);
  } else {
    return Status(kMEFailed, "Could not open directory: " + parent_dir);
  }
  if (found_target_so.empty()) {
    return Status(kMEFailed, "Could not find target so " + target_so + " in " + parent_dir);
  }
  std::string unreal_path = parent_dir + found_target_so;
  auto realpath = FileUtils::GetRealPath(unreal_path.c_str());
  if (!realpath.has_value()) {
    return Status(kMEFailed, "Get target so " + target_so + " real path failed, path: " + unreal_path);
  }

  *target_so_path = realpath.value();
  return kSuccess;
}

inline Status DLSoOpen(const std::string &dl_path, const std::string &func_name, void **handle, void **function,
                       bool runtime_convert = false) {
  // do dlopen and export functions from c_dataengine
  if (handle == nullptr) {
    MS_LOG(WARNING) << "Input parameter handle cannot be nullptr";
    return Status(kMEFailed, "Input parameter handle cannot be nullptr");
  }
  int mode = runtime_convert ? RTLD_GLOBAL : RTLD_LOCAL;
  *handle = dlopen(dl_path.c_str(), RTLD_LAZY | mode);

  auto get_dl_error = []() -> std::string {
    auto error = dlerror();
    return error == nullptr ? "" : error;
  };
  if (*handle == nullptr) {
    auto error = get_dl_error();
    MS_LOG(WARNING) << "dlopen " << dl_path << " failed, error: " << error;
    return Status(kMEFailed, "dlopen " + dl_path + " failed, error: " + error);
  }
  if (!func_name.empty()) {
    if (function == nullptr) {
      MS_LOG(WARNING) << "Input parameter function cannot be nullptr";
      return Status(kMEFailed, "Input parameter function cannot be nullptr");
    }
    *function = dlsym(*handle, func_name.c_str());
    if (*function == nullptr) {
      auto error = get_dl_error();
      MS_LOG(WARNING) << "Could not find " + func_name + " in " + dl_path + ", error: " << error;
      return Status(kMEFailed, "Could not find " + func_name + " in " + dl_path + ", error: " + error);
    }
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
#else
inline Status DLSoPath(const std::string &benchmark_so, const std::string &target_so, std::string *target_so_path) {
  MS_LOG(ERROR) << "Not support dlopen so";
  return kMEFailed;
}

inline Status DLSoOpen(const std::string &dl_path, const std::string &func_name, void **handle, void **function,
                       bool runtime_convert = false) {
  MS_LOG(ERROR) << "Not support dlopen so";
  return kMEFailed;
}
#endif
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_DLUTILS_H_
