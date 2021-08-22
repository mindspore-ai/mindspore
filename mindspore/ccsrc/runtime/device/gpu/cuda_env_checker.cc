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

#include "runtime/device/gpu/cuda_env_checker.h"
#include <dirent.h>
#include <cstdlib>
#include <algorithm>
#include "utils/log_adapter.h"

namespace mindspore {
namespace device {
namespace gpu {
bool CudaEnvChecker::CheckNvccInPath() {
  if (already_check_nvcc_) {
    return find_nvcc_;
  }

  auto checker = [](const std::string &path) {
    bool find_nvcc = false;
    DIR *dir = opendir(path.c_str());
    if (dir == nullptr) {
      return find_nvcc;
    }
    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {
      std::string bin_file = entry->d_name;
      if (bin_file == kNvcc) {
        find_nvcc = true;
        break;
      }
    }
    (void)closedir(dir);
    return find_nvcc;
  };

  std::set<std::string> paths;
  GetRealPaths(&paths);
  find_nvcc_ = std::any_of(paths.begin(), paths.end(), checker);
  already_check_nvcc_ = true;
  return find_nvcc_;
}

void CudaEnvChecker::GetRealPaths(std::set<std::string> *paths) const {
  if (paths == nullptr) {
    MS_LOG(ERROR) << "The pointer paths is nullptr";
    return;
  }
  auto env_paths_ptr = std::getenv(kPathEnv);
  if (env_paths_ptr == nullptr) {
    MS_LOG(ERROR) << "Please export environment variable PATH";
    return;
  }
  std::string env_paths = env_paths_ptr;
  if (env_paths.empty()) {
    MS_LOG(ERROR) << "Empty environment variable PATH";
    return;
  }

  std::string cur_path;
  for (const auto &ch : env_paths) {
    if (ch != ':') {
      cur_path += ch;
      continue;
    }
    if (!cur_path.empty()) {
      (void)paths->insert(cur_path);
    }
    cur_path.clear();
  }
  if (!cur_path.empty()) {
    (void)paths->insert(cur_path);
  }
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
