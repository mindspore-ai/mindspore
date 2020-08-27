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

  auto checker = [](const std::string &cuda_path) {
    bool find_nvcc = false;
    DIR *dir = opendir(cuda_path.c_str());
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

  auto cuda_paths = GetCudaRealPaths();
  find_nvcc_ = any_of(cuda_paths.begin(), cuda_paths.end(), checker);
  already_check_nvcc_ = true;
  return find_nvcc_;
}

std::vector<std::string> CudaEnvChecker::GetCudaRealPaths() const {
  std::vector<std::string> res;
  auto env_paths_ptr = std::getenv(kPathEnv);
  if (env_paths_ptr == nullptr) {
    MS_LOG(ERROR) << "Please export env: PATH";
    return res;
  }
  std::string env_paths = env_paths_ptr;
  if (env_paths.empty()) {
    MS_LOG(ERROR) << "env PATH is empty";
    return res;
  }

  std::string cur_path;
  for (const auto &ch : env_paths) {
    if (ch != ':') {
      cur_path += ch;
      continue;
    }
    auto real_path_pair = IsCudaRealPath(cur_path);
    if (real_path_pair.second) {
      res.push_back(real_path_pair.first);
    }
    cur_path.clear();
  }
  if (!cur_path.empty()) {
    auto last_real_path_pair = IsCudaRealPath(cur_path);
    if (last_real_path_pair.second) {
      res.push_back(last_real_path_pair.first);
    }
  }
  return res;
}

std::pair<std::string, bool> CudaEnvChecker::IsCudaRealPath(const std::string &path) const {
  std::string real_path = path;
  bool valid_path = false;

  // 8: string length of kCudaSoftLinkPath
  if (real_path.size() < 8) {
    return {"", false};
  }

  // remove redundance space in path
  auto front_space_pos = real_path.find_first_not_of(' ');
  if (front_space_pos != 0) {
    real_path.erase(0, front_space_pos);
  }
  auto back_space_pos = real_path.find_last_not_of(' ');
  if (back_space_pos != real_path.size() - 1) {
    real_path.erase(back_space_pos + 1);
  }

  auto cuda_softlink_path_pos = real_path.rfind(kCudaSoftLinkPath);
  auto cuda_real_path_pos = real_path.rfind(kCudaRealPath);
  auto start = (cuda_softlink_path_pos == std::string::npos || cuda_real_path_pos == std::string::npos)
                 ? std::min(cuda_softlink_path_pos, cuda_real_path_pos)
                 : std::max(cuda_softlink_path_pos, cuda_real_path_pos);
  if (start == std::string::npos) {
    return {"", false};
  }

  auto end = real_path.find('n', start);
  valid_path = (end == real_path.size() - 1) ? true : ((end == real_path.size() - 2) && (real_path.back() == '/'));
  return {real_path.substr(0, end + 1), valid_path};
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
