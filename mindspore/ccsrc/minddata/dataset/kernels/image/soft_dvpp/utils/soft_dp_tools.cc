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

#include "minddata/dataset/kernels/image/soft_dvpp/utils/soft_dp_tools.h"
#include <sys/stat.h>
#include <cstring>
#include "minddata/dataset/kernels/image/soft_dvpp/utils/soft_dp_check.h"

const uint32_t kMaxPath = 4096;

std::pair<bool, std::string> GetRealpath(const std::string &path) {
  char resolvedPath[kMaxPath];
#ifndef DVPP_UTST
  if (path.size() > kMaxPath) {
    API_LOGD("path size too large.");
    return std::make_pair(false, std::string(strerror(errno)));
  }
#endif  // !DVPP_UTST

#ifdef _WIN32
  auto err = _fullpath(resolvedPath, path.c_str(), kMaxPath);
#else
  auto err = realpath(path.c_str(), resolvedPath);
#endif
  if (err == nullptr) {
    return std::make_pair(false, std::string(strerror(errno)));
  } else {
    return std::make_pair(true, std::string(resolvedPath, strlen(resolvedPath)));
  }
}

bool IsDirectory(const std::string &path) {
  struct stat buf {};
  if (stat(path.c_str(), &buf) != 0) {
    return false;
  }

  return S_ISDIR(buf.st_mode);
}
