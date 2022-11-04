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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_COMMON_FILE_UTIL_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_COMMON_FILE_UTIL_H_

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>
#endif
#include <string>
#include <vector>
#include <fstream>
#include <map>
#include "include/errorcode.h"
#include "mindapi/base/logging.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::STATUS;

namespace mindspore {
namespace dpico {
constexpr size_t kMaximumNumOfFolders = 1000;

inline int AccessFile(const std::string &file_path, int access_mode) {
#ifdef _WIN32
  return _access(file_path.c_str(), access_mode);
#else
  return access(file_path.c_str(), access_mode);
#endif
}

inline int Mkdir(const std::string &file_path) {
#ifdef _WIN32
  return _mkdir(file_path.c_str());
#else
  return mkdir(file_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif
}

inline int WriteToBin(const std::string &file_path, void *data, const size_t size) {
  if (data == nullptr) {
    MS_LOG(ERROR) << "data is nullptr.";
    return RET_ERROR;
  }
  std::ofstream out_file;
  out_file.open(file_path.c_str(), std::ios::binary);
  if (!out_file.good() || !out_file.is_open()) {
    return RET_ERROR;
  }
  (void)out_file.write(reinterpret_cast<char *>(data), size);
  out_file.close();
  return RET_OK;
}

std::string RealPath(const char *path);
int CreateDir(std::string *file_path);
int ReadFileToIfstream(const std::string &file_path, std::ifstream *ifstream);
int RemoveDir(const std::string &path);
}  // namespace dpico
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_COMMON_FILE_UTIL_H_
