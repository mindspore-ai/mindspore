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
#include "core/util/file_system_operation.h"
#include <unistd.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <ctime>
#include <fstream>
#include <memory>
#include "mindspore/ccsrc/utils/log_adapter.h"

namespace mindspore {
namespace serving {
char *ReadFile(const char *file, size_t *size) {
  if (file == nullptr) {
    MS_LOG(ERROR) << "file is nullptr";
    return nullptr;
  }
  MS_ASSERT(size != nullptr);
  std::string realPath = file;
  std::ifstream ifs(realPath);
  if (!ifs.good()) {
    MS_LOG(ERROR) << "file: " << realPath << " is not exist";
    return nullptr;
  }

  if (!ifs.is_open()) {
    MS_LOG(ERROR) << "file: " << realPath << "open failed";
    return nullptr;
  }

  ifs.seekg(0, std::ios::end);
  *size = ifs.tellg();
  std::unique_ptr<char> buf(new (std::nothrow) char[*size]);
  if (buf == nullptr) {
    MS_LOG(ERROR) << "malloc buf failed, file: " << realPath;
    ifs.close();
    return nullptr;
  }

  ifs.seekg(0, std::ios::beg);
  ifs.read(buf.get(), *size);
  ifs.close();

  return buf.release();
}

bool DirOrFileExist(const std::string &file_path) {
  int ret = access(file_path.c_str(), 0);
  return (ret == -1) ? false : true;
}

std::vector<std::string> GetAllSubDirs(const std::string &dir_path) {
  DIR *dir = nullptr;
  struct dirent *ptr = nullptr;
  std::vector<std::string> SubDirs;

  if ((dir = opendir(dir_path.c_str())) == NULL) {
    MS_LOG(ERROR) << "Open " << dir_path << " error!";
    return std::vector<std::string>();
  }

  while ((ptr = readdir(dir)) != NULL) {
    std::string name = ptr->d_name;
    if (name == "." || name == "..") {
      continue;
    }
    if (ptr->d_type == DT_DIR) {
      SubDirs.push_back(dir_path + "/" + name);
    }
  }
  closedir(dir);
  std::sort(SubDirs.begin(), SubDirs.end());
  return SubDirs;
}

time_t GetModifyTime(const std::string &file_path) {
  struct stat info;
  (void)stat(file_path.c_str(), &info);
  return info.st_mtime;
}
}  // namespace serving
}  // namespace mindspore
