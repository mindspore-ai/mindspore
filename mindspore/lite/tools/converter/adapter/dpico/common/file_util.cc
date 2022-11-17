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

#include "common/file_util.h"
#include <vector>
#include <string>
#include <fstream>
#include <climits>
#include <memory>
#include <cstring>
#include "mindapi/base/logging.h"

namespace mindspore {
namespace dpico {
namespace {
#ifdef _WIN32
constexpr size_t kMaxPathLen = 1024;
#endif
}  // namespace
int CreateDir(std::string *file_path) {  // support multi-level directory generated.
  if (file_path == nullptr) {
    MS_LOG(ERROR) << "file_path is nullptr.";
    return RET_ERROR;
  }
  if (file_path->empty()) {
    MS_LOG(ERROR) << "input file path is empty.";
    return RET_ERROR;
  } else if (file_path->size() >= PATH_MAX) {
    MS_LOG(ERROR) << "input file path is too long";
    return RET_ERROR;
  }

  if (file_path->back() != '\\' && file_path->back() != '/') {
#ifdef _WIN32
    *file_path += "\\";
#else
    *file_path += "/";
#endif
  }
  for (size_t i = 0; i < file_path->size(); i++) {
    if (file_path->at(i) == '\\' || file_path->at(i) == '/') {
      if (AccessFile(file_path->substr(0, i + 1), F_OK) != 0) {
        int ret = Mkdir(file_path->substr(0, i + 1));
        if (ret != RET_OK) {
          MS_LOG(ERROR) << "mkdir failed. " << file_path->substr(0, i + 1);
          return RET_ERROR;
        }
      }
    }
  }

  return RET_OK;
}

std::string RealPath(const char *path) {
  if (path == nullptr) {
    MS_LOG(ERROR) << "path is nullptr";
    return "";
  }
  if ((std::strlen(path)) >= PATH_MAX) {
    MS_LOG(ERROR) << "path is too long";
    return "";
  }
  auto resolved_path = std::make_unique<char[]>(PATH_MAX);
  if (resolved_path == nullptr) {
    MS_LOG(ERROR) << "new resolved_path failed";
    return "";
  }
#ifdef _WIN32
  char *real_path = _fullpath(resolved_path.get(), path, kMaxPathLen);
#else
  char *real_path = realpath(path, resolved_path.get());
#endif
  if (real_path == nullptr || strlen(real_path) == 0) {
    MS_LOG(ERROR) << "file path is not valid : " << path;
    return "";
  }
  std::string res = resolved_path.get();
  return res;
}

int ReadFileToIfstream(const std::string &file_path, std::ifstream *ifstream) {
  if (file_path.empty()) {
    MS_LOG(ERROR) << "file path is empty.";
    return RET_ERROR;
  }
  auto real_file_path = RealPath(file_path.c_str());
  if (real_file_path.empty()) {
    MS_LOG(ERROR) << "get real path failed.";
    return RET_ERROR;
  }
  ifstream->open(real_file_path, std::ios::in);
  if (!ifstream->good()) {
    MS_LOG(ERROR) << "file: " << real_file_path << " is not exist.";
    return RET_ERROR;
  }
  if (!ifstream->is_open()) {
    MS_LOG(ERROR) << "file: " << real_file_path << " open failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int RemoveDir(const std::string &path) {
  if (path.empty()) {
    MS_LOG(ERROR) << "input path is empty.";
    return RET_ERROR;
  }
  std::string str_path = path;
#ifdef _WIN32
  struct _finddata_t fb;
  if (str_path.at(str_path.length() - 1) != '\\') {
    (void)str_path.append("\\");
  }
  std::string find_path = str_path + "*";
  intptr_t handle = _findfirst(find_path.c_str(), &fb);
  if (handle != -1L) {
    std::string tmp_path;
    do {
      if (strcmp(fb.name, "..") != 0 && strcmp(fb.name, ".") != 0) {
        tmp_path.clear();
        tmp_path = str_path + std::string(fb.name);
        if (fb.attrib == _A_SUBDIR) {
          (void)RemoveDir(tmp_path.c_str());
        } else {
          remove(tmp_path.c_str());
        }
      }
    } while (_findnext(handle, &fb) == 0);
    _findclose(handle);
  }
  return RMDIR(str_path.c_str()) == 0 ? RET_OK : RET_ERROR;

#else
  if (str_path.at(str_path.length() - 1) != '/') {
    (void)str_path.append("/");
  }
  DIR *d = opendir(str_path.c_str());
  if (d != nullptr) {
    struct dirent *dt = nullptr;
    dt = readdir(d);
    while (dt != nullptr) {
      if (std::strcmp(dt->d_name, "..") != 0 && std::strcmp(dt->d_name, ".") != 0) {
        struct stat st {};
        auto file_name = str_path + std::string(dt->d_name);
        (void)stat(file_name.c_str(), &st);
        if (S_ISDIR(st.st_mode)) {
          (void)RemoveDir(file_name);
        } else {
          (void)remove(file_name.c_str());
        }
      }
      dt = readdir(d);
    }
    (void)closedir(d);
  }
  return rmdir(str_path.c_str()) == 0 ? RET_OK : RET_ERROR;
#endif
}
}  // namespace dpico
}  // namespace mindspore
