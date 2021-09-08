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

#include "src/common/file_utils.h"
#ifdef _WIN32
#include <direct.h>
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>
#endif

#include <cstdlib>
#include "securec/include/securec.h"

#if defined(_WIN32) && defined(SUPPORT_MSVC)
#define PATH_MAX 1024
#define F_OK 0
#endif

namespace mindspore {
namespace lite {
namespace {
const int MAXIMUM_NUMBERS_OF_FOLDER = 1000;

inline int AccessFile(const std::string &file_path, int access_mode) {
#ifdef _WIN32
  return _access(file_path.c_str(), access_mode);
#else
  // android access interface always return true
  struct stat st;
  if (stat(file_path.c_str(), &st) == 0) {
    mode_t perm = st.st_mode;
    auto can_read = perm & S_IRUSR;
    return (can_read && access(file_path.c_str(), access_mode) == 0) ? 0 : -1;
  }
  return -1;
#endif
}

inline int Mkdir(const std::string &file_path) {
#ifdef _WIN32
  return _mkdir(file_path.c_str());
#else
  return mkdir(file_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif
}
}  // namespace

char *ReadFile(const char *file, size_t *size) {
  if (file == nullptr) {
    MS_LOG(ERROR) << "file is nullptr";
    return nullptr;
  }
  MS_ASSERT(size != nullptr);
  std::string real_path = RealPath(file);
  if (AccessFile(real_path, R_OK) != 0) {
    MS_LOG(ERROR) << "cannot access file:" << real_path << ".please check file if exists and file mod";
    return nullptr;
  }
  std::ifstream ifs(real_path, std::ifstream::in | std::ifstream::binary);
  if (!ifs.good()) {
    MS_LOG(ERROR) << "file: " << real_path << " is not exist";
    return nullptr;
  }

  if (!ifs.is_open()) {
    MS_LOG(ERROR) << "file: " << real_path << " open failed";
    return nullptr;
  }

  ifs.seekg(0, std::ios::end);
  *size = ifs.tellg();
  auto buf = std::make_unique<char[]>(*size);
  if (buf == nullptr) {
    MS_LOG(ERROR) << "malloc buf failed, file: " << real_path;
    ifs.close();
    return nullptr;
  }

  ifs.seekg(0, std::ios::beg);
  ifs.read(buf.get(), *size);
  ifs.close();

  return buf.release();
}

std::string RealPath(const char *path) {
  if (path == nullptr) {
    MS_LOG(ERROR) << "path is nullptr";
    return "";
  }
  if ((strlen(path)) >= PATH_MAX) {
    MS_LOG(ERROR) << "path is too long";
    return "";
  }
  auto resolved_path = std::make_unique<char[]>(PATH_MAX);
  if (resolved_path == nullptr) {
    MS_LOG(ERROR) << "new resolved_path failed";
    return "";
  }
#ifdef _WIN32
  char *real_path = _fullpath(resolved_path.get(), path, PATH_MAX);
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

int CreateOutputDir(std::string *file_path) {
  if (file_path->empty()) {
    MS_LOG(ERROR) << "input file path is empty.";
    return RET_ERROR;
  } else if (file_path->size() >= PATH_MAX) {
    MS_LOG(ERROR) << "input file path is too long";
    return RET_ERROR;
  }

  for (size_t i = 0; i < file_path->size(); i++) {
    if ((*file_path).at(i) == '\\' || (*file_path).at(i) == '/') {
      if (AccessFile(file_path->substr(0, i + 1), F_OK) != 0) {
        int ret = Mkdir(file_path->substr(0, i + 1));
        if (ret != RET_OK) {
          MS_LOG(ERROR) << "mkdir failed. " << file_path->substr(0, i + 1);
          return RET_ERROR;
        }
      }
    }
  }

  if (file_path->back() != '\\' && file_path->back() != '/') {
    if (AccessFile(*file_path, F_OK) != 0) {
      int ret = Mkdir(*file_path);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "mkdir failed. " << file_path;
        return RET_ERROR;
      }
    }
  }

  int count = 0;
  while (AccessFile((*file_path + "/" + std::to_string(count)), F_OK) == 0) {
    MS_LOG(INFO) << "current file_path has existed, file_path cnt plus 1.";  // such as: /xxx/0 ==> /xxx/1
    count++;
    if (count >= MAXIMUM_NUMBERS_OF_FOLDER) {
      MS_LOG(ERROR) << "the number of file folders exceeds the upper limit.";
      return RET_ERROR;
    }
  }
#ifdef _WIN32
  *file_path += "\\" + std::to_string(count);
#else
  *file_path += "/" + std::to_string(count);
#endif
  int ret = Mkdir(*file_path);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "mkdir failed. " << file_path->c_str();
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
