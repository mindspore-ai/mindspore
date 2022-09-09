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
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#endif

#include <cstdlib>
#include "securec/include/securec.h"

#ifdef _MSC_VER
#define PATH_MAX 1024
#define F_OK 0
#define R_OK 4
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

bool IsCharEndWith(const char *src, const char *end) {
  if (strlen(src) > strlen(end)) {
    const char *src_end = src + (strlen(src) - strlen(end));
    if (strcmp(src_end, end) == 0) {
      return true;
    }
  }
  return false;
}

// do not call RealPath function in OpenFile, because OpenFile may open a non-exist file
std::fstream *OpenFile(const std::string &file_path, std::ios_base::openmode open_mode) {
  auto fs = new (std::nothrow) std::fstream();
  if (fs == nullptr) {
    MS_LOG(DEBUG) << "Create file stream failed";
    return nullptr;
  }
  fs->open(file_path, open_mode);
  if (!fs->good()) {
    MS_LOG(DEBUG) << "File is not exist: " << file_path;
    delete fs;
    return nullptr;
  }
  if (!fs->is_open()) {
    MS_LOG(DEBUG) << "Can not open file: " << file_path;
    delete fs;
    return nullptr;
  }
  return fs;
}

// read file in [offset, offset + len)
char *ReadFileSegment(const std::string &file, int64_t offset, int64_t len) {
  if (len <= 0) {
    return nullptr;
  }
  auto len_pos = static_cast<size_t>(len);
  if (offset < 0) {
    MS_LOG(DEBUG) << "offset is invalid, offset: " << offset;
    return nullptr;
  }
  auto offset_pos = static_cast<size_t>(offset);
  std::string real_path = lite::RealPath(file.c_str());
  if (lite::AccessFile(real_path, R_OK) != 0) {
    MS_LOG(DEBUG) << "cannot access file:" << real_path << ".please check file if exists and file mod";
    return nullptr;
  }
  std::ifstream ifs(real_path, std::ifstream::in | std::ifstream::binary);
  if (!ifs.good()) {
    MS_LOG(DEBUG) << "file: " << real_path << " is not exist";
    return nullptr;
  }

  if (!ifs.is_open()) {
    MS_LOG(DEBUG) << "file: " << real_path << " open failed";
    return nullptr;
  }

  ifs.seekg(0, std::ios::end);
  size_t total_size = ifs.tellg();
  if (offset_pos + len_pos > total_size) {
    MS_LOG(ERROR) << "file segment out of range";
    ifs.close();
    return nullptr;
  }

  auto buf = reinterpret_cast<char *>(malloc(len));
  if (buf == nullptr) {
    MS_LOG(ERROR) << "malloc buf failed, file: " << real_path;
    ifs.close();
    return nullptr;
  }

  ifs.seekg(offset, std::ios::beg);
  ifs.read(buf, len);
  ifs.close();

  return buf;
}

char *ReadFile(const char *file, size_t *size) {
  if (file == nullptr) {
    MS_LOG(ERROR) << "File path is nullptr";
    return nullptr;
  }
  MS_ASSERT(size != nullptr);
  std::string real_path = RealPath(file);
  if (real_path.empty()) {
    MS_LOG(DEBUG) << "File path not regular: " << file;
    return nullptr;
  }
#ifndef _MSC_VER
  if (access(real_path.c_str(), F_OK) != 0) {
    MS_LOG(ERROR) << "File is not exist: " << real_path;
    return nullptr;
  }
  if (access(real_path.c_str(), R_OK) != 0) {
    MS_LOG(ERROR) << "File " << real_path << " can't be read. Please change the file permission.";
    return nullptr;
  }
#endif
  auto ifs = OpenFile(real_path, std::ifstream::in | std::ifstream::binary);
  if (ifs == nullptr) {
    MS_LOG(ERROR) << "Open file failed.";
    return nullptr;
  }

  ifs->seekg(0, std::ios::end);
  *size = ifs->tellg();
  auto buf = new (std::nothrow) char[*size];
  if (buf == nullptr) {
    MS_LOG(ERROR) << "malloc buf failed, file: " << file;
    ifs->close();
    delete ifs;
    return nullptr;
  }

  ifs->seekg(0, std::ios::beg);
  (void)ifs->read(buf, static_cast<std::streamsize>(*size));
  ifs->close();
  delete ifs;
  return buf;
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
    MS_LOG(ERROR) << "file path not exists: " << path;
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

std::string GetDirectory(const std::string &path) {
  auto pos = path.find_last_of('/');
  if (pos == std::string::npos) {
    pos = path.find_last_of('\\');
  }
  std::string dir;
  if (pos != std::string::npos) {
    dir = path.substr(0, pos + 1);
  }
  return dir;
}

bool ParserPathAndModelName(const std::string &output_path, std::string *save_path, std::string *model_name) {
  auto pos = output_path.find_last_of('/');
  if (pos == std::string::npos) {
    pos = output_path.find_last_of('\\');
  }
  std::string tmp_model_name;
  if (pos == std::string::npos) {
#ifdef _WIN32
    *save_path = ".\\";
#else
    *save_path = "./";
#endif
    tmp_model_name = output_path;
  } else {
    *save_path = output_path.substr(0, pos + 1);
    tmp_model_name = output_path.substr(pos + 1);
  }
  *save_path = RealPath(save_path->c_str());
  if (save_path->empty()) {
    MS_LOG(DEBUG) << "File path not regular: " << save_path;
    return false;
  }
  auto suffix_pos = tmp_model_name.find_last_of('.');
  if (suffix_pos == std::string::npos) {
    *model_name = tmp_model_name;
  } else {
    if (tmp_model_name.substr(suffix_pos + 1) == "ms") {
      *model_name = tmp_model_name.substr(0, suffix_pos);
    } else {
      *model_name = tmp_model_name;
    }
  }
  return true;
}
}  // namespace lite
}  // namespace mindspore
