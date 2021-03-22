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

#include "coder/utils/dir_utils.h"
#include <sys/stat.h>
#if defined(_WIN32) || defined(_WIN64)
#include <direct.h>
#endif
#include <string>
#include <fstream>
#include "include/errorcode.h"
#include "src/common/log_adapter.h"

namespace mindspore::lite::micro {

#if defined(_WIN32) || defined(_WIN64)
constexpr _mode_t kMicroDirMode = 0777;
#else
constexpr __mode_t kMicroDirMode = 0777;
#endif

static std::array<std::string, 2> kWorkDirs = {"src", "benchmark"};

bool DirExists(const std::string &dir_path) {
  struct stat file_info;
  if (stat(dir_path.c_str(), &file_info) != 0) {
    return false;
  }
  return (file_info.st_mode & S_IFDIR) != 0;
}

bool FileExists(const std::string &path) {
  std::ifstream file(path);
  return file.good();
}

static int MkMicroDir(const std::string &currentDir) {
#if defined(_WIN32) || defined(_WIN64)
  std::ofstream currentFile;
  std::string readMeFile = currentDir + "\\readMe.txt";
  currentFile.open(readMeFile);
  currentFile << "This is a directory for generating coding files. Do not edit !!!\n";
  if (!currentFile.is_open()) {
    if (_mkdir(currentDir.c_str()) != 0) {
      MS_LOG(ERROR) << currentDir << ": mkdir failed, please check filePath!!!";
      currentFile.close();
      return RET_ERROR;
    }
  }
  currentFile.close();
#else
  std::ifstream currentFile;
  currentFile.open(currentDir);
  if (!currentFile.is_open()) {
    if (mkdir(currentDir.c_str(), kMicroDirMode) != 0) {
      MS_LOG(ERROR) << currentDir << ": mkdir failed, please check filePath!!!";
      currentFile.close();
      return RET_ERROR;
    }
  }
  currentFile.close();
#endif
  return RET_OK;
}

int InitProjDirs(const std::string &project_root_dir, const std::string &proj_name) {
#if defined(_WIN32) || defined(_WIN64)
  std::ofstream pro_file;
  std::string read_me_file = project_root_dir + "\\readMe.txt";
  pro_file.open(read_me_file.c_str());
  pro_file << "This is a directory for generating coding files. Do not edit !!!\n";
#else
  std::ifstream pro_file;
  pro_file.open(project_root_dir.c_str());
#endif
  if (!pro_file.is_open()) {
    MS_LOG(ERROR) << project_root_dir << ":  model's root dir not exists or have no access to open, please check it!!!";
    pro_file.close();
    return RET_ERROR;
  }
  // check other dirs && make them if not exists
  // 1. coderDir 2.WorkRootDir 3. WorkChildDir
  std::string current_dir;
  std::string slashCh = std::string(kSlash);
  if (project_root_dir.back() != slashCh.back()) {
    current_dir = project_root_dir + slashCh;
  }
  current_dir += proj_name;
  std::string work_dir = current_dir;
  STATUS ret = MkMicroDir(current_dir);
  if (ret == RET_ERROR) {
    pro_file.close();
    return ret;
  }

  for (const auto &work : kWorkDirs) {
    current_dir = work_dir + slashCh + work;
    ret = MkMicroDir(current_dir);
    if (ret == RET_ERROR) {
      pro_file.close();
      return ret;
    }
  }
  pro_file.close();
  return RET_OK;
}
}  // namespace mindspore::lite::micro
