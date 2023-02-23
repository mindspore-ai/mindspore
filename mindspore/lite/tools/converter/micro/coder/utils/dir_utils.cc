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

#include "tools/converter/micro/coder/utils/dir_utils.h"
#include <sys/stat.h>
#if defined(_WIN32) || defined(_WIN64)
#include <direct.h>
#endif
#include <string>
#include <fstream>
#include <array>
#include "include/errorcode.h"
#include "src/common/log_adapter.h"

namespace mindspore::lite::micro {
#if defined(_WIN32) || defined(_WIN64)
#ifdef _MSC_VER
constexpr unsigned int kMicroDirMode = 511;
#else
constexpr _mode_t kMicroDirMode = 0777;
#endif
#else
constexpr __mode_t kMicroDirMode = 0777;
#endif

const int kUnitDirsNum = 3;
static std::array<std::string, kUnitDirsNum> kUnitDirs = {"src", "benchmark", "include"};

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

bool DirectoryGenerator::CreateStaticDir(const std::string &work_dir, const std::string &proj_name) {
  if (!work_dir_.empty()) {
    MS_LOG(INFO) << "Work directory has been created";
    return true;
  }
  if (work_dir.empty() || proj_name.empty() || !DirExists(work_dir)) {
    MS_LOG(ERROR) << "Work directory or project name is empty";
    return false;
  }
  work_dir_ = work_dir;
  project_name_ = proj_name;
#if defined(_WIN32) || defined(_WIN64)
  std::ofstream pro_file;
  std::string read_me_file = work_dir_ + "\\readMe.txt";
  pro_file.open(read_me_file.c_str());
  pro_file << "This is a directory for generating coding files. Do not edit !!!\n";
#else
  std::ifstream pro_file;
  pro_file.open(work_dir_.c_str());
#endif
  if (!pro_file.is_open()) {
    MS_LOG(ERROR) << work_dir_ << ":  root dir not exists or have no access to open, please check it!!!";
    pro_file.close();
    return false;
  }

  std::string slashCh = std::string(kSlash);
  std::string project_dir = work_dir_ + project_name_;
  if (work_dir_.substr(work_dir_.size() - 1, 1) != slashCh) {
    project_dir = work_dir_ + slashCh + project_name_;
  }
  STATUS ret = MkMicroDir(project_dir);
  if (ret == RET_ERROR) {
    pro_file.close();
    return false;
  }

  for (const auto &unit : kUnitDirs) {
    std::string unit_dir = project_dir + slashCh + unit;
    ret = MkMicroDir(unit_dir);
    if (ret == RET_ERROR) {
      pro_file.close();
      return false;
    }
  }
  return true;
}

bool DirectoryGenerator::CreateDynamicDir(const int model_index) {
  if (work_dir_.empty() || project_name_.empty()) {
    MS_LOG(ERROR) << "Work directory or project name is empty";
    return false;
  }
  std::string current_work_dir = work_dir_ + project_name_ + std::string(kSlash) + kUnitDirs[0];
#if defined(_WIN32) || defined(_WIN64)
  std::ofstream pro_file;
  std::string read_me_file = current_work_dir + "\\readMe.txt";
  pro_file.open(read_me_file.c_str());
  pro_file << "This is a directory for generating coding files. Do not edit !!!\n";
#else
  std::ifstream pro_file;
  pro_file.open(current_work_dir.c_str());
#endif
  if (!pro_file.is_open()) {
    MS_LOG(ERROR) << current_work_dir << ": model's upper dir not exists or have no access to open, please check it!!!";
    pro_file.close();
    return false;
  }

  std::string model_dir = current_work_dir + std::string(kSlash) + "model" + std::to_string(model_index);
  auto ret = MkMicroDir(model_dir);
  if (ret == RET_ERROR) {
    pro_file.close();
    return ret;
  }
  pro_file.close();
  return true;
}
}  // namespace mindspore::lite::micro
