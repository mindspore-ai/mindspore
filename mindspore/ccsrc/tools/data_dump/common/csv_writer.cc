/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "tools/data_dump/common/csv_writer.h"

#include <map>
#include <memory>
#include <string>

#include "include/utils/common.h"
#include "utils/file_utils.h"
#include "utils/system/env.h"
#include "utils/system/file_system.h"

namespace mindspore {

CsvFileMutexManager &CsvFileMutexManager::GetInstance() {
  static CsvFileMutexManager instance;
  return instance;
}

std::mutex &CsvFileMutexManager::GetCsvMutex(const std::string &file_name) {
  std::lock_guard<std::mutex> lock(manager_mutex);
  if (csv_mutex_map.find(file_name) == csv_mutex_map.end()) {
    csv_mutex_map[file_name];
  }
  return csv_mutex_map[file_name];
}

CsvWriter &CsvWriter::GetInstance() {
  static CsvWriter instance = CsvWriter();
  return instance;
}

bool CsvWriter::OpenFile(const std::string &path, const std::string &header, bool trunc) {
  std::shared_lock<std::shared_mutex> lock(write_mutex_);

  if (file_.is_open() && path == file_path_str_) {
    return true;
  }
  if (file_.is_open()) {
    CloseFile();
  }
  auto file_path = Common::CreatePrefixPath(path);
  if (!file_path.has_value()) {
    MS_LOG(WARNING) << "CreatePrefixPath failed, skipping current statistics";
    return false;
  }
  // try to open file
  std::string file_path_value = file_path.value();
  std::shared_ptr<system::FileSystem> fs = system::Env::GetFileSystem();
  MS_EXCEPTION_IF_NULL(fs);
  bool first_time_opening = !fs->FileExist(file_path_value);
  ChangeFileMode(file_path_value, S_IWUSR | S_IRUSR);
  if (first_time_opening || trunc) {
    // remove any possible output from previous runs
    file_.open(file_path_value, std::ios::out | std::ios::trunc | std::ios::binary);
  } else {
    file_.open(file_path_value, std::ios::out | std::ios::app | std::ios::binary);
  }
  if (!file_.is_open()) {
    MS_LOG(WARNING) << "Open file " << file_path_value << " failed." << ErrnoToString(errno);
    return false;
  }
  if (first_time_opening || trunc) {
    file_ << header;
    (void)file_.flush();
    file_path_str_ = path;
  }
  MS_LOG(INFO) << "Opened file: " << file_path_value;
  return true;
}

void CsvWriter::CloseFile() noexcept {
  if (file_.is_open()) {
    file_.close();
    ChangeFileMode(file_path_str_, S_IRUSR);
    MS_LOG(INFO) << "Closed statistics dump file: " << file_path_str_;
  }
}

CsvWriter::~CsvWriter() { CloseFile(); }
}  // namespace mindspore
