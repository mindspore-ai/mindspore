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

#include "utils/temp_file_manager.h"

#include <exception>
#include "utils/log_adapter.h"
#include "utils/system/env.h"

namespace mindspore {
TempFileManager &TempFileManager::GetInstance() {
  static TempFileManager instance{};
  return instance;
}

TempFileManager::~TempFileManager() {
  try {
    CleanTempFiles();
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Exception occurred when cleaning temp files. Error info " << e.what();
  } catch (...) {
    MS_LOG(ERROR) << "Exception occurred when cleaning temp files.";
  }
}

void TempFileManager::Register(const std::string &file_path) { (void)temp_file_paths_.insert(file_path); }

void TempFileManager::UnRegister(const std::string &file_path) { (void)temp_file_paths_.erase(file_path); }

void TempFileManager::RemoveFile(const std::string &file_path) const {
  auto fs = system::Env::GetFileSystem();
  MS_EXCEPTION_IF_NULL(fs);
  if (file_path.empty()) {
    return;
  }
  if (fs->FileExist(file_path)) {
    if (!fs->DeleteFile(file_path)) {
      MS_LOG(WARNING) << "Delete tensor file path: " << file_path << " failed!";
    }
  } else {
    MS_LOG(WARNING) << "Invalid tensor file path: " << file_path;
  }
}

void TempFileManager::CleanTempFiles() {
  MS_LOG(INFO) << "Start delete temp files.";
  for (auto file_path : temp_file_paths_) {
    RemoveFile(file_path);
  }
  MS_LOG(INFO) << "End delete temp files.";
}
}  // namespace mindspore
