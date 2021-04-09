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
#include "debug/rdr/base_recorder.h"
#include <sys/stat.h>
#include <fstream>
#include "debug/common.h"
#include "utils/utils.h"

namespace mindspore {
void BaseRecorder::SetDirectory(const std::string &directory) {
  std::string error_message = module_ + ":" + name_ + " set directory failed.";
  if (Common::IsPathValid(directory, MAX_DIRECTORY_LENGTH, error_message)) {
    directory_ = directory;
    if (directory_.back() != '/') {
      directory_ += "/";
    }
  }
}

void BaseRecorder::SetFilename(const std::string &filename) {
  std::string error_message = module_ + ":" + name_ + " set filename failed.";
  if (Common::IsFilenameValid(filename, MAX_DIRECTORY_LENGTH, error_message)) {
    filename_ = filename;
  }
}

std::optional<std::string> BaseRecorder::GetFileRealPath(const std::string &suffix) {
  std::string filename;
  if (filename_.empty()) {
    filename = module_ + delimiter_ + name_;
    if (!suffix.empty()) {
      filename += delimiter_ + suffix;
    }
    filename += delimiter_ + timestamp_;
  } else {
    filename = filename_;
    if (!suffix.empty()) {
      filename = filename_ + delimiter_ + suffix;
    }
  }
  std::string file_path = directory_ + filename;
  auto realpath = Common::GetRealPath(file_path);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path failed. "
                  << "Info: module=" << module_ << ", name=" << name_ << ", "
                  << "path=" << file_path << ".";
  }

  return realpath;
}
}  // namespace mindspore
