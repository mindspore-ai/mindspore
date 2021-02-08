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
  std::string error_message = module_ + ":" + tag_ + " set directory failed.";
  if (Common::IsPathValid(directory, maxDirectoryLength, error_message)) {
    directory_ = directory;
    if (directory_.back() != '/') {
      directory_ += "/";
    }
  }
}

void BaseRecorder::SetFilename(const std::string &filename) {
  std::string error_message = module_ + ":" + tag_ + " set filename failed.";
  if (Common::IsFilenameValid(filename, maxDirectoryLength, error_message)) {
    filename_ = filename;
  }
}

std::optional<std::string> BaseRecorder::GetFileRealPath() {
  if (filename_.empty()) {
    filename_ = module_ + "_" + tag_ + "_" + timestamp_;
  }
  std::string file_path = directory_ + filename_;
  auto realpath = Common::GetRealPath(file_path);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path failed. "
                  << "Info: module=" << module_ << ", tag=" << tag_ << ", "
                  << "path=" << file_path << ".";
  }

  return realpath;
}
}  // namespace mindspore
