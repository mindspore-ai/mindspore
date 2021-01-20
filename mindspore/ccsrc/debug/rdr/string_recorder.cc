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
#include "debug/rdr/string_recorder.h"
#include <sys/stat.h>
#include <fstream>
#include "debug/common.h"
#include "utils/utils.h"
#include "mindspore/core/utils/log_adapter.h"

namespace mindspore {
void StringRecorder::Export() {
  if (directory_.back() != '/') {
    directory_ += "/";
  }

  if (filename_.empty()) {
    filename_ = module_ + "_" + tag_ + "_" + timestamp_ + ".txt";
  }
  std::string file_path = directory_ + filename_;

  auto realpath = Common::GetRealPath(file_path);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path failed. path=" << file_path;
    return;
  }

  ChangeFileMode(realpath.value(), S_IRWXU);
  std::ofstream fout(realpath.value(), std::ofstream::app);
  if (!fout.is_open()) {
    MS_LOG(WARNING) << "Open file for saving string failed.";
    return;
  }
  fout << data_;
  fout.close();
  // set file mode to read only by user
  ChangeFileMode(realpath.value(), S_IRUSR);
}
}  // namespace mindspore
