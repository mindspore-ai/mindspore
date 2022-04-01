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
#include "include/common/debug/rdr/base_recorder.h"
#include "include/common/debug/common.h"
#include "include/common/utils/comm_manager.h"
#include "include/common/debug/env_config_parser.h"

namespace mindspore {
namespace {
constexpr int kMaxNameLength = 64;
}  // namespace
BaseRecorder::BaseRecorder() : module_(""), name_(""), directory_(""), filename_(""), timestamp_("") {}
BaseRecorder::BaseRecorder(const std::string &module, const std::string &name)
    : module_(module), name_(name), filename_("") {
  directory_ = mindspore::EnvConfigParser::GetInstance().RdrPath();

  if (name.length() > kMaxNameLength) {
    name_ = name.substr(0, kMaxNameLength);
    MS_LOG(WARNING) << "The name length is " << name.length() << ", exceeding the limit " << kMaxNameLength
                    << ". It will be intercepted as '" << name_ << "'.";
  }

  std::string err_msg = module_ + ":" + name_ + " set filename failed.";
  if (!filename_.empty() && !Common::IsFilenameValid(filename_, MAX_FILENAME_LENGTH, err_msg)) {
    filename_ = "";
  }
  auto sys_time = GetTimeString();
  for (auto ch : sys_time) {
    if (ch == '.') {
      break;
    }
    if (ch != '-' && ch != ':') {
      timestamp_.push_back(ch);
    }
  }
}

std::optional<std::string> BaseRecorder::GetFileRealPath(const std::string &suffix) const {
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
  std::string file_path = directory_ + "rank_" + std::to_string(GetRank()) + "/rdr/" + filename;
  auto realpath = Common::CreatePrefixPath(file_path);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path failed. "
                  << "Info: module=" << module_ << ", name=" << name_ << ", "
                  << "path=" << file_path << ".";
  }

  return realpath;
}
}  // namespace mindspore
