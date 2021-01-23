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
#ifndef MINDSPORE_CCSRC_DEBUG_RDR_BASE_RECORDER_H_
#define MINDSPORE_CCSRC_DEBUG_RDR_BASE_RECORDER_H_

#include <memory>
#include <string>
#include "debug/env_config_parser.h"

namespace mindspore {
class BaseRecorder {
 public:
  BaseRecorder() : module_(""), tag_(""), directory_(""), filename_(""), timestamp_("") {}
  BaseRecorder(const std::string &module, const std::string &tag)
      : module_(module), tag_(tag), directory_(""), filename_(""), timestamp_("") {
    auto &config_parser_ptr = mindspore::EnvConfigParser::GetInstance();
    config_parser_ptr.Parse();
    directory_ = config_parser_ptr.rdr_path();
  }
  ~BaseRecorder() {}

  std::string GetModule() { return module_; }
  std::string GetTag() { return tag_; }

  void SetDirectory(const std::string &directory) { directory_ = directory; }

  virtual void Export() {}

 protected:
  std::string module_;
  std::string tag_;
  std::string directory_;
  std::string filename_;
  std::string timestamp_;  // year,month,day,hour,minute,second
};

using BaseRecorderPtr = std::shared_ptr<BaseRecorder>;
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEBUG_RDR_BASE_RECORDER_H_
