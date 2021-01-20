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
#ifndef MINDSPORE_CCSRC_DEBUG_RDR_STRING_RECORDER_H_
#define MINDSPORE_CCSRC_DEBUG_RDR_STRING_RECORDER_H_

#include <string>
#include <memory>

#include "debug/rdr/base_recorder.h"
namespace mindspore {
class StringRecorder : public BaseRecorder {
 public:
  StringRecorder() : BaseRecorder() {}
  StringRecorder(const std::string &module, const std::string &tag, const std::string &data,
                 const std::string &file_type)
      : BaseRecorder(module, tag), data_(data) {}
  ~StringRecorder() {}
  void SetModule(const std::string &module) { module_ = module; }
  void SetFilename(const std::string &filename) { filename_ = filename; }
  virtual void Export();

 private:
  std::string data_;
};
using StringRecorderPtr = std::shared_ptr<StringRecorder>;
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEBUG_RDR_STRING_RECORDER_H_
