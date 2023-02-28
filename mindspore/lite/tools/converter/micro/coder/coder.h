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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_CODER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_CODER_H_
#include <string>
#include <memory>
#include "tools/converter/micro/coder/session.h"
#include "flatbuffers/flatbuffers.h"
#include "schema/inner/model_generated.h"

namespace mindspore::lite::micro {
constexpr size_t kFlatbuffersBuilderInitSize = 1024;

class Coder final {
 public:
  Coder() = default;

  ~Coder() = default;
  static int MicroSourceCodeGeneration(const schema::MetaGraphT &graph, const std::string &output_path,
                                       const std::string &codegen_mode, const std::string &device,
                                       bool support_parallel, bool debug_mode, bool end_flag);

 private:
  int Init(const std::string &code_mode, const std::string &target, bool support_parallel, bool debug_mode_) const;
  int Run(const void *model_buff, size_t size, const std::string &model_name, bool end_flag);
  bool InitPath(const std::string &output_path);
  std::shared_ptr<CoderSession> session_{nullptr};

  std::string save_path_;
  std::string model_name_;
};
}  // namespace mindspore::lite::micro
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_CODER_H_
