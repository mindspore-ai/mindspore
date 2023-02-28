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
#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_CONFIG_PARSER_MICRO_PARAM_PARSER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_CONFIG_PARSER_MICRO_PARAM_PARSER_H_

#include <string>
#include "tools/converter/config_parser/config_file_parser.h"
#include "tools/converter/micro/coder/config.h"
#include "include/errorcode.h"

namespace mindspore {
namespace lite {
class MicroParamParser {
 public:
  STATUS ParseMicroParam(const MicroParamString &micro_param_string, micro::MicroParam *micro_param);

 private:
  STATUS ParseEnableMicro(const std::string &enable_micro, micro::MicroParam *micro_param);
  STATUS ParseTarget(const std::string &target, micro::MicroParam *micro_param);
  STATUS ParseCodeGenMode(const std::string &codegen_mode, micro::MicroParam *micro_param);
  STATUS ParseSupportParallel(const std::string &support_parallel, micro::MicroParam *micro_param);
  STATUS ParseDebugMode(const std::string &debug_mode, micro::MicroParam *micro_param);
  STATUS ParseSavePath(const std::string &debug_mode, micro::MicroParam *micro_param);
  STATUS ParseProjName(const std::string &debug_mode, micro::MicroParam *micro_param);
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_CONFIG_PARSER_MICRO_PARAM_PARSER_H_
