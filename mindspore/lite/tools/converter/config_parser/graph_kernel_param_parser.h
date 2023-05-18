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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_CONFIG_PARSER_GRAPH_KERNEL_PARAM_PARSER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_CONFIG_PARSER_GRAPH_KERNEL_PARAM_PARSER_H
#include <string>
#include "tools/converter/cxx_api/converter_para.h"
#include "tools/converter/config_parser/config_file_parser.h"
#include "include/errorcode.h"

namespace mindspore {
namespace lite {
class GraphKernelParamParser {
 public:
  STATUS ParseGraphKernelCfg(const GraphKernelString &graph_kernel_string, GraphKernelCfg *graph_kernel_cfg);
};
}  // namespace lite
}  // namespace mindspore
#endif
