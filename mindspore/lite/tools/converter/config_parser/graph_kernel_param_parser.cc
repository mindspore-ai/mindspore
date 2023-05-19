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
#include "tools/converter/config_parser/graph_kernel_param_parser.h"
#include <fstream>
namespace mindspore {
namespace lite {
STATUS GraphKernelParamParser::ParseGraphKernelCfg(const GraphKernelString &graph_kernel_string,
                                                   GraphKernelCfg *graph_kernel_cfg) {
  if (graph_kernel_string.empty()) {
    return RET_OK;
  }
  std::stringstream oss;
  for (auto &item : graph_kernel_string) {
    oss << item << " ";
  }
  graph_kernel_cfg->graph_kernel_flags = oss.str();
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
