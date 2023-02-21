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

#include "tools/converter/config_parser/cpu_option_param_parser.h"

namespace mindspore {
namespace lite {
STATUS CpuOptionParamParser::ParseCpuOptionCfg(const CpuOptionCfgString &cpu_option_string,
                                               CpuOptionCfg *cpu_option_cfg) {
  if (cpu_option_string.architecture.empty() || cpu_option_string.instruction.empty()) {
    return RET_OK;
  }

  if (cpu_option_string.architecture != "ARM64") {
    MS_LOG(ERROR) << "cpu instruction only supported ARM64. But get " << cpu_option_string.architecture;
    return RET_INPUT_PARAM_INVALID;
  }

  if (cpu_option_string.instruction != "SIMD_DOT") {
    MS_LOG(ERROR) << "cpu instruction only supported SIMD_DOT. But get " << cpu_option_string.instruction;
    return RET_INPUT_PARAM_INVALID;
  }
  cpu_option_cfg->instruction = cpu_option_string.instruction;
  cpu_option_cfg->architecture = cpu_option_string.architecture;
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
