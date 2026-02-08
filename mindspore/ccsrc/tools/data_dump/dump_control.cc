/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "tools/data_dump/dump_control.h"

#include "tools/data_dump/dump_json_parser.h"
#include "utils/log_adapter.h"

namespace mindspore {

void DumpControl::DynamicDumpStart() {
  MS_VLOG(VL_DUMP) << "Dynamic Dump Start.";
  if (!dynamic_switch_) {
    MS_LOG(WARNING) << "dump_start before dump_set_dynamic-Warning: dump_set_dynamic has not been set!";
  }
  dump_switch_ = true;
}

void DumpControl::DynamicDumpStop() {
  MS_VLOG(VL_DUMP) << "Dynamic Dump Stop.";
  if (!dynamic_switch_) {
    MS_LOG(WARNING) << "dump_stop before dump_set_dynamic-Warning: dump_set_dynamic has not been set!";
  }
  dump_switch_ = false;
}

void DumpControl::SetInitialIteration(std::uint32_t initial_iteration) {
  DumpJsonParser::GetInstance().SetInitialIteration(initial_iteration);
}

void DumpControl::UpdateUserDumpStep(const std::uint32_t step) {
  DumpJsonParser::GetInstance().UpdateUserDumpStep(step);
}
}  // namespace mindspore
