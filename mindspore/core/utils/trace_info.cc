/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "utils/trace_info.h"
#include "utils/info.h"
#include "utils/log_adapter.h"

namespace mindspore {
std::string TraceInfo::GetActionBetweenNode(const DebugInfoPtr &info) const {
  if (info == nullptr) {
    return "";
  }
  std::string act_name = action_name();
  if (debug_info() == nullptr) {
    MS_LOG(EXCEPTION) << "Traced debug info is null";
  }
  if (debug_info() == info) {
    return act_name;
  } else if (debug_info()->trace_info() != nullptr) {
    return act_name + debug_info()->trace_info()->GetActionBetweenNode(info);
  }
  return "Not in the traced info";
}
}  // namespace mindspore
