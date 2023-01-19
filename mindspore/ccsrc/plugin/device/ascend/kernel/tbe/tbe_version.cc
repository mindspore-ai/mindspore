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

#include "plugin/device/ascend/kernel/tbe/tbe_version.h"

namespace mindspore {
namespace kernel {
namespace tbe {
std::string GetPyTeVersion() {
  auto cmd_env = GetPyExe();
  std::string cmd = cmd_env;
  auto cmd_script =
    "-c "
    "\""
    "from te import version;"
    "print('[~]' + version.version)"
    "\"";
  (void)cmd.append(1, ' ').append(cmd_script);
  auto result = GetCmdResult(cmd);
  if (result.empty() || strlen(kTag) > result.size()) {
    MS_LOG(EXCEPTION) << "result size seems incorrect, result(" << result.size() << "): {" << result << "}";
  }
  result = result.substr(strlen(kTag));
  return result;
}
}  // namespace tbe
}  // namespace kernel
}  // namespace mindspore
