/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/platform.h"
#include <string>

namespace mindspore {
namespace opt {
bool PlatformInfoInitialization(const std::string &soc_version) {
  static bool platform_init = false;
  if (platform_init) {
    return true;
  }
  fe::PlatformInfo platform_info;
  fe::OptionalInfo opti_compilation_info;
  if (GET_PLATFORM.InitializePlatformInfo() != 0) {
    MS_LOG(ERROR) << "Initialize PlatformInfo failed.";
    return false;
  }
  if (GET_PLATFORM.GetPlatformInfo(soc_version, platform_info, opti_compilation_info) != 0) {
    MS_LOG(WARNING) << "GetPlatformInfo failed.";
    return false;
  }
  opti_compilation_info.soc_version = soc_version;
  GET_PLATFORM.SetOptionalCompilationInfo(opti_compilation_info);
  platform_init = true;
  return true;
}
}  // namespace opt
}  // namespace mindspore
