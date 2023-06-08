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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PLATFORM_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PLATFORM_H_
#include <string>

#include "common/util/platform_info.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace opt {
#define GET_PLATFORM fe::PlatformInfoManager::Instance()

#define MS_CHECK_CUBE_VECTOR_SPLIT()                                                           \
  do {                                                                                         \
    fe::PlatformInfo platform_info;                                                            \
    fe::OptionalInfo opti_compilation_info;                                                    \
    (void)GET_PLATFORM.GetPlatformInfoWithOutSocVersion(platform_info, opti_compilation_info); \
    if (platform_info.ai_core_spec.cube_vector_split == 1) {                                   \
      MS_LOG(INFO) << name() << "not support cube vector split.";                              \
      return;                                                                                  \
    }                                                                                          \
  } while (0)

#define MS_CHECK_CUBE_VECTOR_NOT_SPLIT()                                                       \
  do {                                                                                         \
    fe::PlatformInfo platform_info;                                                            \
    fe::OptionalInfo opti_compilation_info;                                                    \
    (void)GET_PLATFORM.GetPlatformInfoWithOutSocVersion(platform_info, opti_compilation_info); \
    if (platform_info.ai_core_spec.cube_vector_split != 1) {                                   \
      MS_LOG(INFO) << name() << "just support cube vector split.";                             \
      return nullptr;                                                                          \
    }                                                                                          \
  } while (0)

#define MS_CHECK_CORE_CNT_32()                                                                 \
  do {                                                                                         \
    fe::PlatformInfo platform_info;                                                            \
    fe::OptionalInfo opti_compilation_info;                                                    \
    (void)GET_PLATFORM.GetPlatformInfoWithOutSocVersion(platform_info, opti_compilation_info); \
    if (platform_info.soc_info.ai_core_cnt != 32) {                                            \
      MS_LOG(INFO) << name() << "not support (ai_core_cnt != 32).";                            \
      return nullptr;                                                                          \
    }                                                                                          \
  } while (0)

bool PlatformInfoInitialization(const std::string &soc_version);
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PLATFORM_H_
