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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_PLATFORM_INFO_UTILS_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_PLATFORM_INFO_UTILS_H_
#include <string>

#include "common/util/platform_info.h"
#include "utils/log_adapter.h"

namespace mindspore::device::ascend {
class PlatformInfoUtil {
 public:
  static PlatformInfoUtil &GetInstance();
  bool Init(const std::string &soc_version);
  std::string soc_version() const;
  fe::PlatFormInfos platform_infos() const;
  bool IsCubeVectorSplit() const;
  bool IsCoreNum32() const;

 private:
  PlatformInfoUtil() = default;
  ~PlatformInfoUtil();
  std::string soc_version_;
  fe::PlatformInfo platform_info_;
  fe::OptionalInfo opti_compilation_info_;
  fe::PlatFormInfos platform_infos_;
  fe::OptionalInfos opti_compilation_infos_;
};

#define GET_PLATFORM device::ascend::PlatformInfoUtil::GetInstance()

#define MS_CHECK_CUBE_VECTOR_SPLIT()                              \
  do {                                                            \
    if (GET_PLATFORM.IsCubeVectorSplit()) {                       \
      MS_LOG(INFO) << name() << "not support cube vector split."; \
      return;                                                     \
    }                                                             \
  } while (0)

#define MS_CHECK_CUBE_VECTOR_NOT_SPLIT()                           \
  do {                                                             \
    if (!GET_PLATFORM.IsCubeVectorSplit()) {                       \
      MS_LOG(INFO) << name() << "just support cube vector split."; \
      return nullptr;                                              \
    }                                                              \
  } while (0)

#define MS_CHECK_CORE_CNT_32()                                      \
  do {                                                              \
    if (!GET_PLATFORM.IsCoreNum32()) {                              \
      MS_LOG(INFO) << name() << "not support (ai_core_cnt != 32)."; \
      return nullptr;                                               \
    }                                                               \
  } while (0)
}  // namespace mindspore::device::ascend
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_PLATFORM_INFO_UTILS_H_
