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
#include "plugin/device/ascend/hal/common/platform_info_util.h"

#include <string>

namespace mindspore::device::ascend {
const auto kDefaultCoreType = "AiCore";
PlatformInfoUtil::~PlatformInfoUtil() { fe::PlatformInfoManager::Instance().Finalize(); }

PlatformInfoUtil &PlatformInfoUtil::GetInstance() {
  static PlatformInfoUtil inst;
  return inst;
}

bool PlatformInfoUtil::Init(const std::string &soc_version) {
  auto &platform_inst = fe::PlatformInfoManager::Instance();
  if (platform_inst.InitializePlatformInfo() != 0) {
    MS_LOG(ERROR) << "Initialize PlatformInfo failed.";
    return false;
  }
  if (platform_inst.GetPlatformInfo(soc_version, platform_info_, opti_compilation_info_) != 0) {
    MS_LOG(ERROR) << "GetPlatformInfo failed.";
    return false;
  }
  opti_compilation_info_.soc_version = soc_version;
  platform_inst.SetOptionalCompilationInfo(opti_compilation_info_);
  if (platform_inst.GetPlatformInfos(soc_version, platform_infos_, opti_compilation_infos_) != 0) {
    MS_LOG(ERROR) << "GetPlatformInfos failed.";
    return false;
  }
  platform_infos_.SetCoreNumByCoreType(kDefaultCoreType);
  opti_compilation_infos_.SetSocVersion(soc_version);
  platform_inst.SetOptionalCompilationInfo(opti_compilation_infos_);
  soc_version_ = soc_version;
  return true;
}

std::string PlatformInfoUtil::soc_version() const { return soc_version_; }

fe::PlatFormInfos PlatformInfoUtil::platform_infos() const { return platform_infos_; }

bool PlatformInfoUtil::IsCubeVectorSplit() const { return platform_info_.ai_core_spec.cube_vector_split == 1; }

bool PlatformInfoUtil::IsCoreNum32() const { return platform_info_.soc_info.ai_core_cnt == 32; }
}  // namespace mindspore::device::ascend
