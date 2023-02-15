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

#include "common/util/platform_info.h"
#include "common/util/platform_info_def.h"

namespace fe {
PlatformInfoManager &PlatformInfoManager::Instance() {
  static PlatformInfoManager instance{};
  return instance;
}

uint32_t PlatformInfoManager::InitializePlatformInfo() { return 0; }

uint32_t PlatformInfoManager::Finalize() { return 0; }

uint32_t PlatformInfoManager::GetPlatformInfo(const std::string soc_version, PlatformInfo &platform_info,
                                              OptionalInfo &optional_info) {
  return 0;
}

uint32_t PlatformInfoManager::GetPlatformInfoWithOutSocVersion(PlatformInfo &platform_info,
                                                               OptionalInfo &optional_info) {
  return 0;
}

void PlatformInfoManager::SetOptionalCompilationInfo(OptionalInfo &optional_info) {}

uint32_t PlatformInfoManager::GetPlatformInfos(const std::string soc_version, PlatFormInfos &platform_info,
                                               OptionalInfos &optional_info) {
  return 0;
}

uint32_t PlatformInfoManager::GetPlatformInfoWithOutSocVersion(PlatFormInfos &platform_infos,
                                                               OptionalInfos &optional_infos) {
  return 0;
}

void PlatformInfoManager::SetOptionalCompilationInfo(OptionalInfos &optional_infos) {}

PlatformInfoManager::PlatformInfoManager() {}

PlatformInfoManager::~PlatformInfoManager() {}

void PlatFormInfos::SetCoreNumByCoreType(const std::string &core_type) {}

void OptionalInfos::SetSocVersion(std::string soc_version) {}
}  // namespace fe
