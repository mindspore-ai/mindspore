/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "include/runtime/memory/mem_pool/mem_env.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace memory {
namespace mem_pool {
std::string GetAllocConfigValue(const std::string &alloc_config) {
  const auto &value = common::GetConfigValue(kAllocConf, alloc_config);
  return value;
}

bool IsEnableAllocConfig(const std::string &alloc_config) {
  const auto &value = GetAllocConfigValue(alloc_config);
  return ((value == "True") || (value == "true"));
}

bool IsDisableAllocConfig(const std::string &alloc_config) {
  const auto &value = GetAllocConfigValue(alloc_config);
  return ((value == "False") || (value == "false"));
}
}  // namespace mem_pool
}  // namespace memory
}  // namespace mindspore
