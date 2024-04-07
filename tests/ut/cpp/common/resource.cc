/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "resource.h"
#include <mutex>

namespace UT {
std::shared_ptr<UTResourceManager> UTResourceManager::inst_resource_manager_ = nullptr;

std::shared_ptr<UTResourceManager> UTResourceManager::GetInstance() {
  static std::once_flag init_flag_ = {};
  std::call_once(init_flag_, [&]() {
    if (inst_resource_manager_ == nullptr) {
      inst_resource_manager_ = std::make_shared<UTResourceManager>();
    }
  });
  MS_EXCEPTION_IF_NULL(inst_resource_manager_);
  return inst_resource_manager_;
}
}  // namespace UT
