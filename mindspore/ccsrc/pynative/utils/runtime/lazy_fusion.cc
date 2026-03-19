/**
 * Copyright 2024-2026 Huawei Technologies Co., Ltd
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

#include "include/pynative/utils/runtime/lazy_fusion.h"
#include "utils/ms_context.h"
#include "utils/log_adapter.h"

namespace mindspore {
LazyFusionFactory g_lazy_fusion;

void LazyFusionFactory::Init() {
  auto iter = init_funcs_.find(kAscendDevice);
  if (iter != init_funcs_.end()) {
    MS_LOG(INFO) << "Start init lazy fusion.";
    auto func = iter->second;
    MS_EXCEPTION_IF_NULL(func);
    if (func()) {
      enabled_devices_.insert(kAscendDevice);
      MS_LOG(INFO) << "Enable lazy fusion for device " << kAscendDevice << ".";
    }
    MS_LOG(INFO) << "End init lazy fusion.";
  } else {
    MS_LOG(INFO) << "lazy fusion initialize function not registered.";
  }
}

void LazyFusionFactory::Flush() {
  static const auto func = []() -> LazyFusionFlushFunc {
    auto iter = g_lazy_fusion.flush_funcs_.find(kAscendDevice);
    if (iter == g_lazy_fusion.flush_funcs_.end()) {
      return nullptr;
    }
    return iter->second;
  }();
  if (func == nullptr) {
    return;
  }
  func();
}
}  // namespace mindspore
