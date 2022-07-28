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

#include "runtime/device/context_extends.h"
#include <string>
#include <memory>
#include <vector>
#include "utils/ms_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace context {
// Register for device type.
struct DeviceTypeSetRegister {
  DeviceTypeSetRegister() {
    MsContext::device_type_seter([](std::shared_ptr<MsContext> &device_type_seter) {
#if defined(ENABLE_D)
      auto enable_ge = mindspore::common::GetEnv("MS_ENABLE_GE");
      if (enable_ge == "1") {
        device_type_seter.reset(new (std::nothrow) MsContext("ge", kAscendDevice));
      } else {
        device_type_seter.reset(new (std::nothrow) MsContext("ms", kAscendDevice));
      }
#elif defined(ENABLE_GPU)
      device_type_seter.reset(new (std::nothrow) MsContext("ms", kGPUDevice));
#else
      device_type_seter.reset(new (std::nothrow) MsContext("vm", kCPUDevice));
#endif
    });
  }
  DeviceTypeSetRegister(const DeviceTypeSetRegister &) = delete;
  DeviceTypeSetRegister &operator=(const DeviceTypeSetRegister &) = delete;
  ~DeviceTypeSetRegister() = default;
} device_type_set_regsiter;
}  // namespace context
}  // namespace mindspore
