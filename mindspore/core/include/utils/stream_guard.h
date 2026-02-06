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

#ifndef MINDSPORE_MINDSPORE_CORE_INCLUDE_UTILS_STREAM_GUARD_H_
#define MINDSPORE_MINDSPORE_CORE_INCLUDE_UTILS_STREAM_GUARD_H_

#include <cstdint>
#include "mindapi/base/macros.h"

namespace mindspore {
namespace device {
class DeviceContext;
}  // namespace device

class MS_CORE_API CurrentStream {
 public:
  static uint32_t id();
  static void set_id(uint32_t stream_id);
};

class MS_CORE_API CurrentDeviceContext {
 public:
  static const device::DeviceContext *get();
  static void set(const device::DeviceContext *device_context);
};
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CORE_INCLUDE_UTILS_STREAM_GUARD_H_
