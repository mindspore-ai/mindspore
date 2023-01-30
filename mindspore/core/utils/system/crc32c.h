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

#ifndef MINDSPORE_CORE_UTILS_SYSTEM_CRC32C_H_
#define MINDSPORE_CORE_UTILS_SYSTEM_CRC32C_H_

#include <cstddef>
#include <cstdint>
#include "utils/system/base.h"
#include "utils/system/env.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace system {
// Align n to (1 << m) byte boundary
inline uintptr_t MemAlign(uintptr_t n, size_t m) { return ((n) + ((1 << (m)) - 1)) & (~((1 << (m)) - 1)); }

// Masked for crc.
static constexpr uint32 kMaskDelta = 0xa282ead8ul;
static const int kRightShift = 15;
static const int kLeftShift = (32 - kRightShift);
// Provide the Crc32c function
class MS_CORE_API Crc32c {
 public:
  Crc32c() = default;
  ~Crc32c() = default;

  // Calculate the crc32c value, use the 8 table method
  static uint32 MakeCrc32c(uint32 init_crc, const char *data, size_t size);

  // return the crc32c value(need mask)
  static uint32 GetMaskCrc32cValue(const char *data, size_t n) {
    auto crc = MakeCrc32c(0, data, n);
    // Rotate right by kRightShift bits and add kMaskDelta(a constant).
    return ((crc >> kRightShift) | (crc << kLeftShift)) + kMaskDelta;
  }
};
}  // namespace system
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_SYSTEM_CRC32C_H_
