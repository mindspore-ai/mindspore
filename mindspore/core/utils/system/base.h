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

#ifndef MINDSPORE_CORE_UTILS_SYSTEM_BASE_H_
#define MINDSPORE_CORE_UTILS_SYSTEM_BASE_H_

#include <string>
#include <memory>
#include "securec/include/securec.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace system {
using string = std::string;

using int8 = int8_t;
using int16 = int16_t;
using int32 = int32_t;
using int64 = int64_t;

using uint8 = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;
using uint64 = uint64_t;

// Use the macro to confirm the system env
#if defined(ANDROID) || defined(__ANDROID__)

#define SYSTEM_ENV_POSIX_ANDROID

#elif defined(__APPLE__)

#define SYSTEM_ENV_POSIX

#elif defined(_WIN32) || defined(_WIN64)

#define SYSTEM_ENV_WINDOWS

#elif defined(__arm__)

#define SYSTEM_ENV_POSIX

#else  // default set the POSIX

#define SYSTEM_ENV_POSIX

#endif

// define the platform
enum PlatformDefine {
  kPlatformPosix = 0,     // Posix platform
  kPlatformPosixAndroid,  // Android of posix
  kPlatformWindows,       // Windows system
  kPlatformUnknow = 0xFF  // Error
};

class Platform {
 public:
  Platform() {
    platform_ = kPlatformUnknow;
#if defined(SYSTEM_ENV_POSIX)
    platform_ = kPlatformPosix;
#elif defined(SYSTEM_ENV_POSIX_ANDROID)
    platform_ = kPlatformPosixAndroid;
#elif defined(SYSTEM_ENV_WINDOWS)
    platform_ = kPlatformWindows;
#endif
  }

  ~Platform() = default;

  static PlatformDefine get_platform() {
    static const auto sys_env = std::make_shared<Platform>();
    return sys_env->platform_;
  }

 private:
  PlatformDefine platform_;
};

// define the big or little endian type
constexpr bool kLittleEndian = true;

// implement common define function
// Get the 32 bits align value
inline uint32 DecodeFixed32(const char *ptr) {
  uint32 result;
  if (EOK != memcpy_s(&result, sizeof(result), ptr, sizeof(result))) {
    MS_LOG(EXCEPTION) << "Call DecodeFixed32 memcpy value failure.";
  }
  return result;
}
// Used to fetch a naturally-aligned 32-bit word in little endian byte-order
inline uint32 LE_LOAD32(const uint8_t *p) { return DecodeFixed32(reinterpret_cast<const char *>(p)); }
// Encode the data to buffer
inline void EncodeFixed32(char *buf, uint32 value) {
  if (EOK != memcpy_s(buf, sizeof(value), &value, sizeof(value))) {
    MS_LOG(EXCEPTION) << "Call EncodeFixed32 memcpy value failure.";
  }
}
inline void EncodeFixed64(char *buf, const unsigned int array_len, int64 value) {
  if (sizeof(value) > array_len) {
    MS_LOG(EXCEPTION) << "Buffer overflow, real size is " << array_len << ", but required " << sizeof(value) << ".";
  }
  if (EOK != memcpy_s(buf, sizeof(value), &value, sizeof(value))) {
    MS_LOG(EXCEPTION) << "Call EncodeFixed64 memcpy value failure.";
  }
}
}  // namespace system
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_SYSTEM_BASE_H_
