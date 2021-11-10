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

#ifndef MINDSPORE_CORE_UTILS_SYSTEM_ENV_H_
#define MINDSPORE_CORE_UTILS_SYSTEM_ENV_H_

#include <memory>
#include "utils/system/base.h"
#include "utils/log_adapter.h"
#include "utils/system/file_system.h"

namespace mindspore {
namespace system {
// Confirm the system env and create the filesystem, time...
class Env {
 public:
  Env() { platform_ = Platform::get_platform(); }
  virtual ~Env() = default;

  static std::shared_ptr<FileSystem> GetFileSystem() {
#if defined(SYSTEM_ENV_POSIX)
    auto fs = std::make_shared<PosixFileSystem>();
    return fs;
#elif defined(SYSTEM_ENV_WINDOWS)
    auto fs = std::make_shared<WinFileSystem>();
    return fs;
#else
    MS_LOG(EXCEPTION) << "Now not support the platform.";
#endif
  }

 private:
  PlatformDefine platform_;
};
}  // namespace system
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_SYSTEM_ENV_H_
