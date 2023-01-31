/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef AICPU_CONTEXT_COMMON_THREAD_CTX_H_
#define AICPU_CONTEXT_COMMON_THREAD_CTX_H_

#include <string>

#include "cpu_kernel/inc/cpu_types.h"
#include "aicpu_sharder/aicpu_context.h"

namespace aicpu {
class ThreadCtx {
 public:
  explicit ThreadCtx(DeviceType device) : device_(device) {}

  virtual ~ThreadCtx() = default;

  virtual uint32_t SetThreadCtxInfo(CtxType type, const std::string &key, const std::string &value) const = 0;

  virtual uint32_t GetThreadCtxInfo(CtxType type, const std::string &key, std::string &value) const = 0;

  virtual uint32_t RemoveThreadCtxInfo(CtxType type, const std::string &key) const = 0;

 private:
  ThreadCtx(const ThreadCtx &) = delete;
  ThreadCtx(ThreadCtx &&) = delete;
  ThreadCtx &operator=(const ThreadCtx &) = delete;
  ThreadCtx &operator=(ThreadCtx &&) = delete;

 private:
  DeviceType device_;  // device type, HOST/DEVICE
};
}  // namespace aicpu
#endif  // AICPU_CONTEXT_COMMON_THREAD_CTX_H_
