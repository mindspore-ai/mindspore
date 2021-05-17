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

#ifndef MINDSPORE_LITE_SRC_INNER_CONTEXT_H
#define MINDSPORE_LITE_SRC_INNER_CONTEXT_H

#include "include/context.h"
#include "src/runtime/runtime_api.h"
#include "src/runtime/allocator.h"
#ifdef ENABLE_ARM
#include "src/cpu_info.h"
#endif
#ifdef SUPPORT_NPU
#include "src/runtime/agent/npu/npu_manager.h"
#endif

namespace mindspore::lite {
struct InnerContext : public Context {
 public:
  struct ThreadPool *thread_pool_ = nullptr;

 public:
  InnerContext() = default;

  explicit InnerContext(const Context *context);
#if SUPPORT_NPU
  InnerContext(const Context *context, NPUManager *npu_manager);
#endif
  int Init();

  bool IsCpuFloat16Enabled() const;

  bool IsGpuFloat16Enabled() const;

  bool IsCpuEnabled() const;

  bool IsGpuEnabled() const;

  bool IsNpuEnabled() const;

  CpuDeviceInfo GetCpuInfo() const;

  GpuDeviceInfo GetGpuInfo() const;

  NpuDeviceInfo GetNpuInfo() const;

  int IsValid() const;

  virtual ~InnerContext();

 private:
  bool IsUserSetCpu() const;

  bool IsUserSetGpu() const;

  bool IsUserSetNpu() const;

  bool IsSupportFloat16() const;

  bool fp16_flag_ = false;

#ifdef ENABLE_ARM
  CpuInfo *cpu_info_ = nullptr;
#endif

#if SUPPORT_NPU
  NPUManager *npu_manager_ = nullptr;
#endif
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_INNER_CONTEXT_H
