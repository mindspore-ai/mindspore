/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_INCLUDE_API_LITE_CONTEXT_H
#define MINDSPORE_INCLUDE_API_LITE_CONTEXT_H

#include <string>
#include <memory>
#include <map>
#include <any>
#include "include/api/types.h"

namespace mindspore {
namespace lite {
/// \brief CpuBindMode defined for holding bind cpu strategy argument.
typedef enum : uint32_t {
  NO_BIND = 0,    /**< no bind */
  HIGHER_CPU = 1, /**< bind higher cpu first */
  MID_CPU = 2     /**< bind middle cpu first */
} CpuBindMode;

class Allocator;
}  // namespace lite

struct MS_API Context {
 public:
  static void Clear(const std::shared_ptr<Context> &contxet);

  static void SetAsDefault(const std::shared_ptr<Context> &contxet);

  static void SetVendorName(const std::shared_ptr<Context> &contxet, const std::string &name);
  static std::string GetVendorName(const std::shared_ptr<Context> &contxet);

  static void SetThreadNum(const std::shared_ptr<Context> &contxet, int num);
  static int GetThreadNum(const std::shared_ptr<Context> &contxet);

  static void SetAllocator(const std::shared_ptr<Context> &contxet, std::shared_ptr<lite::Allocator> alloc);
  static std::shared_ptr<lite::Allocator> GetAllocator(const std::shared_ptr<Context> &contxet);

  static void ConfigCPU(const std::shared_ptr<Context> &contxet, bool config);
  static bool IfCPUEnabled(const std::shared_ptr<Context> &contxet);

  static void ConfigCPUFp16(const std::shared_ptr<Context> &contxet, bool config);
  static bool IfCPUFp16Enabled(const std::shared_ptr<Context> &contxet);

  static void SetCPUBindMode(const std::shared_ptr<Context> &contxet, lite::CpuBindMode mode);
  static lite::CpuBindMode GetCPUBindMode(const std::shared_ptr<Context> &contxet);

  static void ConfigGPU(const std::shared_ptr<Context> &contxet, bool config);
  static bool IfGPUEnabled(const std::shared_ptr<Context> &contxet);

  static void ConfigGPUFp16(const std::shared_ptr<Context> &contxet, bool config);
  static bool IfGPUFp16Enabled(const std::shared_ptr<Context> &contxet);

  static void ConfigNPU(const std::shared_ptr<Context> &contxet, bool config);
  static bool IfNPUEnabled(const std::shared_ptr<Context> &contxet);

  static void SetNPUFrequency(const std::shared_ptr<Context> &contxet, int freq);
  static int GetNPUFrequency(const std::shared_ptr<Context> &contxet);

 private:
  std::map<std::string, std::any> context_;
};
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_LITE_CONTEXT_H
