/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_AKG_KERNEL_BUILD_MANAGER_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_AKG_KERNEL_BUILD_MANAGER_H_
#include "kernel/akg/akg_kernel_build.h"
#include <map>
#include <utility>
#include <memory>
#include <string>
#include "include/backend/visible.h"

namespace mindspore {
namespace kernel {
using AkgKernelBuildCreator = std::function<std::shared_ptr<AkgKernelBuilder>()>;

class BACKEND_EXPORT AkgKernelBuildManager {
 public:
  static AkgKernelBuildManager &Instance();
  void Register(const std::string &device_type, AkgKernelBuildCreator &&creator);
  void Clear() { base_map_.clear(); }
  std::shared_ptr<AkgKernelBuilder> GetAkgKernelBuilder(const std::string &device_type);

 private:
  std::map<std::string, AkgKernelBuildCreator> base_map_;
};

class AkgKernelBuildRegister {
 public:
  AkgKernelBuildRegister(const std::string &device_type, AkgKernelBuildCreator &&creator) {
    AkgKernelBuildManager::Instance().Register(device_type, std::move(creator));
  }
  ~AkgKernelBuildRegister() = default;
};

#define REG_AKG_KERNEL_BUILDER(DEVICE_TYPE, BUILDER_CLASS)                         \
  static const AkgKernelBuildRegister g_akg_kernel_builder_##DEVICE_TYPE##_##_reg( \
    DEVICE_TYPE, []() { return std::make_shared<BUILDER_CLASS>(); });
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_AKG_KERNEL_BUILD_MANAGER_H_
