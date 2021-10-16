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

#ifndef MINDSPORE_LITE_SRC_REGISTRY_REGISTER_KERNEL_IMPL_H_
#define MINDSPORE_LITE_SRC_REGISTRY_REGISTER_KERNEL_IMPL_H_

#include <string>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <set>
#include "include/registry/register_kernel.h"

namespace mindspore::registry {
class RegistryKernelImpl {
 public:
  RegistryKernelImpl() = default;
  virtual ~RegistryKernelImpl();

  static RegistryKernelImpl *GetInstance() {
    static RegistryKernelImpl instance;
    return &instance;
  }

  Status RegCustomKernel(const std::string &arch, const std::string &provider, DataType data_type,
                         const std::string &type, registry::CreateKernel creator);

  Status RegKernel(const std::string &arch, const std::string &provider, DataType data_type, int type,
                   registry::CreateKernel creator);

  virtual registry::CreateKernel GetProviderCreator(const schema::Primitive *primitive, registry::KernelDesc *desc);

  const std::map<std::string, std::unordered_map<std::string, registry::CreateKernel *>> &kernel_creators() {
    return kernel_creators_;
  }

 protected:
  // keys:provider, arch
  std::map<std::string, std::unordered_map<std::string, registry::CreateKernel *>> kernel_creators_;

  // keys:provider, arch, type
  std::map<std::string, std::map<std::string, std::unordered_map<std::string, registry::CreateKernel *>>>
    custom_kernel_creators_;

 private:
  std::mutex lock_;

  registry::CreateKernel GetCustomKernelCreator(const schema::Primitive *primitive, registry::KernelDesc *desc);
  int GetFuncIndex(const registry::KernelDesc &desc) const;
};
}  // namespace mindspore::registry

#endif  // MINDSPORE_LITE_SRC_REGISTRY_REGISTER_KERNEL_IMPL_H_
