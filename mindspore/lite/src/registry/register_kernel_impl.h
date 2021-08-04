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
#include "src/registry/register_utils.h"

using mindspore::schema::PrimitiveType_MAX;
using mindspore::schema::PrimitiveType_MIN;

namespace mindspore::lite {
class RegistryKernelImpl {
 public:
  RegistryKernelImpl() = default;
  virtual ~RegistryKernelImpl();

  static RegistryKernelImpl *GetInstance() {
    static RegistryKernelImpl instance;
    return &instance;
  }

  int GetFuncIndex(const kernel::KernelDesc &desc);

  int RegCustomKernel(const std::string &arch, const std::string &provider, TypeId data_type, const std::string &type,
                      kernel::CreateKernel creator);

  int RegKernel(const std::string &arch, const std::string &provider, TypeId data_type, int type,
                kernel::CreateKernel creator);

  virtual kernel::CreateKernel GetProviderCreator(const schema::Primitive *primitive, kernel::KernelDesc *desc);

  const std::map<std::string, std::unordered_map<std::string, kernel::CreateKernel *>> &kernel_creators() {
    return kernel_creators_;
  }

 protected:
  static const int data_type_length_{kNumberTypeEnd - kNumberTypeBegin + 1};
  static const int op_type_length_{PrimitiveType_MAX - PrimitiveType_MIN + 1};
  std::map<std::string, std::unordered_map<std::string, kernel::CreateKernel *>> kernel_creators_;
  // keys:provider, arch, type
  std::map<std::string, std::map<std::string, std::unordered_map<std::string, kernel::CreateKernel *>>>
    custom_kernel_creators_;

 private:
  std::mutex lock_;

  kernel::CreateKernel GetCustomKernelCreator(const schema::Primitive *primitive, kernel::KernelDesc *desc);
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_REGISTRY_REGISTER_KERNEL_IMPL_H_
