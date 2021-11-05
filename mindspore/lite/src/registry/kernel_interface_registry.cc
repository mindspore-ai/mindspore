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
#include "src/registry/kernel_interface_registry.h"
#include <memory>
#include "include/registry/register_kernel_interface.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/common/version_manager.h"
#include "schema/model_generated.h"
#include "include/api/kernel.h"

using mindspore::registry::KernelInterfaceCreator;
using mindspore::schema::PrimitiveType_MAX;
using mindspore::schema::PrimitiveType_MIN;
namespace mindspore {
namespace registry {
namespace {
static constexpr auto kMaxProviderNum = 10;
static constexpr auto KMaxCustomTypeNum = 200;
static const auto kMaxKernelNum = PrimitiveType_MAX - PrimitiveType_MIN + 1;
std::string GetCustomType(const schema::Primitive *primitive) {
  auto param = primitive->value_as_Custom();
  if (param == nullptr || param->type() == nullptr) {
    return "";
  }

  return param->type()->str();
}
}  // namespace

Status KernelInterfaceRegistry::CustomReg(const std::string &provider, const std::string &type,
                                          const KernelInterfaceCreator creator) {
  auto provider_iter = custom_creators_.find(provider);
  if (provider_iter == custom_creators_.end() && custom_creators_.size() >= kMaxProviderNum) {
    MS_LOG(ERROR) << "register too many provider!";
    return kLiteError;
  }
  if (provider_iter != custom_creators_.end()) {
    auto type_iter = provider_iter->second.find(type);
    if (type_iter == provider_iter->second.end() && provider_iter->second.size() >= KMaxCustomTypeNum) {
      MS_LOG(ERROR) << "register too many custom type!";
      return kLiteError;
    }
  }
  custom_creators_[provider][type] = creator;
  return kSuccess;
}

std::shared_ptr<kernel::KernelInterface> KernelInterfaceRegistry::GetCacheInterface(const std::string &provider,
                                                                                    int op_type) {
  if (provider.empty()) {
    return nullptr;
  }
  auto provider_iter = kernel_interfaces_.find(provider);
  if (provider_iter != kernel_interfaces_.end()) {
    auto kernel_iter = provider_iter->second.find(op_type);
    if (kernel_iter != provider_iter->second.end()) {
      return kernel_iter->second;
    }
  }
  return nullptr;
}

std::shared_ptr<kernel::KernelInterface> KernelInterfaceRegistry::GetCustomCacheInterface(const std::string &provider,
                                                                                          const std::string &type) {
  if (provider.empty()) {
    return nullptr;
  }
  auto provider_iter = custom_kernels_.find(provider);
  if (provider_iter == custom_kernels_.end()) {
    return nullptr;
  }
  auto kernel_iter = provider_iter->second.find(type);
  if (kernel_iter != provider_iter->second.end()) {
    return kernel_iter->second;
  }
  return nullptr;
}

std::shared_ptr<kernel::KernelInterface> KernelInterfaceRegistry::GetCustomKernelInterface(
  const schema::Primitive *primitive, const kernel::Kernel *kernel) {
  std::unique_lock<std::mutex> lock(mutex_);
  std::string type;
  if (kernel == nullptr) {
    type = GetCustomType(primitive);
  } else {
    type = kernel->GetAttr("type");
  }
  for (auto &&item : custom_creators_) {
    auto &&provider = item.first;
    auto kernel_interface = GetCustomCacheInterface(provider, type);
    if (kernel_interface != nullptr) {
      return kernel_interface;
    }
    auto provider_iter = custom_creators_.find(provider);
    if (provider_iter == custom_creators_.end()) {
      return nullptr;
    }
    auto creator_iter = provider_iter->second.find(type);
    if (creator_iter != provider_iter->second.end()) {
      kernel_interface = creator_iter->second();
      custom_kernels_[provider][type] = kernel_interface;
      return kernel_interface;
    }
  }

  return nullptr;
}

std::shared_ptr<kernel::KernelInterface> KernelInterfaceRegistry::GetKernelInterface(const std::string &provider,
                                                                                     const schema::Primitive *primitive,
                                                                                     const kernel::Kernel *kernel) {
  if (primitive == nullptr && kernel == nullptr) {
    return nullptr;
  }
  int op_type;
  if (kernel == nullptr) {
    op_type = static_cast<int>(primitive->value_type());
  } else {
    op_type = static_cast<int>(kernel->type());
  }
  if (op_type > PrimitiveType_MAX || op_type <= PrimitiveType_MIN) {
    return nullptr;
  }
  if (op_type == schema::PrimitiveType_Custom) {
    return GetCustomKernelInterface(primitive, kernel);
  }

  std::unique_lock<std::mutex> lock(mutex_);
  auto kernel_interface = GetCacheInterface(provider, op_type);
  if (kernel_interface != nullptr) {
    return kernel_interface;
  }
  auto iter = kernel_creators_.find(provider);
  if (iter == kernel_creators_.end()) {
    return nullptr;
  }

  auto creator = iter->second[op_type];
  if (creator != nullptr) {
    kernel_interface = creator();
    kernel_interfaces_[provider][op_type] = kernel_interface;
    return kernel_interface;
  }
  return nullptr;
}

Status KernelInterfaceRegistry::Reg(const std::string &provider, int op_type, const KernelInterfaceCreator creator) {
  if (op_type <= PrimitiveType_MIN || op_type > PrimitiveType_MAX) {
    MS_LOG(ERROR) << "reg op_type invalid!op_type: " << op_type << ", max value: " << PrimitiveType_MAX;
    return kLiteParamInvalid;
  }

  if (provider.empty()) {
    MS_LOG(ERROR) << "Input provider is empty!";
    return kLiteParamInvalid;
  }
  std::unique_lock<std::mutex> lock(mutex_);
  auto iter = kernel_creators_.find(provider);
  if (iter == kernel_creators_.end()) {
    if (kernel_creators_.size() >= kMaxProviderNum) {
      MS_LOG(ERROR) << "register too many provider!";
      return kLiteError;
    }
    kernel_creators_[provider] =
      reinterpret_cast<KernelInterfaceCreator *>(calloc(kMaxKernelNum, sizeof(KernelInterfaceCreator)));
    if (kernel_creators_[provider] == nullptr) {
      MS_LOG(ERROR) << "malloc kernel dev delegate creator fail!";
      return kLiteError;
    }
  }

  kernel_creators_[provider][op_type] = creator;
  return kSuccess;
}

KernelInterfaceRegistry::~KernelInterfaceRegistry() {
  for (auto &&item : kernel_creators_) {
    free(item.second);
    item.second = nullptr;
  }
}
}  // namespace registry
}  // namespace mindspore
