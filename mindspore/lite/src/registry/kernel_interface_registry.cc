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
#include "include/registry/kernel_interface.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/common/version_manager.h"
#include "schema/model_generated.h"

using mindspore::kernel::KernelInterfaceCreator;
using mindspore::schema::PrimitiveType_MAX;
using mindspore::schema::PrimitiveType_MIN;
namespace mindspore {
namespace lite {
namespace {
static const auto kMaxKernelNum = PrimitiveType_MAX - PrimitiveType_MIN;
std::string GetCustomType(const schema::Primitive *primitive) {
  auto param = primitive->value_as_Custom();
  MS_ASSERT(param != nullptr);
  return param->type()->str();
}
}  // namespace

int KernelInterfaceRegistry::CustomReg(const std::string &provider, const std::string &type,
                                       KernelInterfaceCreator creator) {
  custom_creators_[provider][type] = creator;
  return RET_OK;
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
  const schema::Primitive *primitive) {
  MS_ASSERT(primitive != nullptr);
  std::unique_lock<std::mutex> lock(mutex_);
  auto &&type = GetCustomType(primitive);
  for (auto &&item : custom_creators_) {
    auto &&provider = item.first;
    auto kernel = GetCustomCacheInterface(provider, type);
    if (kernel != nullptr) {
      return kernel;
    }
    auto provider_iter = custom_creators_.find(provider);
    if (provider_iter == custom_creators_.end()) {
      return nullptr;
    }
    auto creator_iter = provider_iter->second.find(type);
    if (creator_iter != provider_iter->second.end()) {
      kernel = creator_iter->second();
      custom_kernels_[provider][type] = kernel;
      return kernel;
    }
  }

  return nullptr;
}

std::shared_ptr<kernel::KernelInterface> KernelInterfaceRegistry::GetKernelInterface(
  const std::string &provider, const schema::Primitive *primitive) {
  MS_ASSERT(primitive != nullptr);
  int op_type = primitive->value_type();
  if (op_type == schema::PrimitiveType_Custom) {
    return GetCustomKernelInterface(primitive);
  }
  std::unique_lock<std::mutex> lock(mutex_);
  auto kernel = GetCacheInterface(provider, op_type);
  if (kernel != nullptr) {
    return kernel;
  }
  auto iter = kernel_creators_.find(provider);
  if (iter == kernel_creators_.end()) {
    return nullptr;
  }

  auto creator = iter->second[op_type];
  if (creator != nullptr) {
    kernel = creator();
    kernel_interfaces_[provider][op_type] = kernel;
    return kernel;
  }
  return nullptr;
}

int KernelInterfaceRegistry::Reg(const std::string &provider, int op_type, KernelInterfaceCreator creator) {
  if (op_type < PrimitiveType_MIN || op_type > kMaxKernelNum) {
    MS_LOG(ERROR) << "reg op_type invalid!op_type: " << op_type << ", max value: " << kMaxKernelNum;
    return RET_ERROR;
  }

  std::unique_lock<std::mutex> lock(mutex_);
  auto iter = kernel_creators_.find(provider);
  if (iter == kernel_creators_.end()) {
    kernel_creators_[provider] =
      reinterpret_cast<KernelInterfaceCreator *>(calloc(kMaxKernelNum, sizeof(KernelInterfaceCreator)));
    if (kernel_creators_[provider] == nullptr) {
      MS_LOG(ERROR) << "malloc kernel dev delegate creator fail!";
      return RET_ERROR;
    }
  }

  kernel_creators_[provider][op_type] = creator;
  return RET_OK;
}

KernelInterfaceRegistry::~KernelInterfaceRegistry() {
  for (auto &&item : kernel_creators_) {
    free(item.second);
    item.second = nullptr;
  }
}
}  // namespace lite
}  // namespace mindspore
