/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "src/registry/register_kernel_impl.h"
#include "include/registry/register_kernel.h"
#include "include/errorcode.h"
#include "src/common/version_manager.h"
#include "src/common/log_adapter.h"

using mindspore::registry::CreateKernel;
using mindspore::registry::KernelDesc;
using mindspore::schema::PrimitiveType_MAX;
using mindspore::schema::PrimitiveType_MIN;
namespace mindspore::registry {
namespace {
static const auto kOpTypeLen = PrimitiveType_MAX - PrimitiveType_MIN + 1;
static const auto kDataTypeLen =
  static_cast<int>(DataType::kNumberTypeEnd) - static_cast<int>(DataType::kNumberTypeBegin) - 1;
static const auto kKernelMaxNum = kOpTypeLen * kDataTypeLen;
static constexpr auto kMaxProviderNum = 10;
static constexpr auto kMaxArchPerProviderNum = 10;
static constexpr auto kMaxCustomTypeNum = 200;
}  // namespace

int RegistryKernelImpl::GetFuncIndex(const KernelDesc &desc) const {
  if (desc.data_type >= DataType::kNumberTypeEnd) {
    return -1;
  }
  int data_type_index = static_cast<int>(desc.data_type) - static_cast<int>(DataType::kNumberTypeBegin) - 1;
  if (data_type_index < 0) {
    return -1;
  }
  int index = data_type_index * kOpTypeLen + desc.type;
  if (index >= kKernelMaxNum) {
    return -1;
  }
  return index;
}

Status RegistryKernelImpl::RegCustomKernel(const std::string &arch, const std::string &provider, DataType data_type,
                                           const std::string &type, const CreateKernel creator) {
  int data_type_index = static_cast<int>(data_type) - static_cast<int>(DataType::kNumberTypeBegin) - 1;
  if (data_type_index < 0 || data_type_index >= kDataTypeLen) {
    MS_LOG(ERROR) << "invalid data_type: " << static_cast<int>(data_type) << "!provider: " << provider;
    return kLiteError;
  }
  std::unique_lock<std::mutex> lock(lock_);
  auto provider_iter = custom_kernel_creators_.find(provider);
  if (provider_iter == custom_kernel_creators_.end() && custom_kernel_creators_.size() >= kMaxProviderNum) {
    MS_LOG(ERROR) << "register too many provider!";
    return kLiteError;
  }
  if (provider_iter != custom_kernel_creators_.end()) {
    auto arch_iter = provider_iter->second.find(arch);
    if (arch_iter == provider_iter->second.end()) {
      if (provider_iter->second.size() >= kMaxArchPerProviderNum) {
        MS_LOG(ERROR) << "register too many arch!";
        return kLiteError;
      }
    } else {
      auto type_iter = arch_iter->second.find(type);
      if (type_iter == arch_iter->second.end() && arch_iter->second.size() >= kMaxCustomTypeNum) {
        MS_LOG(ERROR) << "register too many type!";
        return kLiteError;
      }
    }
  }
  if (custom_kernel_creators_[provider][arch][type] == nullptr) {
    custom_kernel_creators_[provider][arch][type] =
      reinterpret_cast<CreateKernel *>(calloc(kDataTypeLen, sizeof(CreateKernel)));
    if (custom_kernel_creators_[provider][arch][type] == nullptr) {
      MS_LOG(ERROR) << "malloc custom kernel creator fail!provider: " << provider << ", arch: " << arch;
      return kLiteError;
    }
  }

  custom_kernel_creators_[provider][arch][type][data_type_index] = creator;
  return kSuccess;
}

Status RegistryKernelImpl::RegKernel(const std::string &arch, const std::string &provider, DataType data_type, int type,
                                     const registry::CreateKernel creator) {
  if (type <= static_cast<int>(PrimitiveType_MIN) || type > static_cast<int>(PrimitiveType_MAX)) {
    MS_LOG(ERROR) << "Invalid op type : " << type;
    return kLiteParamInvalid;
  }
  KernelDesc desc = {data_type, type, arch, provider};
  int index = GetFuncIndex(desc);
  if (index < 0) {
    MS_LOG(ERROR) << "invalid kernel key, arch " << arch << ", data_type" << static_cast<int>(data_type) << ",op type "
                  << type;
    return kLiteError;
  }
  std::unique_lock<std::mutex> lock(lock_);
  auto iter = kernel_creators_.find(provider);
  if (iter == kernel_creators_.end()) {
    if (kernel_creators_.size() >= kMaxProviderNum) {
      MS_LOG(ERROR) << "register too many provider!";
      return kLiteError;
    }
    kernel_creators_[provider][arch] = reinterpret_cast<CreateKernel *>(calloc(kKernelMaxNum, sizeof(CreateKernel)));
    if (kernel_creators_[provider][arch] == nullptr) {
      MS_LOG(ERROR) << "malloc kernel creator buffer fail! provider: " << provider << ",arch:" << arch;
      return kLiteError;
    }
  } else {
    auto iter_arch = iter->second.find(arch);
    if (iter_arch == iter->second.end()) {
      if (iter->second.size() >= kMaxArchPerProviderNum) {
        MS_LOG(ERROR) << "register too many arch!";
        return kLiteError;
      }
      iter->second[arch] = reinterpret_cast<CreateKernel *>(calloc(kKernelMaxNum, sizeof(CreateKernel)));
      if (iter->second[arch] == nullptr) {
        MS_LOG(ERROR) << "malloc kernel creator buffer fail! provider: " << provider << ",arch:" << arch;
        return kLiteError;
      }
    }
  }

  kernel_creators_[provider][arch][index] = creator;
  return kSuccess;
}

registry::CreateKernel RegistryKernelImpl::GetCustomKernelCreator(const schema::Primitive *primitive,
                                                                  KernelDesc *desc) {
  int data_type_index = static_cast<int>(desc->data_type) - static_cast<int>(DataType::kNumberTypeBegin) - 1;
  if (data_type_index < 0 || desc->data_type >= DataType::kNumberTypeEnd) {
    return nullptr;
  }
  auto param = primitive->value_as_Custom();
  if (param == nullptr || param->type() == nullptr) {
    return nullptr;
  }
  auto custom_type = param->type()->str();
  if (!desc->provider.empty() && !desc->arch.empty()) {
    auto creator_buf = custom_kernel_creators_[desc->provider][desc->arch][custom_type];
    if (creator_buf != nullptr && creator_buf[data_type_index] != nullptr) {
      return creator_buf[data_type_index];
    }
    return nullptr;
  }
  for (auto &&providers : custom_kernel_creators_) {
    auto archs = providers.second;
    auto archs_iter = std::find_if(archs.begin(), archs.end(), [custom_type, data_type_index](auto &&item) {
      return item.second[custom_type] != nullptr && item.second[custom_type][data_type_index] != nullptr;
    });
    if (archs_iter != archs.end()) {
      desc->arch = archs_iter->first;
      return archs_iter->second[custom_type][data_type_index];
    }
  }

  return nullptr;
}

registry::CreateKernel RegistryKernelImpl::GetProviderCreator(const schema::Primitive *primitive, KernelDesc *desc) {
  registry::CreateKernel creator = nullptr;
  std::unique_lock<std::mutex> lock(lock_);
  if (desc->type == schema::PrimitiveType_Custom) {
    return GetCustomKernelCreator(primitive, desc);
  }

  auto index = GetFuncIndex(*desc);
  if (index < 0) {
    return nullptr;
  }
  for (auto &&item : kernel_creators_) {
    if (item.first != desc->provider) {
      continue;
    }
    for (auto &&arch_item : item.second) {
      if (arch_item.first != desc->arch) {
        continue;
      }
      creator = arch_item.second[index];
      if (creator != nullptr) {
        break;
      }
    }
    if (creator != nullptr) {
      break;
    }
  }
  return creator;
}

RegistryKernelImpl::~RegistryKernelImpl() {
  for (auto &&item : kernel_creators_) {
    for (auto &&creator : item.second) {
      free(creator.second);
      creator.second = nullptr;
    }
  }
  for (auto &&provider : custom_kernel_creators_) {
    for (auto &&arch : provider.second) {
      for (auto &&creator : arch.second) {
        free(creator.second);
        creator.second = nullptr;
      }
    }
  }
}
}  // namespace mindspore::registry
