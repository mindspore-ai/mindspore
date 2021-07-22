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

using mindspore::kernel::CreateKernel;
using mindspore::kernel::KernelDesc;

namespace mindspore::lite {
namespace {
static const int kKernelMaxNum = (kNumberTypeEnd - kNumberTypeBegin - 1) * (PrimitiveType_MAX - PrimitiveType_MIN);
}  // namespace

int RegistryKernelImpl::GetFuncIndex(const kernel::KernelDesc &desc) {
  if (desc.data_type >= kNumberTypeEnd) {
    return -1;
  }
  int data_type_index = static_cast<int>(desc.data_type) - kNumberTypeBegin - 1;
  if (data_type_index < 0) {
    return -1;
  }
  return data_type_index * op_type_length_ + desc.type;
}

int RegistryKernelImpl::RegCustomKernel(const std::string &arch, const std::string &provider, TypeId data_type,
                                        const std::string &type, CreateKernel creator) {
  if (data_type >= kNumberTypeEnd) {
    MS_LOG(ERROR) << "invalid data_type: " << data_type << "!provider: " << provider;
    return RET_ERROR;
  }
  std::unique_lock<std::mutex> lock(lock_);
  if (custom_kernel_creators_[provider][arch][type] == nullptr) {
    custom_kernel_creators_[provider][arch][type] =
      reinterpret_cast<CreateKernel *>(calloc(data_type_length_, sizeof(CreateKernel)));
    if (custom_kernel_creators_[provider][arch][type] == nullptr) {
      MS_LOG(ERROR) << "malloc custom kernel creator fail!provider: " << provider << ", arch: " << arch;
      return RET_ERROR;
    }
  }

  int data_type_index = data_type - kNumberTypeBegin - 1;
  if (data_type_index < 0 || data_type_index >= data_type_length_) {
    MS_LOG(ERROR) << "invalid data_type: " << data_type << "!provider: " << provider;
    return RET_ERROR;
  }
  custom_kernel_creators_[provider][arch][type][data_type_index] = creator;
  return RET_OK;
}

int RegistryKernelImpl::RegKernel(const std::string &arch, const std::string &provider, TypeId data_type, int type,
                                  kernel::CreateKernel creator) {
  std::unique_lock<std::mutex> lock(lock_);
  auto iter = kernel_creators_.find(provider);
  if (iter == kernel_creators_.end()) {
    kernel_creators_[provider][arch] = reinterpret_cast<CreateKernel *>(calloc(kKernelMaxNum, sizeof(CreateKernel)));
    if (kernel_creators_[provider][arch] == nullptr) {
      MS_LOG(ERROR) << "malloc kernel creator buffer fail! provider: " << provider << ",arch:" << arch;
      return RET_ERROR;
    }
  } else {
    auto iter_arch = iter->second.find(arch);
    if (iter_arch == iter->second.end()) {
      iter->second[arch] = reinterpret_cast<CreateKernel *>(calloc(kKernelMaxNum, sizeof(CreateKernel)));
      if (iter->second[arch] == nullptr) {
        MS_LOG(ERROR) << "malloc kernel creator buffer fail! provider: " << provider << ",arch:" << arch;
        return RET_ERROR;
      }
    }
  }

  KernelDesc desc = {data_type, type, arch, provider};
  int index = GetFuncIndex(desc);
  if (index >= kKernelMaxNum || index < 0) {
    MS_LOG(ERROR) << "invalid kernel key, arch " << arch << ", data_type" << data_type << ",op type " << type;
    return RET_ERROR;
  }

  kernel_creators_[provider][arch][index] = creator;
  return RET_OK;
}

kernel::CreateKernel RegistryKernelImpl::GetCustomKernelCreator(const schema::Primitive *primitive,
                                                                kernel::KernelDesc *desc) {
  int data_type_index = static_cast<int>(desc->data_type) - kNumberTypeBegin - 1;
  if (data_type_index < 0) {
    return nullptr;
  }
  auto param = primitive->value_as_Custom();
  MS_ASSERT(param != nullptr);
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

kernel::CreateKernel RegistryKernelImpl::GetProviderCreator(const schema::Primitive *primitive,
                                                            kernel::KernelDesc *desc) {
  kernel::CreateKernel creator = nullptr;
  std::unique_lock<std::mutex> lock(lock_);
  if (desc->type == schema::PrimitiveType_Custom) {
    return GetCustomKernelCreator(primitive, desc);
  }

  auto index = GetFuncIndex(*desc);
  if (index >= kKernelMaxNum || index < 0) {
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
}  // namespace mindspore::lite
