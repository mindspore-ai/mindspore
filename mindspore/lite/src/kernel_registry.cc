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
#include "src/kernel_registry.h"
#include <utility>
#include "include/errorcode.h"
#include "src/ops/populate/populate_register.h"
#include "src/common/version_manager.h"
#include "nnacl/pooling_parameter.h"
#if defined(ENABLE_FP16) && defined(ENABLE_ARM)
#if defined(__ANDROID__)
#include <asm/hwcap.h>
#endif
#include "common/utils.h"
#include "src/common/log_adapter.h"
#include "src/common/utils.h"
#endif
#include "src/common/tensor_util.h"

using mindspore::kernel::CreateKernel;
using mindspore::kernel::kBuiltin;
using mindspore::kernel::kCPU;
using mindspore::kernel::KERNEL_ARCH;
using mindspore::kernel::KernelCreator;
using mindspore::kernel::KernelKey;

namespace mindspore::lite {
namespace {
static const int kKernelMaxNum = (kNumberTypeEnd - kNumberTypeBegin - 1) * (PrimitiveType_MAX - PrimitiveType_MIN);
}  // namespace

KernelRegistry *KernelRegistry::GetInstance() {
  static KernelRegistry instance;

  std::unique_lock<std::mutex> malloc_creator_array(instance.lock_);
  if (instance.creator_arrays_ == nullptr) {
    instance.creator_arrays_ = reinterpret_cast<KernelCreator *>(malloc(array_size_ * sizeof(KernelCreator)));
    if (instance.creator_arrays_ == nullptr) {
      return nullptr;
    }
    memset(instance.creator_arrays_, 0, array_size_ * sizeof(KernelCreator));
  }
  return &instance;
}

std::set<std::string> KernelRegistry::AllProviders() {
  std::set<std::string> providers;
  for (auto &&item : kernel_creators_) {
    providers.insert(item.first);
  }
  for (auto &&item : custom_kernel_creators_) {
    providers.insert(item.first);
  }
  return providers;
}

int KernelRegistry::GetFuncIndex(const kernel::KernelKey &desc) {
  if (desc.data_type >= kNumberTypeEnd) {
    return -1;
  }
  int data_type_index = static_cast<int>(desc.data_type) - kNumberTypeBegin - 1;
  if (data_type_index < 0) {
    return -1;
  }
  return data_type_index * op_type_length_ + desc.type;
}

int KernelRegistry::RegCustomKernel(const std::string &arch, const std::string &provider, TypeId data_type,
                                    const std::string &type, CreateKernel creator) {
  if (data_type >= kNumberTypeEnd) {
    MS_LOG(ERROR) << "invalid data_type: " << data_type << "!provider: " << provider;
    return RET_ERROR;
  }
  std::unique_lock<std::mutex> lock(lock_);
  if (custom_kernel_creators_[provider][arch][type] == nullptr) {
    custom_kernel_creators_[provider][arch][type] =
      reinterpret_cast<CreateKernel *>(malloc(data_type_length_ * sizeof(CreateKernel)));
    if (custom_kernel_creators_[provider][arch][type] == nullptr) {
      MS_LOG(ERROR) << "malloc custom kernel creator fail!provider: " << provider << ", arch: " << arch;
      return RET_ERROR;
    }
    memset(custom_kernel_creators_[provider][arch][type], 0, data_type_length_ * sizeof(CreateKernel));
  }

  int data_type_index = data_type - kNumberTypeBegin - 1;
  if (data_type_index < 0 || data_type_index >= data_type_length_) {
    MS_LOG(ERROR) << "invalid data_type: " << data_type << "!provider: " << provider;
    return RET_ERROR;
  }
  custom_kernel_creators_[provider][arch][type][data_type_index] = creator;
  return RET_OK;
}

int KernelRegistry::RegKernel(const std::string &arch, const std::string &provider, TypeId data_type, int type,
                              kernel::CreateKernel creator) {
  std::unique_lock<std::mutex> lock(lock_);
  auto iter = kernel_creators_.find(provider);
  if (iter == kernel_creators_.end()) {
    kernel_creators_[provider][arch] = reinterpret_cast<CreateKernel *>(malloc(kKernelMaxNum * sizeof(CreateKernel)));
    if (kernel_creators_[provider][arch] == nullptr) {
      MS_LOG(ERROR) << "malloc kernel creator buffer fail! provider: " << provider << ",arch:" << arch;
      return RET_ERROR;
    }
    memset(kernel_creators_[provider][arch], 0, kKernelMaxNum * sizeof(CreateKernel));
  } else {
    auto iter_arch = iter->second.find(arch);
    if (iter_arch == iter->second.end()) {
      iter->second[arch] = reinterpret_cast<CreateKernel *>(malloc(kKernelMaxNum * sizeof(CreateKernel)));
      if (iter->second[arch] == nullptr) {
        MS_LOG(ERROR) << "malloc kernel creator buffer fail! provider: " << provider << ",arch:" << arch;
        return RET_ERROR;
      }
      memset(iter->second[arch], 0, kKernelMaxNum * sizeof(CreateKernel));
    }
  }

  KernelKey desc = {kCPU, data_type, type, arch, provider};
  int index = GetFuncIndex(desc);
  if (index >= kKernelMaxNum || index < 0) {
    MS_LOG(ERROR) << "invalid kernel key, arch " << arch << ", data_type" << data_type << ",op type " << type;
    return RET_ERROR;
  }

  kernel_creators_[provider][arch][index] = creator;
  return RET_OK;
}

int KernelRegistry::Init() { return RET_OK; }

kernel::KernelCreator KernelRegistry::GetCreator(const KernelKey &desc) {
  if (desc.provider == kBuiltin) {
    int index = GetCreatorFuncIndex(desc);
    if (index >= array_size_ || index < 0) {
      MS_LOG(ERROR) << "invalid kernel key, arch " << desc.arch << ", data_type " << desc.data_type << ",op type "
                    << desc.type;
      return nullptr;
    }
    return creator_arrays_[index];
  }
  MS_LOG(ERROR) << "Call wrong interface!provider: " << desc.provider;
  return nullptr;
}

kernel::CreateKernel KernelRegistry::GetProviderCreator(const kernel::KernelKey &desc,
                                                        const schema::Primitive *primitive) {
  kernel::CreateKernel creator = nullptr;
  std::unique_lock<std::mutex> lock(lock_);
  if (desc.type == schema::PrimitiveType_Custom) {
    int data_type_index = static_cast<int>(desc.data_type) - kNumberTypeBegin - 1;
    if (data_type_index < 0) {
      return nullptr;
    }
    auto param = primitive->value_as_Custom();
    MS_ASSERT(param != nullptr);
    auto custom_type = param->type()->str();
    auto archs = custom_kernel_creators_[desc.provider];
    auto archs_iter = std::find_if(archs.begin(), archs.end(), [custom_type, data_type_index](auto &&item) {
      return item.second[custom_type] != nullptr && item.second[custom_type][data_type_index] != nullptr;
    });
    if (archs_iter != archs.end()) {
      return archs_iter->second[custom_type][data_type_index];
    }
    return nullptr;
  }
  auto index = GetFuncIndex(desc);
  if (index >= kKernelMaxNum || index < 0) {
    return nullptr;
  }
  for (auto &&item : kernel_creators_) {
    for (auto &&arch_item : item.second) {
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

int KernelRegistry::GetCreatorFuncIndex(const kernel::KernelKey desc) {
  int index;
  int device_index = static_cast<int>(desc.arch) - kKernelArch_MIN;
  int dType_index = static_cast<int>(desc.data_type) - kNumberTypeBegin;
  int op_index = static_cast<int>(desc.type);
  index = device_index * data_type_length_ * op_type_length_ + dType_index * op_type_length_ + op_index;
  return index;
}

void KernelRegistry::RegKernel(const KernelKey desc, const kernel::KernelCreator creator) {
  int index = GetCreatorFuncIndex(desc);
  if (index >= array_size_) {
    MS_LOG(ERROR) << "invalid kernel key, arch " << desc.arch << ", data_type" << desc.data_type << ",op type "
                  << desc.type;
    return;
  }
  creator_arrays_[index] = creator;
}

void KernelRegistry::RegKernel(KERNEL_ARCH arch, TypeId data_type, int op_type, kernel::KernelCreator creator) {
  KernelKey desc = {arch, data_type, op_type};
  int index = GetCreatorFuncIndex(desc);
  if (index >= array_size_) {
    MS_LOG(ERROR) << "invalid kernel key, arch " << desc.arch << ", data_type" << desc.data_type << ",op type "
                  << desc.type;
    return;
  }
  creator_arrays_[index] = creator;
}

bool KernelRegistry::Merge(const std::unordered_map<KernelKey, KernelCreator> &new_creators) { return false; }

KernelRegistry::~KernelRegistry() {
  KernelRegistry *instance = GetInstance();
  std::unique_lock<std::mutex> malloc_creator_array(instance->lock_);
  if (instance->creator_arrays_ != nullptr) {
    free(instance->creator_arrays_);
    instance->creator_arrays_ = nullptr;
  }

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

bool KernelRegistry::SupportKernel(const KernelKey &key) {
  auto kernel_creator = GetCreator(key);
  return kernel_creator != nullptr;
}

int KernelRegistry::GetKernel(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                              const InnerContext *ctx, const kernel::KernelKey &key, OpParameter *parameter,
                              kernel::LiteKernel **kernel, const void *primitive) {
  MS_ASSERT(ctx != nullptr);
  MS_ASSERT(kernel != nullptr);
  if (key.provider == kBuiltin) {
    auto creator = GetCreator(key);
    if (creator != nullptr) {
      auto inner_kernel = creator(in_tensors, out_tensors, parameter, ctx, key);
      if (inner_kernel != nullptr) {
        inner_kernel->set_registry_data_type(key.data_type);
        auto *lite_kernel = new (std::nothrow) kernel::LiteKernel(inner_kernel);
        if (lite_kernel != nullptr) {
          lite_kernel->set_desc(key);
          *kernel = lite_kernel;
          return RET_OK;
        } else {
          delete inner_kernel;
        }
      }
      return RET_ERROR;
    }
  } else {
    auto creator = GetProviderCreator(key, static_cast<const schema::Primitive *>(primitive));
    if (creator == nullptr) {
      return RET_NOT_SUPPORT;
    }
    std::vector<tensor::MSTensor *> tensors_in;
    Tensor2MSTensor(std::move(in_tensors), &tensors_in);
    std::vector<tensor::MSTensor *> tensors_out;
    Tensor2MSTensor(std::move(out_tensors), &tensors_out);
    auto base_kernel = creator(tensors_in, tensors_out, static_cast<const schema::Primitive *>(primitive), ctx);
    if (base_kernel != nullptr) {
      auto *lite_kernel = new (std::nothrow) kernel::LiteKernel(base_kernel);
      if (lite_kernel != nullptr) {
        lite_kernel->set_desc(key);
        *kernel = lite_kernel;
        return RET_OK;
      } else {
        delete base_kernel;
      }
    }
    return RET_ERROR;
  }
  return RET_NOT_SUPPORT;
}
}  // namespace mindspore::lite
