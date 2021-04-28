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
static const int kKernelMaxNum = (kNumberTypeEnd - kNumberTypeBegin + 1) * (PrimitiveType_MAX - PrimitiveType_MIN + 1);
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

int KernelRegistry::GetFuncIndex(const kernel::KernelKey &desc) {
  int dType_index = static_cast<int>(desc.data_type) - kNumberTypeBegin;
  return dType_index * op_type_length_ + desc.type;
}

int KernelRegistry::RegKernel(const std::string &arch, const std::string &vendor, const TypeId data_type,
                              const int type, kernel::CreateKernel creator) {
  auto vendor_hash = std::hash<std::string>{}(vendor);
  auto arch_hash = std::hash<std::string>{}(arch);
  auto iter = kernel_creators_.find(vendor_hash);
  if (iter == kernel_creators_.end()) {
    all_vendors_.insert(vendor);
    kernel_creators_[vendor_hash][arch_hash] =
      reinterpret_cast<CreateKernel *>(malloc(kKernelMaxNum * sizeof(CreateKernel)));
    if (kernel_creators_[vendor_hash][arch_hash] == nullptr) {
      MS_LOG(ERROR) << "malloc kernel creator buffer fail! vendor: " << vendor << ",arch:" << arch;
      return RET_ERROR;
    }
    memset(kernel_creators_[vendor_hash][arch_hash], 0, kKernelMaxNum * sizeof(CreateKernel));
  } else {
    auto iter_arch = iter->second.find(arch_hash);
    if (iter_arch == iter->second.end()) {
      iter->second[arch_hash] = reinterpret_cast<CreateKernel *>(malloc(kKernelMaxNum * sizeof(CreateKernel)));
      if (iter->second[arch_hash] == nullptr) {
        MS_LOG(ERROR) << "malloc kernel creator buffer fail! vendor: " << vendor << ",arch:" << arch;
        return RET_ERROR;
      }
      memset(iter->second[arch_hash], 0, kKernelMaxNum * sizeof(CreateKernel));
    }
  }

  KernelKey desc = {kCPU, data_type, type, arch, vendor};
  int index = GetFuncIndex(desc);
  if (index >= kKernelMaxNum) {
    MS_LOG(ERROR) << "invalid kernel key, arch " << arch << ", data_type" << data_type << ",op type " << type;
    return RET_ERROR;
  }
  kernel_creators_[vendor_hash][arch_hash][index] = creator;
  return RET_OK;
}

int KernelRegistry::Init() { return RET_OK; }

kernel::KernelCreator KernelRegistry::GetCreator(const KernelKey &desc) {
  if (desc.vendor == kBuiltin) {
    int index = GetCreatorFuncIndex(desc);
    if (index >= array_size_ || index < 0) {
      MS_LOG(ERROR) << "invalid kernel key, arch " << desc.arch << ", data_type" << desc.data_type << ",op type "
                    << desc.type;
      return nullptr;
    }
    return creator_arrays_[index];
  }
  MS_LOG(ERROR) << "Call wrong interface!vendor: " << desc.vendor;
  return nullptr;
}

kernel::CreateKernel KernelRegistry::GetDelegateCreator(const kernel::KernelKey &desc) {
  auto vendor_hash = std::hash<std::string>{}(desc.vendor);
  auto it_by_vendor = kernel_creators_.find(vendor_hash);
  if (it_by_vendor == kernel_creators_.end()) {
    return nullptr;
  }
  auto arch_hash = std::hash<std::string>{}(desc.kernel_arch);
  auto it_by_arch = it_by_vendor->second.find(arch_hash);
  if (it_by_arch == it_by_vendor->second.end()) {
    return nullptr;
  }
  auto index = GetFuncIndex(desc);
  if (index < 0 || index >= kKernelMaxNum) {
    MS_LOG(ERROR) << "invalid kernel key, arch " << desc.kernel_arch << ", data_type" << desc.data_type << ",op type "
                  << desc.type << ", vendor: " << desc.vendor;
    return nullptr;
  }

  return it_by_arch->second[index];
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
}

bool KernelRegistry::SupportKernel(const KernelKey &key) {
  auto kernel_creator = GetCreator(key);
  return kernel_creator != nullptr;
}

kernel::LiteKernel *KernelRegistry::GetKernel(const std::vector<Tensor *> &in_tensors,
                                              const std::vector<Tensor *> &out_tensors, const InnerContext *ctx,
                                              const kernel::KernelKey &key, OpParameter *parameter,
                                              const void *primitive) {
  MS_ASSERT(ctx != nullptr);
  if (key.vendor == kBuiltin) {
    auto creator = GetCreator(key);
    if (creator != nullptr) {
      auto kernel = creator(in_tensors, out_tensors, parameter, ctx, key);
      if (kernel != nullptr) {
        kernel->set_desc(key);
        return kernel;
      }
    }
  } else {
    auto creator = GetDelegateCreator(key);
    if (creator == nullptr) {
      return nullptr;
    }
    std::vector<tensor::MSTensor *> tensors_in;
    Tensor2MSTensor(std::move(in_tensors), &tensors_in);
    std::vector<tensor::MSTensor *> tensors_out;
    Tensor2MSTensor(std::move(out_tensors), &tensors_out);
    return creator(tensors_in, tensors_out, static_cast<const schema::Primitive *>(primitive), ctx);
  }
  return nullptr;
}
}  // namespace mindspore::lite
