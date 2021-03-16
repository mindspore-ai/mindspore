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
#include "include/errorcode.h"
#include "src/ops/populate/populate_register.h"
#include "src/common/version_manager.h"
#include "src/common/prim_util.h"
#include "nnacl/pooling_parameter.h"
#ifdef ENABLE_ARM64
#if defined(__ANDROID__)
#include <asm/hwcap.h>
#endif
#include "common/utils.h"
#include "src/common/log_adapter.h"
#include "src/common/utils.h"
#endif

using mindspore::kernel::kCPU;
using mindspore::kernel::KERNEL_ARCH;
using mindspore::kernel::KernelCreator;
using mindspore::kernel::KernelKey;

namespace mindspore::lite {
KernelRegistry *KernelRegistry::GetInstance() {
  static KernelRegistry instance;

  std::unique_lock<std::mutex> malloc_creator_array(instance.lock_);
  if (instance.creator_arrays_ == nullptr) {
    instance.creator_arrays_ = reinterpret_cast<KernelCreator *>(malloc(array_size_ * sizeof(KernelRegistry)));
    if (instance.creator_arrays_ == nullptr) {
      return nullptr;
    }
    memset(instance.creator_arrays_, 0, array_size_ * sizeof(KernelRegistry));
  }
  return &instance;
}

int KernelRegistry::Init() {
#ifdef ENABLE_ARM64
  if (mindspore::lite::IsSupportSDot()) {
    MS_LOG(INFO) << "The current device supports Sdot.";
  } else {
    MS_LOG(INFO) << "The current device NOT supports Sdot.";
  }
  if (mindspore::lite::IsSupportFloat16()) {
    MS_LOG(INFO) << "The current device supports float16.";
  } else {
    MS_LOG(INFO) << "The current device NOT supports float16.";
  }
#endif
  return RET_OK;
}

kernel::KernelCreator KernelRegistry::GetCreator(const KernelKey &desc) {
  int index = GetCreatorFuncIndex(desc);
  if (index >= array_size_ || index < 0) {
    MS_LOG(ERROR) << "invalid kernel key, arch " << desc.arch << ", data_type" << desc.data_type << ",op type "
                  << desc.type;
    return nullptr;
  }
  auto it = creator_arrays_[index];
  if (it != nullptr) {
    return it;
  }
  return nullptr;
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

int KernelRegistry::GetKernel(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                              const InnerContext *ctx, const kernel::KernelKey &key, OpParameter *parameter,
                              kernel::LiteKernel **kernel) {
  MS_ASSERT(ctx != nullptr);
  MS_ASSERT(kernel != nullptr);
  auto creator = GetCreator(key);
  if (creator != nullptr) {
    *kernel = creator(in_tensors, out_tensors, parameter, ctx, key);
    if (*kernel != nullptr) {
      (*kernel)->set_desc(key);
      return RET_OK;
    }
    return RET_ERROR;
  }
  return RET_NOT_SUPPORT;
}
}  // namespace mindspore::lite
