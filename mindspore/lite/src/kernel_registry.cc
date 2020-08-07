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
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "ir/dtype/type_id.h"
#ifdef ENABLE_ARM64
#include <asm/hwcap.h>
#include "common/utils.h"
#include "utils/log_adapter.h"
#include "src/runtime/kernel/arm/nnacl/optimized_kernel.h"
#endif

using mindspore::kernel::kCPU;
using mindspore::kernel::KERNEL_ARCH;
using mindspore::kernel::KernelCreator;
using mindspore::kernel::KernelKey;
using mindspore::kernel::kKernelArch_MAX;
using mindspore::kernel::kKernelArch_MIN;
using mindspore::schema::PrimitiveType_MAX;
using mindspore::schema::PrimitiveType_MIN;

namespace mindspore::lite {
KernelRegistry::KernelRegistry() {
  device_type_length_ = kKernelArch_MAX - kKernelArch_MIN + 1;
  data_type_length_ = kNumberTypeEnd - kNumberTypeBegin + 1;
  op_type_length_ = PrimitiveType_MAX - PrimitiveType_MIN + 1;
  // malloc an array contain creator functions of kernel
  auto total_len = device_type_length_ * data_type_length_ * op_type_length_;
  creator_arrays_ = (kernel::KernelCreator *)malloc(total_len * sizeof(kernel::KernelCreator));
  if (creator_arrays_ == nullptr) {
    MS_LOG(ERROR) << "malloc creator_arrays_ failed.";
  } else {
    for (int i = 0; i < total_len; ++i) {
      creator_arrays_[i] = nullptr;
    }
  }
}

KernelRegistry::~KernelRegistry() { FreeCreatorArray(); }

KernelRegistry *KernelRegistry::GetInstance() {
  static KernelRegistry instance;
  return &instance;
}

int KernelRegistry::Init() {
#ifdef ENABLE_ARM64
  void *optimized_lib_handler = OptimizeModule::GetInstance()->optimized_op_handler_;
  if (optimized_lib_handler != nullptr) {
    MS_LOG(INFO) << "load optimize lib success.";
  } else {
    MS_LOG(INFO) << "load optimize lib failed.";
  }
#endif
  return RET_OK;
}

void KernelRegistry::FreeCreatorArray() {
  if (creator_arrays_ != nullptr) {
    free(creator_arrays_);
    creator_arrays_ = nullptr;
  }
}

kernel::KernelCreator KernelRegistry::GetCreator(const KernelKey &desc) {
  if (creator_arrays_ == nullptr) {
    MS_LOG(ERROR) << "Creator func array is null.";
    return nullptr;
  }
  int index = GetCreatorFuncIndex(desc);
  auto it = creator_arrays_[index];
  if (it != nullptr) {
    return it;
  }
  return nullptr;
}

int KernelRegistry::GetCreatorFuncIndex(const kernel::KernelKey desc) {
  int index;
  int device_index = static_cast<int>(desc.arch);
  int dType_index = static_cast<int>(desc.data_type);
  int op_index = static_cast<int>(desc.type);
  index = device_index * data_type_length_ * op_type_length_ + dType_index * op_type_length_ + op_index;
  return index;
}

void KernelRegistry::RegKernel(const KernelKey desc, kernel::KernelCreator creator) {
  if (creator_arrays_ == nullptr) {
    MS_LOG(ERROR) << "Creator func array is null.";
    return;
  }
  int index = GetCreatorFuncIndex(desc);
  creator_arrays_[index] = creator;
}

void KernelRegistry::RegKernel(const KERNEL_ARCH arch, const TypeId data_type, const schema::PrimitiveType op_type,
                               kernel::KernelCreator creator) {
  if (creator_arrays_ == nullptr) {
    MS_LOG(ERROR) << "Creator func array is null.";
    return;
  }
  KernelKey desc = {arch, data_type, op_type};
  int index = GetCreatorFuncIndex(desc);
  creator_arrays_[index] = creator;
}

bool KernelRegistry::Merge(const std::unordered_map<KernelKey, KernelCreator> &newCreators) { return false; }

const kernel::KernelCreator *KernelRegistry::GetCreatorArrays() { return creator_arrays_; }
}  // namespace mindspore::lite
