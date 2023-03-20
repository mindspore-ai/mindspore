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
#include "src/litert/kernel_registry.h"
#include <utility>
#include <memory>
#include "include/errorcode.h"
#ifndef CUSTOM_KERNEL_REGISTRY_CLIP
#include "include/registry/register_kernel.h"
#endif
#include "src/common/ops/populate/populate_register.h"
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
#include "src/litert/kernel/cpu/nnacl/nnacl_manager.h"
#include "nnacl/kernel.h"

using mindspore::kernel::kBuiltin;
using mindspore::kernel::kCPU;
using mindspore::kernel::KERNEL_ARCH;
using mindspore::kernel::KernelKey;

namespace mindspore::lite {
void KernelRegistry::CreatorArraysInit() {
  std::unique_lock<std::mutex> malloc_creator_array(lock_);
  if (creator_arrays_ == nullptr) {
    creator_arrays_ = reinterpret_cast<kernel::KernelCreator *>(malloc(array_size_ * sizeof(kernel::KernelCreator)));
    if (creator_arrays_ != nullptr) {
      memset(creator_arrays_, 0, array_size_ * sizeof(kernel::KernelCreator));
    }
  }
  if (inner_op_creator_arrays_ == nullptr) {
    inner_op_creator_arrays_ =
      reinterpret_cast<kernel::KernelCreator *>(malloc(inner_op_array_size_ * sizeof(kernel::KernelCreator)));
    if (inner_op_creator_arrays_ != nullptr) {
      memset(inner_op_creator_arrays_, 0, inner_op_array_size_ * sizeof(kernel::KernelCreator));
    }
  }
  return;
}

KernelRegistry *KernelRegistry::GetInstance() {
  static KernelRegistry instance;
  return &instance;
}

kernel::KernelCreator KernelRegistry::GetCreator(const KernelKey &desc) {
  if (desc.format != NHWC) {
    /* nchw kernel using nnacl kernel */
    return nullptr;
  }

  if (desc.provider == kBuiltin) {
    int index = GetCreatorFuncIndex(desc);
    if (desc.type >= PrimType_MIN && desc.type < PrimType_MAX) {
      if (index >= array_size_ || index < 0) {
        MS_LOG(ERROR) << "invalid kernel key, arch " << desc.arch << ", data_type " << desc.data_type << ",op type "
                      << desc.type;
        return nullptr;
      }
      if (creator_arrays_ != nullptr) {
        return creator_arrays_[index];
      }
    } else if (desc.type >= PrimType_InnerOpMin && desc.type < PrimType_InnerOpMax) {
      MS_CHECK_TRUE_RET(index >= 0 && index < inner_op_array_size_, nullptr);
      if (inner_op_creator_arrays_ != nullptr) {
        return inner_op_creator_arrays_[index];
      }
    }
  }
  MS_LOG(ERROR) << "Call wrong interface!provider: " << desc.provider;
  return nullptr;
}

int KernelRegistry::GetCreatorFuncIndex(const kernel::KernelKey desc) {
  int device_index = static_cast<int>(desc.arch) - kKernelArch_MIN;
  int dType_index = desc.data_type == kObjectTypeString ? 0 : static_cast<int>(desc.data_type) - kNumberTypeBegin;
  int op_index = static_cast<int>(desc.type);
  int op_type_length = op_type_length_;
  if (op_index >= PrimType_InnerOpMin && desc.type < PrimType_InnerOpMax) {
    op_type_length = inner_op_type_length_;
    op_index -= PrimType_InnerOpMin;
  }
  int index = device_index * data_type_length_ * op_type_length + dType_index * op_type_length + op_index;
  return index;
}

void KernelRegistry::RegKernel(const KernelKey desc, const kernel::KernelCreator creator) {
  CreatorArraysInit();
  int index = GetCreatorFuncIndex(desc);
  if (desc.type >= PrimType_MIN && desc.type < PrimType_MAX) {
    if (index >= array_size_ || index < 0) {
      MS_LOG(ERROR) << "invalid kernel key, arch " << desc.arch << ", data_type" << desc.data_type << ",op type "
                    << desc.type;
      return;
    }
    if (creator_arrays_ != nullptr) {
      creator_arrays_[index] = creator;
    }
  } else if (desc.type >= PrimType_InnerOpMin && desc.type < PrimType_InnerOpMax) {
    MS_CHECK_TRUE_RET_VOID(index >= 0 && index < inner_op_array_size_);
    if (inner_op_creator_arrays_ != nullptr) {
      inner_op_creator_arrays_[index] = creator;
    }
  }
}

void KernelRegistry::RegKernel(KERNEL_ARCH arch, TypeId data_type, int op_type, kernel::KernelCreator creator) {
  CreatorArraysInit();
  KernelKey desc = {arch, data_type, NHWC, op_type};
  int index = GetCreatorFuncIndex(desc);
  if (desc.type >= PrimType_MIN && desc.type < PrimType_MAX) {
    if (index >= array_size_ || index < 0) {
      MS_LOG(ERROR) << "invalid kernel key, arch " << desc.arch << ", data_type" << desc.data_type << ",op type "
                    << desc.type;
      return;
    }
    if (creator_arrays_ != nullptr) {
      creator_arrays_[index] = creator;
    }
  } else if (desc.type >= PrimType_InnerOpMin && desc.type < PrimType_InnerOpMax) {
    MS_CHECK_TRUE_RET_VOID(index >= 0 && index < inner_op_array_size_);
    if (inner_op_creator_arrays_ != nullptr) {
      inner_op_creator_arrays_[index] = creator;
    }
  }
}

KernelRegistry::~KernelRegistry() {
  KernelRegistry *instance = GetInstance();
  std::unique_lock<std::mutex> malloc_creator_array(instance->lock_);
  if (instance->creator_arrays_ != nullptr) {
    free(instance->creator_arrays_);
    instance->creator_arrays_ = nullptr;
  }
  if (instance->inner_op_creator_arrays_ != nullptr) {
    free(instance->inner_op_creator_arrays_);
    instance->inner_op_creator_arrays_ = nullptr;
  }
}

bool KernelRegistry::SupportKernel(const KernelKey &key) {
  auto kernel_creator = GetCreator(key);
  if (kernel_creator != nullptr) {
    return true;
  }
  return SupportKernelC(key.type, key.data_type);
}

int KernelRegistry::GetCustomKernel(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                                    const mindspore::Context *ms_ctx, const kernel::KernelKey &key,
                                    kernel::KernelExec **kernel, const void *primitive) {
#ifndef CUSTOM_KERNEL_REGISTRY_CLIP
  MS_ASSERT(ms_ctx != nullptr);
  MS_ASSERT(kernel != nullptr);
  registry::KernelDesc desc{static_cast<DataType>(key.data_type), key.type, key.kernel_arch, key.provider};
  auto creator = registry::RegisterKernel::GetCreator(static_cast<const schema::Primitive *>(primitive), &desc);
  if (creator == nullptr) {
    return RET_NOT_SUPPORT;
  }

  auto base_kernel = creator(LiteTensorsToMSTensors(in_tensors), LiteTensorsToMSTensors(out_tensors),
                             static_cast<const schema::Primitive *>(primitive), ms_ctx);
  if (base_kernel != nullptr) {
    auto *kernel_exec = new (std::nothrow) kernel::KernelExec(base_kernel);
    if (kernel_exec != nullptr) {
      constexpr auto kArchCPU = "CPU";
      constexpr auto kArchGPU = "GPU";
      kernel::KernelKey tmp_key = key;
      if (desc.arch == kArchCPU) {
        tmp_key.arch = kernel::kCPU;
      } else if (desc.arch == kArchGPU) {
        tmp_key.arch = kernel::kGPU;
      } else {
        tmp_key.arch = kernel::kCustom;
      }
      kernel_exec->set_desc(tmp_key);
      *kernel = kernel_exec;
      return RET_OK;
    }
  }
#endif
  return RET_ERROR;
}

kernel::LiteKernel *KernelRegistry::GetLiteKernel(const std::vector<Tensor *> &in_tensors,
                                                  const std::vector<Tensor *> &out_tensors, const InnerContext *ctx,
                                                  const kernel::KernelKey &key, OpParameter *parameter) {
  auto creator = GetCreator(key);
  if (creator != nullptr) {
    auto lite_kernel = creator(in_tensors, out_tensors, parameter, ctx, key);
    if (lite_kernel != nullptr) {
      lite_kernel->set_registry_data_type(key.data_type);
      return lite_kernel;
    }
    return nullptr;
  }
  if (key.arch != KERNEL_ARCH::kCPU) {
    return nullptr;
  }

  auto *lite_kernel = nnacl::NnaclKernelRegistry(parameter, in_tensors, out_tensors, ctx, key);
  if (lite_kernel == nullptr) {
    MS_LOG(ERROR) << "Registry cpu kernel failed:  " << parameter->name_;
    return nullptr;
  }
  lite_kernel->set_registry_data_type(key.data_type);
  return lite_kernel;
}

int KernelRegistry::GetKernelExec(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                                  const InnerContext *ctx, const mindspore::Context *ms_ctx,
                                  const kernel::KernelKey &key, OpParameter *parameter, kernel::KernelExec **kernel,
                                  const void *primitive) {
  CHECK_NULL_RETURN(kernel);
#ifndef CUSTOM_KERNEL_REGISTRY_CLIP
  if (key.provider != kBuiltin) {
    CHECK_NULL_RETURN(ms_ctx);
    auto ret = GetCustomKernel(in_tensors, out_tensors, ms_ctx, key, kernel, primitive);
    if (ret == RET_OK) {
      (*kernel)->set_context(ctx);
    }
    return ret;
  }
#endif

  CHECK_NULL_RETURN(ctx);
  auto lite_kernel = GetLiteKernel(in_tensors, out_tensors, ctx, key, parameter);
  if (lite_kernel != nullptr) {
    std::shared_ptr<kernel::Kernel> shared_kernel(lite_kernel);
    auto *kernel_exec = new (std::nothrow) kernel::KernelExec(shared_kernel);
    if (kernel_exec != nullptr) {
      kernel_exec->set_desc(key);
      kernel_exec->set_context(ctx);
      *kernel = kernel_exec;
      return RET_OK;
    }
  }
  MS_LOG(ERROR) << "common cpu kernel registry failed";
  return RET_ERROR;
}

int KernelRegistry::ReplaceKernelExec(kernel::KernelExec *kernel_exec, const kernel::KernelKey &key) {
  CHECK_NULL_RETURN(kernel_exec);
  if (key.provider != kBuiltin) {
    MS_LOG(DEBUG) << "The replace kernel function is only used for inner kernel.";
    return RET_NOT_SUPPORT;
  }
  if (kernel_exec->desc() == key) {
    MS_LOG(DEBUG) << "The kernel " << kernel_exec->name() << " is already be the specific desc.";
    return RET_NO_CHANGE;
  }
  auto op_parameter = kernel_exec->op_parameter();
  auto lite_kernel =
    GetLiteKernel(kernel_exec->in_tensors(), kernel_exec->out_tensors(), kernel_exec->Context(), key, op_parameter);
  if (lite_kernel == nullptr) {
    return RET_ERROR;
  }
  lite_kernel->set_name(kernel_exec->name());
  std::shared_ptr<kernel::Kernel> shared_kernel(lite_kernel);
  kernel_exec->RepalceKernel(shared_kernel);
  kernel_exec->set_desc(key);
  return RET_OK;
}
}  // namespace mindspore::lite
