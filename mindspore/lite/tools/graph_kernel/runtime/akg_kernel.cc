/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "tools/graph_kernel/runtime/akg_kernel.h"
#include <dlfcn.h>
#include <algorithm>
#include <utility>
#include <numeric>
#include <functional>
#include "src/tensor.h"
#include "src/common/utils.h"
#include "src/litert/kernel_registry.h"
#include "schema/model_generated.h"

namespace mindspore::kernel {
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
constexpr auto kAkgKernelSo = "akgkernels.so";
namespace {
int TmpAkgParallelLaunchFunc(AkgParallelLambda flambda, void *cdata, int num_task) {
  /*
  The `cdata` is a second-level pointer, which first element is a pointer to a structure object.
  The structure contains original AkgCallBack's elements, but except the first `parallel_launch_func`.
  It seems like `{malloc_func, free_func, extend_data}`, all elements are also pointers.
  So, to get the `extend_data`, we can treat the `cdata` as a third-level pointer,
  and then offset TWO elements for the first structure object.
  The `extend_data` was set as the `this` pointer of `AkgKernel` object.
  */
  const auto kExtendDataOffset = 2;
  void *extend_data = static_cast<void ***>(cdata)[0][kExtendDataOffset];
  static_cast<AkgKernel *>(extend_data)->AkgParallelLaunchFunc(flambda, cdata, num_task);
  return 0;
}

class AkgCallBack {
 public:
  void *parallel_launch_func = nullptr;
  void *(*malloc_func)(size_t) = nullptr;
  void (*free_func)(void *) = nullptr;
  void *extend_data = nullptr;

  AkgCallBack() {
    parallel_launch_func = reinterpret_cast<void *>(TmpAkgParallelLaunchFunc);
    malloc_func = &malloc;
    free_func = &free;
  }
  ~AkgCallBack() = default;
};
}  // namespace

void AkgKernel::ExtractKernelName() {
  auto prim = static_cast<schema::Primitive *>(params_)->value_as_Custom();
  for (size_t i = 0; i < prim->attr()->size(); i++) {
    auto attr = prim->attr()->Get(i);
    if (attr->name()->str() == "kernel_name") {
      auto data = attr->data();
      kernel_name_ = std::string(reinterpret_cast<const char *>(data->Data()), data->size());
      break;
    }
  }
}

AkgKernel::~AkgKernel() {
  if (handle_ != nullptr) {
    (void)dlclose(handle_);
  }
}

int TmpDoTask(void *obj, int task_id, float lhs_scale, float rhs_scale) {
  return static_cast<AkgKernel *>(obj)->DoTask(task_id, lhs_scale, rhs_scale);
}

int AkgKernel::DoTask(int task_id, float, float) {
  (void)cached_akg_lambda_(task_id, nthread_, cached_runtimeargs_);
  return RET_OK;
}

void AkgKernel::AkgParallelLaunchFunc(AkgParallelLambda flambda, void *cdata, int) {
  cached_akg_lambda_ = flambda;
  cached_runtimeargs_ = cdata;
  (void)ParallelLaunch(this->ms_context_, TmpDoTask, this, this->nthread_);
  cached_akg_lambda_ = nullptr;
  cached_runtimeargs_ = nullptr;
}

int AkgKernel::Prepare() {
  if (handle_ != nullptr || kernel_func_ != nullptr) {
    return RET_OK;
  }
  handle_ = dlopen(kAkgKernelSo, RTLD_LAZY | RTLD_LOCAL);
  if (handle_ == nullptr) {
    MS_LOG(ERROR) << "Load [" << kAkgKernelSo << "] failed. kernel: [" << kernel_name_ << "]";
    return RET_ERROR;
  }
  kernel_func_ = dlsym(handle_, kernel_name_.c_str());
  if (kernel_func_ == nullptr) {
    MS_LOG(ERROR) << "Undefined symbol [" << kernel_name_ << "] in [" << kAkgKernelSo << "]";
    return RET_ERROR;
  }
  const size_t kAddrAlign = 32;
  const size_t kAddrAlignMask = 0x1f;
  const_inputs_.reserve(in_tensors_.size());
  for (auto &input : in_tensors_) {
    // the data address should align in 32 bytes.
    if (input->IsConst() && (reinterpret_cast<size_t>(input->data()) & kAddrAlignMask) != 0) {
      auto buffer = static_cast<int8_t *>(input->data());
      int tensor_size = input->Size();
      if (tensor_size == 0) {
        MS_LOG(ERROR) << "The tensor \'" << input->tensor_name() << "\' size is 0. kernel: " << kernel_name_;
        return RET_ERROR;
      }
      std::vector<int8_t> input_align(tensor_size + kAddrAlign);
      auto p = input_align.data();
      while ((reinterpret_cast<size_t>(p) & kAddrAlignMask) != 0) {
        p++;
      }
      (void)std::copy(buffer, buffer + tensor_size, p);
      (void)const_inputs_.emplace_back(static_cast<void *>(p));
      (void)const_data_align_cache_.emplace_back(std::move(input_align));
    } else {
      (void)const_inputs_.emplace_back(nullptr);
    }
  }
  return RET_OK;
}

int AkgKernel::Run() {
  if (kernel_func_ == nullptr) {
    MS_LOG(ERROR) << "Kernel function [" << kernel_name_ << "] is nullptr.";
    return RET_ERROR;
  }
  nthread_ = op_parameter_->thread_num_;
  std::vector<void *> runtimeargs;
  runtimeargs.reserve(in_tensors_.size() + out_tensors_.size() + 1);
  static AkgCallBack akg_callback;
  akg_callback.extend_data = static_cast<void *>(this);
  (void)runtimeargs.emplace_back(static_cast<void *>(&akg_callback));
  for (size_t i = 0; i < in_tensors_.size(); i++) {
    if (const_inputs_[i] != nullptr) {
      (void)runtimeargs.emplace_back(const_inputs_[i]);
    } else {
      (void)runtimeargs.emplace_back(in_tensors_[i]->data());
    }
  }
  (void)std::transform(std::begin(out_tensors_), std::end(out_tensors_), std::back_inserter(runtimeargs),
                       [](lite::Tensor *output) { return output->MutableData(); });
  using AkgCpuKernelFunction = void (*)(void *);
  reinterpret_cast<AkgCpuKernelFunction>(kernel_func_)(static_cast<void *>(runtimeargs.data()));
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeBool, PrimType_Inner_GraphKernel, LiteKernelCreator<AkgKernel>)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimType_Inner_GraphKernel, LiteKernelCreator<AkgKernel>)
REG_KERNEL(kCPU, kNumberTypeInt16, PrimType_Inner_GraphKernel, LiteKernelCreator<AkgKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimType_Inner_GraphKernel, LiteKernelCreator<AkgKernel>)
REG_KERNEL(kCPU, kNumberTypeInt64, PrimType_Inner_GraphKernel, LiteKernelCreator<AkgKernel>)
REG_KERNEL(kCPU, kNumberTypeUInt8, PrimType_Inner_GraphKernel, LiteKernelCreator<AkgKernel>)
REG_KERNEL(kCPU, kNumberTypeUInt16, PrimType_Inner_GraphKernel, LiteKernelCreator<AkgKernel>)
REG_KERNEL(kCPU, kNumberTypeUInt32, PrimType_Inner_GraphKernel, LiteKernelCreator<AkgKernel>)
REG_KERNEL(kCPU, kNumberTypeUInt64, PrimType_Inner_GraphKernel, LiteKernelCreator<AkgKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimType_Inner_GraphKernel, LiteKernelCreator<AkgKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimType_Inner_GraphKernel, LiteKernelCreator<AkgKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat64, PrimType_Inner_GraphKernel, LiteKernelCreator<AkgKernel>)
}  // namespace mindspore::kernel
