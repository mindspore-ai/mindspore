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
#include "kernel/graph_kernel/graph_kernel_json_flags.h"
#include "tools/graph_kernel/common/utils.h"
#include "src/tensor.h"
#include "src/common/utils.h"
#include "src/common/tensor_util.h"
#include "src/litert/kernel_registry.h"
#include "schema/model_generated.h"
#include "src/common/dynamic_library_loader.h"
#include "src/common/file_utils.h"

namespace mindspore::kernel {
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
constexpr auto kNumberTwo = 2;
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

int AkgKernel::CheckAkgKernelInfo() {
  std::string current_arch;
#if defined(ENABLE_ARM64)
  current_arch = "aarch64";
#elif defined(ENABLE_ARM)
  current_arch = "arm";
#else
  current_arch = "x86_64";
#endif
  if (current_arch != arch) {
    MS_LOG(ERROR) << "Current cpu arch is " << current_arch << ", but got a " << arch
                  << " AKGKernel. AkgKernel info ckeck failed.";
    return RET_ERROR;
  }
#if defined(ENABLE_AVX512)
  return RET_OK;
#elif defined(ENABLE_AVX)
  if (cpu_feature == "avx512") {
    MS_LOG(ERROR)
      << "Current Runtime not support avx512, but AkgKernel got an avx512 kernel. AkgKernel info ckeck failed.";
    return RET_ERROR;
  }
#elif defined(ENABLE_AVX)
  if (cpu_feature == "avx512" || cpu_feature == "avx") {
    MS_LOG(ERROR) << "Current Runtime not support avx512 and avx, but AkgKernel got an " << cpu_feature
                  << " kernel. AkgKernel info ckeck failed.";
    return RET_ERROR;
  }
#endif
  return RET_OK;
}

void AkgKernel::ExtractKernelAttr() {
  auto prim = static_cast<schema::Primitive *>(params_)->value_as_Custom();
  for (size_t i = 0; i < prim->attr()->size(); i++) {
    auto attr = prim->attr()->Get(i);
    if (attr->name()->str() == "kernel_name") {
      kernel_name_ = std::string(reinterpret_cast<const char *>(attr->data()->Data()), attr->data()->size());
    } else if (attr->name()->str() == "inputs_shape") {
      std::string inputs_shape_str(reinterpret_cast<const char *>(attr->data()->Data()), attr->data()->size());
      (void)graphkernel::GetCustomShape(inputs_shape_str, &origin_inputs_shape_);
    } else if (attr->name()->str() == "dynamic_input_index") {
      dynamic_batch_size_ = 1;
      std::string dynamic_input_index_str(reinterpret_cast<const char *>(attr->data()->Data()), attr->data()->size());
      graphkernel::GetCustomIndex(dynamic_input_index_str, &dynamic_input_index_);
    } else if (attr->name()->str() == mindspore::graphkernel::kJsonKeyProcess) {
      process = std::string(reinterpret_cast<const char *>(attr->data()->Data()), attr->data()->size());
    } else if (attr->name()->str() == mindspore::graphkernel::kJsonKeyArch) {
      arch = std::string(reinterpret_cast<const char *>(attr->data()->Data()), attr->data()->size());
    } else if (attr->name()->str() == mindspore::graphkernel::kJsonKeySystem) {
      system = std::string(reinterpret_cast<const char *>(attr->data()->Data()), attr->data()->size());
    } else if (attr->name()->str() == mindspore::graphkernel::kJsonKeyCpuFeature) {
      cpu_feature = std::string(reinterpret_cast<const char *>(attr->data()->Data()), attr->data()->size());
    } else {
      continue;
    }
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
  if (kernel_func_ != nullptr) {
    return RET_OK;
  }
  if (CheckAkgKernelInfo() != RET_OK) {
    return RET_ERROR;
  }
  if (in_tensors_.size() < kNumberTwo) {
    MS_LOG(ERROR) << "The number of input tensor in AkgKernel must greater than 2, but now got " << in_tensors_.size();
    return lite::RET_INPUT_TENSOR_ERROR;
  }
  auto akg_lib_tensor = in_tensors_.at(in_tensors_.size() - 1);
  auto akg_lib_ptr = akg_lib_tensor->data();
  auto akg_kernel_so = kernel_name_ + ".so";
  std::string kernle_meta = "akg_kernel_meta_runtime";
  if (lite::CreateDir(kernle_meta) != RET_OK) {
    MS_LOG(ERROR) << "cannot create dir " << kernle_meta;
    return lite::RET_ERROR;
  }
  auto akg_kernel_path = kernle_meta + "/" + akg_kernel_so;
  if (lite::WriteToBin(akg_kernel_path, akg_lib_ptr, akg_lib_tensor->Size())) {
    MS_LOG(ERROR) << "write data to " << akg_kernel_so << " failed.";
    return lite::RET_ERROR;
  }
  auto real_path = lite::RealPath(akg_kernel_path.c_str());
  if (real_path.empty()) {
    MS_LOG(ERROR) << "cannot access file:" << real_path << ".please check file if exists and file mod";
    return lite::RET_ERROR;
  }
  lib_handle_ = dlopen(real_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (lib_handle_ == nullptr) {
    MS_LOG(ERROR) << "Load library from tensor failed. Kernel name is  [" << akg_kernel_so << "]";
    return RET_ERROR;
  }
  kernel_func_ = dlsym(lib_handle_, kernel_name_.c_str());
  if (kernel_func_ == nullptr) {
    MS_LOG(ERROR) << "Undefined symbol [" << kernel_name_ << "] in [" << akg_kernel_so << "]";
    return RET_ERROR;
  }
  // the last input tensor is akgkernels.so, so we need to remove it.
  in_tensors_.pop_back();
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
  // callbackfunc and dynamic batch size
  const size_t extra_arg_num_with_batch = 2;
  runtimeargs.reserve(in_tensors_.size() + out_tensors_.size() + extra_arg_num_with_batch);

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
  if (dynamic_batch_size_ != 0) {
    (void)runtimeargs.emplace_back(static_cast<void *>(&dynamic_batch_size_));
  }
  using AkgCpuKernelFunction = void (*)(void *);
  reinterpret_cast<AkgCpuKernelFunction>(kernel_func_)(static_cast<void *>(runtimeargs.data()));
  return RET_OK;
}

int AkgKernel::ReSize() {
  if (in_tensors_.empty() || dynamic_batch_size_ == 0) {
    return mindspore::lite::RET_OK;
  }
  std::vector<TensorC> input_tensorc(in_tensors_.size());
  for (size_t i = 0; i < in_tensors_.size(); i++) {
    int ret = lite::Tensor2TensorC(in_tensors_[i], &input_tensorc[i]);
    if (ret != mindspore::lite::RET_OK) {
      MS_LOG(ERROR) << "Convert Tensor to TensorC failed.";
      return mindspore::lite::RET_ERROR;
    }
  }
  std::vector<const TensorC *> input_tensorc_pointer;
  (void)std::transform(input_tensorc.begin(), input_tensorc.end(), std::back_inserter(input_tensorc_pointer),
                       [](const TensorC &t) { return &t; });
  if (graphkernel::CalculateDynamicBatchSize(&input_tensorc_pointer[0], in_tensors_.size(), origin_inputs_shape_,
                                             dynamic_input_index_, &dynamic_batch_size_) != RET_OK) {
    return mindspore::lite::RET_ERROR;
  }
  return mindspore::lite::RET_OK;
}

AkgKernel::~AkgKernel() {
  if (lib_handle_ != nullptr) {
    (void)dlclose(lib_handle_);
    lib_handle_ = nullptr;
  }
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
