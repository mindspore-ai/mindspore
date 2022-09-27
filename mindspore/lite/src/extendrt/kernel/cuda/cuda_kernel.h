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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_CUDA_CUDA_KERNEL_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_CUDA_CUDA_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <memory>
#include "src/litert/inner_kernel.h"
#include "src/litert/lite_kernel.h"
#include "cuda_impl/cuda_class/helper_base.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
namespace mindspore::kernel {
class CudaKernel : public InnerKernel {
 public:
  CudaKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
             const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx) {}
  ~CudaKernel() override;
  int Prepare() override {
    type_name_ = std::string(EnumNamePrimitiveType(type()));
    return RET_OK;
  }
  int PreProcess() override;
  int PostProcess() override;
  int ReSize() override;
  int Run() override { return RET_ERROR; }

 protected:
  std::vector<size_t> output_device_size_;
  std::vector<void *> input_device_ptrs_;
  std::vector<void *> output_device_ptrs_;
  std::vector<void *> work_device_ptrs_;
  cudaStream_t stream_;
  std::shared_ptr<cukernel::GpuKernelHelperBase> helper_{nullptr};
  std::string type_name_;
};
template <class T>
kernel::InnerKernel *CudaKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                       const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                       const lite::InnerContext *ctx, const kernel::KernelKey &desc) {
  return new (std::nothrow) T(opParameter, inputs, outputs, ctx);
}
}  // namespace mindspore::kernel
#endif
