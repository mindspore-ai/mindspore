/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_MATRIX_INVERSE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_MATRIX_INVERSE_GPU_KERNEL_H_
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <type_traits>
#include <map>
#include <utility>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"

namespace mindspore {
namespace kernel {
class MatrixInverseGpuKernelMod : public NativeGpuKernelMod {
 public:
  MatrixInverseGpuKernelMod() = default;
  ~MatrixInverseGpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs, void *stream_ptr);
  template <typename T>
  void LaunchKernel_Cublas(const std::vector<kernel::AddressPtr> &inputs,
                           const std::vector<kernel::AddressPtr> &workspace,
                           const std::vector<kernel::AddressPtr> &outputs, void *stream_ptr);
  template <typename T>
  void LaunchKernel_CuSolve(const std::vector<kernel::AddressPtr> &inputs,
                            const std::vector<kernel::AddressPtr> &workspace,
                            const std::vector<kernel::AddressPtr> &outputs, void *stream_ptr);
  using MatrixInverseFunc =
    std::function<bool(MatrixInverseGpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &, void *)>;

  static std::vector<std::pair<KernelAttr, MatrixInverseFunc>> func_list_;
  MatrixInverseFunc kernel_func_{nullptr};

  void InitSizeLists();

  size_t input_size_{1};
  bool adjoint_{false};
  cublasHandle_t handle_;
  cusolverDnHandle_t handle_cus;
  size_t batch_size_{1};
  size_t size_{1};
  size_t dtype_size_{1};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_MATRIX_INVERSE_GPU_KERNEL_H_
