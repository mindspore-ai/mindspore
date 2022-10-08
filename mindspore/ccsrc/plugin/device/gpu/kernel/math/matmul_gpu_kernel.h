/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_MATMUL_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_MATMUL_GPU_KERNEL_H_

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include <utility>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/fill_impl.cuh"
#include "include/common/utils/convert_utils.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace kernel {
constexpr size_t kDimLowerLimit = 2;
constexpr size_t kDimOffset2 = 2;

class MatMulGpuKernelMod : public NativeGpuKernelMod {
 public:
  MatMulGpuKernelMod() { ResetResource(); }
  explicit MatMulGpuKernelMod(const string kernel_name) : kernel_name_(kernel_name) { ResetResource(); }
  ~MatMulGpuKernelMod() = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    CHECK_CUBLAS_RET_WITH_ERROR(cublasSetStream(handle_, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                "cublasSetStream failed");
    VARIABLE_NOT_USED(workspace);
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  void ResetResource() noexcept {
    batch_ = 0;
    m_ = 0;
    n_ = 0;
    k_ = 0;
    transpose_x1_ = CUBLAS_OP_N;
    transpose_x2_ = CUBLAS_OP_N;
    handle_ = nullptr;
    dtype_a_ = CUDA_R_32F;
    dtype_b_ = CUDA_R_32F;
    dtype_c_ = CUDA_R_32F;
    algo_ = CUBLAS_GEMM_DEFAULT;
    is_fused_matmul_biasadd_ = false;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);

#if CUDA_VERSION >= 11000
  cublasComputeType_t GetComputeType();
#endif

  using MatMulFunc = std::function<bool(MatMulGpuKernelMod *, const std::vector<AddressPtr> &,
                                        const std::vector<AddressPtr> &, const std::vector<AddressPtr> &, void *)>;
  MatMulFunc kernel_func_{};
  static std::map<std::string, std::vector<std::pair<KernelAttr, MatMulGpuKernelMod::MatMulFunc>>> kernel_attr_map_;

  size_t batch_;
  size_t m_;
  size_t n_;
  size_t k_;
  std::string kernel_name_;

  cublasOperation_t transpose_x1_;
  cublasOperation_t transpose_x2_;
  cublasHandle_t handle_;
  cudaDataType_t dtype_a_;
  cudaDataType_t dtype_b_;
  cudaDataType_t dtype_c_;
  cublasGemmAlgo_t algo_;

  bool is_fused_matmul_biasadd_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_MATMUL_GPU_KERNEL_H_
