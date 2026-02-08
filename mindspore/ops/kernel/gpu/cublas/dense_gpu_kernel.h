/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_DENSE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_DENSE_GPU_KERNEL_H_

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include <utility>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/kernel_constants.h"
#include "kernel/gpu/cuda_impl/cuda_ops/fill_impl.cuh"
#include "utils/convert_utils.h"
#include "include/utils/utils.h"

namespace mindspore {
namespace kernel {
constexpr size_t kDimLowerLimit = 2;
constexpr size_t kDimOffset2 = 2;

class DenseGpuKernelMod : public NativeGpuKernelMod {
 public:
  DenseGpuKernelMod() = default;
  explicit DenseGpuKernelMod(const string kernel_name) : kernel_name_(kernel_name) {}
  ~DenseGpuKernelMod() = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    CHECK_CUBLAS_RET_WITH_ERROR(cublasSetStream(handle_, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                "cublasSetStream failed");
    VARIABLE_NOT_USED(workspace);
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  void ResetSize();
  void ResetWorkspace();
  void ResetResource();

  template <typename T>
  cudaError_t FillBias(T *src, T *dst, cudaStream_t stream);

  template <typename T, typename S>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);

  using DenseFunc =
    std::function<bool(DenseGpuKernelMod *, const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &,
                       const std::vector<KernelTensor *> &, void *)>;
  DenseFunc kernel_func_{};
  static std::map<std::string, std::vector<std::pair<KernelAttr, DenseGpuKernelMod::DenseFunc>>> kernel_attr_map_;

  size_t m_{0};
  size_t n_{0};
  size_t k_{0};
  int lda_{0};
  int ldb_{0};
  int ldc_{0};
  std::string kernel_name_;
  bool is_empty_tensor_ = false;

  cublasHandle_t handle_{nullptr};
  cudaDataType_t dtype_a_{CUDA_R_32F};
  cudaDataType_t dtype_b_{CUDA_R_32F};
  cudaDataType_t dtype_c_{CUDA_R_32F};
#if CUDA_VERSION >= 11000
  cublasComputeType_t compute_type_{CUBLAS_COMPUTE_32F};
#else
  cudaDataType_t compute_type_{CUDA_R_32F};
#endif
  cublasGemmAlgo_t algo_{CUBLAS_GEMM_DEFAULT};

  bool has_bias_{true};
  bool need_cast_{false};
  size_t x_size_{0};
  size_t w_size_{0};
  size_t b_size_{0};
  size_t out_size_{0};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_DENSE_GPU_KERNEL_H_
