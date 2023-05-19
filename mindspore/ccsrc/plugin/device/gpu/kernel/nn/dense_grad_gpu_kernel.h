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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_DENSE_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_DENSE_GRAD_GPU_KERNEL_H_

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

class DenseGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  DenseGradGpuKernelMod() = default;
  explicit DenseGradGpuKernelMod(const string kernel_name) : kernel_name_(kernel_name) {}
  ~DenseGradGpuKernelMod() = default;

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

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);

#if CUDA_VERSION >= 11000
  cublasComputeType_t GetComputeType();
#endif

  using DenseGradFunc = std::function<bool(DenseGradGpuKernelMod *, const std::vector<AddressPtr> &,
                                           const std::vector<AddressPtr> &, const std::vector<AddressPtr> &, void *)>;
  DenseGradFunc kernel_func_{};
  static std::vector<std::pair<KernelAttr, DenseGradGpuKernelMod::DenseGradFunc>> kernel_attr_vec_;

  int m_{0};
  int n_{0};
  int k_{0};
  int lda_{0};
  int ldb_{0};
  int ldc_{0};
  std::string kernel_name_;

  cublasHandle_t handle_{nullptr};
  cudaDataType_t dtype_x_{CUDA_R_32F};
  cudaDataType_t dtype_w_{CUDA_R_32F};
  cudaDataType_t dtype_dout_{CUDA_R_32F};
  cudaDataType_t dtype_dx_{CUDA_R_32F};
  cudaDataType_t dtype_dw_{CUDA_R_32F};
#if CUDA_VERSION >= 11000
  cublasComputeType_t compute_type_{CUBLAS_COMPUTE_32F};
#else
  cudaDataType_t compute_type_{CUDA_R_32F};
#endif
  cublasGemmAlgo_t algo_{CUBLAS_GEMM_DEFAULT};

  bool has_bias_{true};

  void InitResource() override;
  void InitSizeLists();

  cudnnHandle_t cudnn_handle_{nullptr};
  cudnnTensorDescriptor_t dy_desc_{nullptr};
  cudnnTensorDescriptor_t db_desc_{nullptr};
  cudnnReduceTensorDescriptor_t op_desc_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_DENSE_GRAD_GPU_KERNEL_H_
