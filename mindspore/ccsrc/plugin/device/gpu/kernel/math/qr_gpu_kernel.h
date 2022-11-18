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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_QR_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_QR_GPU_KERNEL_H_

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <complex>
#include <memory>
#include <functional>
#include <cusolverDn.h>
#include "mindspore/core/ops/qr.h"
#include "abstract/utils.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/transpose_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/tril_triu_impl.cuh"

namespace mindspore {
namespace kernel {
class QrGpuKernelMod : public NativeGpuKernelMod {
 public:
  QrGpuKernelMod() { ResetResource(); }
  ~QrGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *cuda_stream) override {
    if (is_null_input_) {
      return true;
    }
    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, workspace, outputs);
  };

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

 protected:
  void ResetResource() noexcept {
    total_size_ = 0;
    input_dims_ = 0;
    m_ = 0;
    n_ = 0;
    p_ = 0;
    s_ = 0;
    batch_size_ = 1;
    is_null_input_ = false;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  };

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);

  template <typename T>
  void RunQr(T *d_input, T *d_A, T *d_tau, int *dev_info, T *d_output_q, T *d_output_r);

  template <typename T>
  void LaunchQr(T *d_input, T *d_A, T *d_tau, T *d_output_q, T *d_output_r, int *dev_info, size_t *d_transpose_shape,
                size_t *d_transpose_axis, T *d_output_r_t, T *output_r);

  using LaunchKernelFunc =
    std::function<bool(QrGpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;

 private:
  size_t unit_input_size_{1};
  size_t total_size_{0};
  size_t input_dims_{0};
  size_t m_{0};
  size_t n_{0};
  size_t p_{0};
  size_t s_{0};
  size_t batch_size_{1};
  bool full_matrices_{false};
  size_t transpose_input_shape_[TRANSPOSE_MAX_DIMENSION] = {0};
  size_t transpose_input_axis_[TRANSPOSE_MAX_DIMENSION] = {0};
  size_t transpose_q_shape_[TRANSPOSE_MAX_DIMENSION] = {0};
  bool is_null_input_{false};
  cusolverDnHandle_t cusolverH_{nullptr};
  void *cuda_stream_{nullptr};
  LaunchKernelFunc kernel_func_{nullptr};
  static std::vector<std::pair<KernelAttr, LaunchKernelFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_QR_GPU_KERNEL_H_
