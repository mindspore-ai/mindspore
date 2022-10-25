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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_SVD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_SVD_GPU_KERNEL_H_
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <functional>
#include <map>
#include <utility>
#include "mindspore/core/ops/svd.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/transpose_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/matrix_transpose_impl.cuh"

namespace mindspore {
namespace kernel {
class SvdGpuKernelMod : public NativeGpuKernelMod {
 public:
  SvdGpuKernelMod() { ResetResource(); }
  ~SvdGpuKernelMod() = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    cuda_stream_ = stream_ptr;
    return launch_kernel_func_(this, inputs, workspace, outputs);
  }

  void ResetResource() noexcept {
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

  std::vector<KernelAttr> GetOpSupport() override;

 protected:
  void InitSizeLists();

  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);
  template <typename T>
  void RunSvd(const size_t m, const size_t n, const size_t batch, T *d_a, int *dev_info, T *output_s, T *d_output_u,
              T *d_output_v);
  template <typename T>
  void RunSvdBatched(const size_t m, const size_t n, T *d_input, T *output_s, T *output_u, T *output_v, int *dev_info);
  template <typename T>
  void TransposeUV(const size_t m, const size_t n, T *d_output_u, T *d_output_v, T *output_u, T *output_v);
  template <typename T>
  void LaunchSvd(const size_t m, const size_t n, T *d_input, T *output_s, T *output_u, T *output_v, T *d_output_u,
                 T *d_output_v, int *dev_info);
  void CheckResult(int *dev_info);

  using LaunchKernelFunc =
    std::function<bool(SvdGpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;
  using InitSizeListsFunc = std::function<void(SvdGpuKernelMod *)>;
  LaunchKernelFunc launch_kernel_func_{nullptr};
  static std::vector<std::pair<KernelAttr, LaunchKernelFunc>> func_list_;

  size_t unit_size_{1};
  bool compute_uv_{false};
  bool full_matrices_{false};
  std::vector<size_t> input_shape_;
  size_t total_size_{0};
  size_t dims_{0};
  size_t m_{0};
  size_t n_{0};
  size_t p_{0};
  size_t batch_size_{0};
  signed char job_{'N'};
  bool m_ge_n_{false};
  bool batched_{false};

  cusolverDnHandle_t handle_{nullptr};
  void *cuda_stream_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_SVD_GPU_KERNEL_H_
