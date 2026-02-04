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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_ORGQR_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_ORGQR_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <functional>
#include <map>
#include <utility>
#include "mindspore/ops/infer/orgqr.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"
#include "kernel/gpu/cuda_impl/cuda_ops/transpose_impl.cuh"
#include "kernel/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;
class OrgqrGpuKernelMod : public NativeGpuKernelMod {
 public:
  OrgqrGpuKernelMod() { ResetResource(); }
  ~OrgqrGpuKernelMod() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    cuda_stream_ = stream_ptr;
    return launch_kernel_func_(this, inputs, workspace, outputs);
  }
  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  void ResetResource() noexcept {
    is_null_input_ = false;

    output_size_list_.clear();
    workspace_size_list_.clear();
  }

  std::vector<KernelAttr> GetOpSupport() override;

 protected:
  template <typename T>
  void InitSizeLists();

  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);
  template <typename T>
  void RunOrgqr(const size_t m, const size_t n, const size_t k, T *d_a, T *tau, int *dev_info, T *d_output_y);
  template <typename T>
  void LaunchOrgqr(T *d_input_x, T *input_tau, T *output_y, int *dev_info);
  void CheckResult(int *dev_info);

  using LaunchKernelFunc =
    std::function<bool(OrgqrGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &)>;
  using InitSizeListsFunc = std::function<void(OrgqrGpuKernelMod *)>;
  LaunchKernelFunc launch_kernel_func_{nullptr};
  InitSizeListsFunc init_lists_func_{nullptr};
  static std::vector<std::pair<KernelAttr, std::pair<LaunchKernelFunc, InitSizeListsFunc>>> func_list_;

  std::vector<size_t> input_x_shape_;
  std::vector<size_t> input_tau_shape_;
  size_t input_x_dims_{0};
  size_t input_tau_dims_{0};
  size_t m_{0};
  size_t n_{0};
  size_t k_{0};
  size_t batch_size_{0};
  bool is_null_input_;
  size_t transpose_input_x_shape_[transpose_max_dimension] = {0};
  size_t transpose_input_x_axis_[transpose_max_dimension] = {0};
  size_t transpose_output_y_shape_[transpose_max_dimension] = {0};
  cusolverDnHandle_t handle_{nullptr};
  void *cuda_stream_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_ORGQR_GPU_KERNEL_H_
