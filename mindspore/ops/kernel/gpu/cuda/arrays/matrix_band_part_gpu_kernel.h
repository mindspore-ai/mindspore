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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_MATRIX_BAND_PART_GPU_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_MATRIX_BAND_PART_GPU_KERNEL_H

#include <vector>
#include <algorithm>
#include <utility>
#include <map>
#include "kernel/gpu/cuda_impl/cuda_ops/matrix_band_part_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/cuda_common.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
class MatrixBandPartGpuKernelMod : public NativeGpuKernelMod {
 public:
  MatrixBandPartGpuKernelMod() = default;
  ~MatrixBandPartGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override {
    if (is_null_input_) {
      return true;
    }
    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T, typename LU>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &outputs);
  template <typename T, typename LU>
  bool LaunchKernelNotBroadcast(const T *x_ptr, const LU *lower_ptr, const LU *upper_ptr, T *output_ptr);
  void BroadcastShape(const std::vector<int64_t> &x_shape, const std::vector<int64_t> &lower_shape,
                      const std::vector<int64_t> &upper_shape, const std::vector<int64_t> &output_shape);
  using MatrixBandPartFunc =
    std::function<bool(MatrixBandPartGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &)>;
  static std::vector<std::pair<KernelAttr, MatrixBandPartFunc>> func_list_;
  MatrixBandPartFunc kernel_func_;
  void *cuda_stream_{nullptr};
  bool is_null_input_{false};
  size_t dim_size_{1};
  size_t output_element_num_{0};
  size_t output_outer_size_{1};
  size_t m_{1};
  size_t n_{1};
  int64_t lower_{0};
  int64_t upper_{0};
  bool need_broadcast_;
  std::vector<int64_t> broadcast_x_shape_;
  std::vector<int64_t> broadcast_lower_shape_;
  std::vector<int64_t> broadcast_upper_shape_;
  std::vector<int64_t> broadcast_output_shape_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_MATRIX_BAND_PART_GPU_KERNEL_H
