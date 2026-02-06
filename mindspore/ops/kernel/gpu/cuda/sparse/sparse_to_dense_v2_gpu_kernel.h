/**
 * Copyright 2022-2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_SPARSE_SPARSE_TO_DENSE_V2_GPU_KERNEL_H_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_SPARSE_SPARSE_TO_DENSE_V2_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>
#include <map>
#include "mindspore/ops/infer/sparse_to_dense_v2.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/cuda_common.h"
#include "kernel/gpu/cuda_impl/cuda_ops/sparse_to_dense_impl.cuh"

namespace mindspore {
namespace kernel {
class SparseToDenseV2GpuKernelMod : public NativeGpuKernelMod {
 public:
  SparseToDenseV2GpuKernelMod() { ResetResource(); }
  ~SparseToDenseV2GpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  void ResetResource() noexcept;
  template <typename I, typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);
  template <typename I, typename T>
  void CheckValidateOneDim(const std::vector<kernel::KernelTensor *> &inputs,
                           const std::vector<kernel::KernelTensor *> &workspace,
                           const std::vector<kernel::KernelTensor *> &outputs);
  template <typename I, typename T>
  void CheckValidateTwoDim(const std::vector<kernel::KernelTensor *> &inputs,
                           const std::vector<kernel::KernelTensor *> &workspace,
                           const std::vector<kernel::KernelTensor *> &outputs);

  using SparseToDenseV2LaunchFunc =
    std::function<bool(SparseToDenseV2GpuKernelMod *, const std::vector<KernelTensor *> &,
                       const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &, void *)>;
  static std::vector<std::pair<KernelAttr, SparseToDenseV2LaunchFunc>> func_list_;
  SparseToDenseV2LaunchFunc kernel_func_{};
  size_t value_size_{1};
  size_t indices_num_{0};
  size_t values_num_{0};
  size_t output_num_{0};
  size_t output_elements{0};
  int ndims{0};
  int num_elems{0};
  bool is_null_input_{false};
  void *cuda_stream_{nullptr};
  bool validate_indices_{true};
  std::vector<size_t> indices_shape_{};
  std::vector<size_t> output_shape_{};
  size_t indices_dims_{0};
  size_t values_size_{0};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_SPARSE_SPARSE_TO_DENSE_V2_GPU_KERNEL_H_
