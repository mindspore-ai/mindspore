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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_TENSOR_TENSOR_SCATTER_ARITHMETIC_GPU_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_TENSOR_TENSOR_SCATTER_ARITHMETIC_GPU_KERNEL_H

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "kernel/gpu/cuda_impl/cuda_ops/tensor_scatter_arithmetic.cuh"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class TensorScatterArithmeticGpuKernelMod : public NativeGpuKernelMod,
                                            public MatchKernelHelper<TensorScatterArithmeticGpuKernelMod> {
 public:
  TensorScatterArithmeticGpuKernelMod() = default;
  ~TensorScatterArithmeticGpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    stream_ptr_ = stream_ptr;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  bool GetOpType();
  void UpdateSize();

  template <typename T, typename S>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);
  using SupportList = std::vector<std::pair<KernelAttr, TensorScatterArithmeticGpuKernelMod::KernelRunFunc>>;

  size_t input_size_{1};
  size_t update_size_{1};
  size_t output_size_{1};
  size_t block_size_{1};
  size_t indices_dim_0_{0};
  size_t indices_dim_1_{0};
  size_t data_unit_size_{0};
  size_t indices_unit_size_{0};
  TensorScatterArithmeticFunctionType op_func_type_{TENSOR_SCATTER_FUNC_INVALID_TYPE};
  std::vector<int64_t> update_shape_;
  std::vector<int64_t> indices_shape_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;
  std::vector<int64_t> vec_indices_stride_;
  void *stream_ptr_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_TENSOR_TENSOR_SCATTER_ARITHMETIC_GPU_KERNEL_H
