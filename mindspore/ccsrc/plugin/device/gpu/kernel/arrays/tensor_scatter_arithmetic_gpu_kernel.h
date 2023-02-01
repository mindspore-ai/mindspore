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

#include <vector>
#include <algorithm>
#include <string>
#include <map>
#include <utility>
#include <memory>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/tensor_scatter_arithmetic.cuh"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
class TensorScatterArithmeticGpuKernelMod : public NativeGpuKernelMod,
                                            public MatchKernelHelper<TensorScatterArithmeticGpuKernelMod> {
 public:
  TensorScatterArithmeticGpuKernelMod() = default;
  ~TensorScatterArithmeticGpuKernelMod() override { FreeResource(); }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    stream_ptr_ = stream_ptr;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  void FreeResource();
  bool GetOpType(const BaseOperatorPtr &base_operator);
  void UpdateSize();
  template <typename S>
  void CheckIndicesValid(S *indices);
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);
  using SupportList = std::vector<std::pair<KernelAttr, TensorScatterArithmeticGpuKernelMod::KernelRunFunc>>;

  bool memcpy_flag_{false};
  size_t input_size_{1};
  size_t update_size_{1};
  size_t output_size_{1};
  size_t block_size_{1};
  size_t indices_dim_0_{0};
  size_t indices_dim_1_{0};
  size_t data_unit_size_{0};
  size_t indices_unit_size_{0};
  TensorScatterArithmeticFunctionType op_func_type_{TENSOR_SCATTER_FUNC_INVALID_TYPE};
  std::vector<size_t> update_shape_;
  std::vector<size_t> indices_shape_;
  std::vector<size_t> input_shape_;
  std::vector<size_t> output_shape_;
  std::vector<size_t> vec_indices_stride_;
  std::vector<size_t> vec_work_shape_;
  void *indices_stride_{nullptr};
  void *work_shape_{nullptr};
  void *stream_ptr_{nullptr};
  size_t slice_size_{1};
  size_t batch_size_{1};
  size_t inner_size_{1};
  size_t total_batch_size_{1};
  std::vector<size_t> batch_strides_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_TENSOR_TENSOR_SCATTER_ARITHMETIC_GPU_KERNEL_H
