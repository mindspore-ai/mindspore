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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ADAPTIVE_MAX_POOL_3D_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ADAPTIVE_MAX_POOL_3D_CPU_KERNEL_H_

#include <map>
#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>
#include <memory>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class AdaptiveMaxPool3DCpuKernelMod : public NativeCpuKernelMod {
 public:
  AdaptiveMaxPool3DCpuKernelMod() = default;
  ~AdaptiveMaxPool3DCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;
  void SyncOutputShape() override;

 private:
  int64_t ComputeStride(const std::vector<int64_t> &shape, size_t index) const;
  template <typename T>
  void AdaptiveMaxPool3DCompute(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
  template <typename T>
  void ComputeKernel(T *input_data, T *output_data, int32_t *indices_data, int64_t start_T, int64_t end_T) const;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;
  const size_t dimB = 0;
  const size_t dimD = 1;
  const size_t dimT = 2;
  const size_t dimH = 3;
  const size_t dimW = 4;
  const size_t kInputNum = 2;
  const size_t kOutputNum = 2;
  int64_t size_B_ = 0;
  int64_t size_D_ = 0;
  int64_t input_size_T_ = 0;
  int64_t input_size_H_ = 0;
  int64_t input_size_W_ = 0;
  int64_t input_stride_B_ = 0;
  int64_t input_stride_D_ = 0;
  int64_t input_stride_T_ = 0;
  int64_t input_stride_H_ = 0;
  int64_t input_stride_W_ = 0;
  int64_t output_size_T_ = 0;
  int64_t output_size_H_ = 0;
  int64_t output_size_W_ = 0;
  size_t input_num_dims_ = 0;
  TypeId dtype_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ADAPTIVE_MAX_POOL_3D_CPU_KERNEL_H_
