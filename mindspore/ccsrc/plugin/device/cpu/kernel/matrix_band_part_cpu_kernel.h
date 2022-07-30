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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MATRIX_BAND_PART_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MATRIX_BAND_PART_CPU_KERNEL_H_
#include <vector>
#include <utility>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class MatrixBandPartCpuKernelMod : public NativeCpuKernelMod {
 public:
  MatrixBandPartCpuKernelMod() = default;
  ~MatrixBandPartCpuKernelMod() override = default;
  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override {
    if (is_null_input_) {
      return true;
    }
    return kernel_func_(this, inputs, outputs);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T, typename LU>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);
  template <typename T, typename LU>
  bool LaunchKernelNotBroadcast(const T *x_ptr, const LU *lower_ptr, const LU *upper_ptr, T *output_ptr);
  template <typename T, typename LU>
  bool LaunchKernelBroadcast(const T *x_ptr, const LU *lower_ptr, const LU *upper_ptr, T *output_ptr);
  void BroadcastShape(const ShapeVector &x_shape, const ShapeVector &lower_shape, const ShapeVector &upper_shape,
                      const ShapeVector &output_shape);
  using MatrixBandPartFunc = std::function<bool(MatrixBandPartCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                                                const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, MatrixBandPartFunc>> func_list_;
  MatrixBandPartFunc kernel_func_;
  bool is_null_input_{false};
  size_t dim_size_{1};
  size_t output_element_num_{0};
  size_t output_outer_size_{1};
  size_t m_{1};
  size_t n_{1};
  size_t lower_{0};
  size_t upper_{0};
  bool need_broadcast_{false};
  ShapeVector broadcast_x_shape_;
  ShapeVector broadcast_lower_shape_;
  ShapeVector broadcast_upper_shape_;
  ShapeVector broadcast_output_shape_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MATRIX_BAND_PART_CPU_KERNEL_H_
