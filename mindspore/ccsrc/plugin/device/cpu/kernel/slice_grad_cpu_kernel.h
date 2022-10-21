/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SLICE_GRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SLICE_GRAD_CPU_KERNEL_H_

#include <memory>
#include <string>
#include <vector>
#include <map>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
constexpr auto kSliceGrad = "SliceGrad";
constexpr auto kStridedSliceGrad = "StridedSliceGrad";
constexpr auto kUnknown = "Unknown";
constexpr auto kSecondIndex = 2;
constexpr auto kIndex = 4;

class SliceGradCpuKernelMod : public NativeCpuKernelMod {
 public:
  SliceGradCpuKernelMod() = default;

  explicit SliceGradCpuKernelMod(const std::string &kernel_type) : kernel_type_(kernel_type) {}

  ~SliceGradCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);
  template <typename T>
  void CopyDataToOutput(const std::vector<kernel::AddressPtr> &inputs, size_t in_offset,
                        const std::vector<kernel::AddressPtr> &outputs, size_t out_offset, size_t copy_num,
                        int id) const;

  template <typename T>
  void InitParams(const std::vector<kernel::AddressPtr> &inputs);

  void ClearVectors();

  void ExpandAllMemberDims(size_t expand_dims = 8);

  bool CanCopyMemoryOnAxis(size_t dim, size_t max_dim = 8) const;

  int SignOfStride(size_t axis) const;

  void FormatArgs(bool stride);

  template <typename T>
  bool SliceGrad8D(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs,
                   T *input_addr, T *output_addr);

  size_t begin_len_{0};
  size_t end_len_{0};
  size_t strides_len_{0};
  size_t size_len_{0};
  std::vector<int64_t> begin_;
  std::vector<int64_t> end_;
  std::vector<int64_t> strides_;
  std::vector<int64_t> size_;
  static constexpr size_t kBeginIndex_{2};
  static constexpr size_t kEndIndex_{3};
  static constexpr size_t kStrideIndex_{4};
  static constexpr size_t kSizeIndex_{3};
  ShapeVector input_shape_;
  std::vector<size_t> input_element_num_;
  ShapeVector output_shape_;
  std::vector<size_t> output_element_num_;
  TypeId dtype_{kTypeUnknown};
  TypeId begin_dtype_{kNumberTypeInt32};
  bool get_attr_value_{false};
  std::string kernel_type_{kUnknown};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SLICE_GRAD_CPU_KERNEL_H_
