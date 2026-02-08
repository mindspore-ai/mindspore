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
#include <utility>
#include "kernel/cpu/cpu_kernel.h"
#include "primitive/array_ops.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace slice_grad_cpu {
constexpr auto kSliceGrad = "SliceGrad";
constexpr auto kStridedSliceGrad = "StridedSliceGrad";
constexpr auto kUnknown = "Unknown";
constexpr auto kSecondIndex = 2;

class SliceGradCpuKernelMod : public NativeCpuKernelMod {
 public:
  SliceGradCpuKernelMod() = default;

  explicit SliceGradCpuKernelMod(const std::string &kernel_type) : kernel_type_(kernel_type) {}

  ~SliceGradCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &outputs);
  template <typename T>
  void CopyDataToOutput(const std::vector<kernel::KernelTensor *> &inputs, size_t in_offset,
                        const std::vector<kernel::KernelTensor *> &outputs, size_t out_offset, size_t copy_num,
                        int id) const;

  template <typename T>
  void InitParams(const std::vector<kernel::KernelTensor *> &inputs);

  void ClearVectors();

  void ExpandAllMemberDims(size_t expand_dims = 8);

  bool CanCopyMemoryOnAxis(size_t dim, size_t max_dim = 8) const;

  int SignOfStride(size_t axis) const;

  void FormatArgs(bool stride);

  std::pair<bool, bool> GetSliceGradParamValue(const std::vector<KernelTensor *> &inputs, const size_t idx,
                                               const std::string &kernel_name, std::vector<int64_t> *attr_value,
                                               const std::string &param_name, size_t *len);
  template <typename T>
  bool SliceGrad8D(const std::vector<kernel::KernelTensor *> &inputs,
                   const std::vector<kernel::KernelTensor *> &outputs, T *input_addr, T *output_addr);

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
}  // namespace slice_grad_cpu
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SLICE_GRAD_CPU_KERNEL_H_
