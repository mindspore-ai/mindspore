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

#include <vector>
#include <memory>
#include <string>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
constexpr auto kSliceGrad = "SliceGrad";
constexpr auto kStridedSliceGrad = "StridedSliceGrad";
constexpr auto kUnknown = "Unknown";
class SliceGradCpuKernelMod : public NativeCpuKernelMod {
 public:
  SliceGradCpuKernelMod() = default;
  explicit SliceGradCpuKernelMod(const std::string &kernel_type) : kernel_type_(kernel_type) {}
  ~SliceGradCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);
  template <typename T>
  void CopyDataToOutput(const std::vector<kernel::AddressPtr> &inputs, size_t in_offset,
                        const std::vector<kernel::AddressPtr> &outputs, size_t out_offset, size_t copy_num,
                        int id) const;
  void InitParams(const std::vector<kernel::AddressPtr> &inputs);
  void ClearVectors();
  void ExpandAllMemberDims();
  bool CanCopyMemoryOnAxis(size_t dim) const;
  int SignOfStride(size_t axis) const;
  void FormatArgs(bool stride);
  std::vector<int> begin_;
  std::vector<int> end_;
  std::vector<int> strides_;
  std::vector<int> size_;
  std::vector<size_t> input_shape_;
  std::vector<size_t> input_element_num_;
  std::vector<size_t> output_shape_;
  std::vector<size_t> output_element_num_;
  TypeId dtype_{kTypeUnknown};
  std::string kernel_type_{kUnknown};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SLICE_GRAD_CPU_KERNEL_H_
