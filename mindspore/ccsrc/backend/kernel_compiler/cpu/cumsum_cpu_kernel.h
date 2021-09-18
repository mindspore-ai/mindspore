/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CUMSUM_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CUMSUM_CPU_KERNEL_H_

#include <memory>
#include <vector>

#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class CumSumCPUKernel : public CPUKernel {
 public:
  CumSumCPUKernel() = default;
  ~CumSumCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  void Reshape();

  template <typename T>
  void InitWorkspaceSize();

  void InitInputOutputSize(const CNodePtr &kernel_node) override;

  template <typename T>
  void LeftMove(const T *input, T *output, size_t dim0, size_t dim1, size_t dim2, size_t stride, size_t stride2,
                size_t start, size_t end) const;

  template <typename T>
  void RightMove(const T *input, T *output, size_t dim0, size_t dim1, size_t dim2, size_t stride, size_t stride2,
                 size_t start, size_t end) const;

  template <typename T>
  void Copy(T *input, T *output, size_t dim0, size_t dim1, size_t dim2, size_t stride, size_t stride2, size_t start,
            size_t end) const;

  template <typename T>
  void CumSumKernelReverse(const T *input, T *output, size_t dim0, size_t dim1, size_t dim2, size_t stride,
                           size_t stride2, size_t start, size_t end) const;

  template <typename T>
  void CumSumKernel(const T *input, T *output, size_t dim0, size_t dim1, size_t dim2, size_t stride, size_t stride2,
                    size_t start, size_t end) const;

  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs) const;

  template <typename T>
  void LaunchCumSum(const T *input_addr, T *output_addr, T *ws_addr, size_t start, size_t end) const;

  std::vector<size_t> shape_;
  std::vector<size_t> dst_shape;
  size_t input_size_0_{0};
  size_t stride_{0};
  size_t stride2_{0};
  size_t dims_[3]{0};
  int exclusive_{0};
  int reverse_{0};
  int axis_{0};
  TypeId dtype_{kTypeUnknown};
};

MS_REG_CPU_KERNEL(CumSum, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  CumSumCPUKernel);
MS_REG_CPU_KERNEL(CumSum, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                  CumSumCPUKernel);
MS_REG_CPU_KERNEL(CumSum, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), CumSumCPUKernel);
MS_REG_CPU_KERNEL(CumSum, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8), CumSumCPUKernel);
MS_REG_CPU_KERNEL(CumSum, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8), CumSumCPUKernel);
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CUMSUM_CPU_KERNEL_H_
