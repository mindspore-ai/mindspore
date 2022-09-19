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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CUMPROD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CUMPROD_CPU_KERNEL_H_

#include <complex>
#include <memory>
#include <vector>
#include <map>
#include <utility>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class CumProdCpuKernelMod : public NativeCpuKernelMod, public MatchKernelHelper<CumProdCpuKernelMod> {
 public:
  CumProdCpuKernelMod() = default;
  ~CumProdCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  void Reshape();

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
  void CumProdKernelReverse(const T *input, T *output, size_t dim0, size_t dim1, size_t dim2, size_t stride,
                            size_t stride2, size_t start, size_t end) const;

  template <typename T>
  void CumProdKernel(const T *input, T *output, size_t dim0, size_t dim1, size_t dim2, size_t stride, size_t stride2,
                     size_t start, size_t end) const;

  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);

  template <typename T>
  void LaunchCumProd(const T *input_addr, T *output_addr, T *ws_addr, size_t start, size_t end) const;

  ShapeVector shape_;
  ShapeVector dst_shape_;
  size_t input_size_0_{0};
  size_t stride_{0};
  size_t stride2_{0};
  size_t dims_[3]{0};
  int exclusive_{0};
  int reverse_{0};
  int axis_{0};
  TypeId dtype_{kTypeUnknown};
  bool is_dynamic_shape_{false};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CUMPROD_CPU_KERNEL_H_
