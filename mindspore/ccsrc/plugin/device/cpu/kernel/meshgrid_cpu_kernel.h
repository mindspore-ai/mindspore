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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MESHGRID_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MESHGRID_CPU_KERNEL_H_

#include <vector>
#include <utility>
#include <map>
#include <complex>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "nnacl/base/broadcast_to.h"

namespace mindspore {
namespace kernel {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

class MeshgridCpuKernelMod : public NativeCpuKernelMod {
 public:
  MeshgridCpuKernelMod() = default;
  ~MeshgridCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  void Mul(const T *input1, const T *input2, T *out);
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);
  using MeshgridFunc = std::function<bool(MeshgridCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                                          const std::vector<AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, MeshgridFunc>> func_list_;
  MeshgridFunc kernel_func_;

  bool swap_indexing_{true};
  std::vector<int64_t> input_shape_{};
  std::vector<int64_t> output_shape_{};
  size_t output_element_ = 0;
  size_t unit_size_ = 0;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MESHGRID_CPU_KERNEL_H_
