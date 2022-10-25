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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_REORDER_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_REORDER_CPU_KERNEL_H_

#include <memory>
#include <unordered_map>
#include <vector>
#include <map>
#include <complex>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
using float_complex = std::complex<float>;
using double_complex = std::complex<double>;
template <typename I, typename T>
struct SparseGradient {
  T *value_{nullptr};
  I *indices_{nullptr};
  int64_t dims_size_{0};
  bool operator()(const int64_t x, const int64_t y) const {
    bool value = false;
    for (auto di = 0; di < dims_size_; ++di) {
      if (indices_[x * dims_size_ + di] < indices_[y * dims_size_ + di]) {
        value = true;
        break;
      }
      if (indices_[x * dims_size_ + di] > indices_[y * dims_size_ + di]) {
        break;
      }
    }
    return value;
  }
};

class SparseReorderCpuKernelMod : public NativeCpuKernelMod {
 public:
  SparseReorderCpuKernelMod() = default;
  ~SparseReorderCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename I, typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);
  using SparseReorderFunc =
    std::function<bool(SparseReorderCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, SparseReorderFunc>> func_list_;
  SparseReorderFunc kernel_func_;
  std::vector<size_t> indices_shape_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_REORDER_CPU_KERNEL_H_
