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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_RESHAPE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_RESHAPE_CPU_KERNEL_H_

#include <map>
#include <utility>
#include <vector>
#include <memory>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class SparseReshapeCpuKernelMod : public NativeCpuKernelMod {
 public:
  SparseReshapeCpuKernelMod() = default;
  ~SparseReshapeCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  };

  template <typename I, typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  using SparseReshapeFunc = std::function<bool(SparseReshapeCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                                               const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, SparseReshapeFunc>> func_list_;
  SparseReshapeFunc kernel_func_;
  std::vector<int64_t> indices_shape_;
  bool SameConvert(int64_t input_size, int64_t output_size, int64_t input_rank, int64_t output_rank, int64_t *in0,
                   int64_t *in1, int64_t *out0, int64_t *out1);
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_RESHAPE_CPU_KERNEL_H_
