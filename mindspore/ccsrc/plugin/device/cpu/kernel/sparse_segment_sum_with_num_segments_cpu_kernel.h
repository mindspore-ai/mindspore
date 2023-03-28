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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_SEGMENT_SUM_WITH_NUM_SGEMENTS_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_SEGMENT_SUM_WITH_NUM_SGEMENTS_CPU_KERNEL_H_

#include <map>
#include <utility>
#include <functional>
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class SparseSegmentSumWithNumSegmentsCpuKernelMod : public NativeCpuKernelMod {
 public:
  SparseSegmentSumWithNumSegmentsCpuKernelMod() = default;
  ~SparseSegmentSumWithNumSegmentsCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  template <typename T1, typename T2>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  std::vector<KernelAttr> GetOpSupport() override {
    std::vector<KernelAttr> kernel_attr_list;
    (void)std::transform(f_list_.begin(), f_list_.end(), std::back_inserter(kernel_attr_list),
                         [](const std::pair<KernelAttr, LaunchKernelFunc> &pair) { return pair.first; });
    return kernel_attr_list;
  }

  using LaunchKernelFunc =
    std::function<void(SparseSegmentSumWithNumSegmentsCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &)>;

 private:
  static std::vector<std::pair<KernelAttr, LaunchKernelFunc>> f_list_;
  LaunchKernelFunc kernel_func_;
  ShapeVector x_shape_;
  ShapeVector segment_ids_shape_;
  ShapeVector y_shape_;
  TypeId x_dtype_{kTypeUnknown};
  TypeId indices_dtype_{kTypeUnknown};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_SEGMENT_SUM_WITH_NUM_SGEMENTS_CPU_KERNEL_H_
