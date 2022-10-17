/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SEGMENT_MAX_MIN_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SEGMENT_MAX_MIN_CPU_KERNEL_H_

#include <map>
#include <vector>
#include <memory>
#include <string>
#include <complex>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class SegmentMaxMinCPUKernelMod : public NativeCpuKernelMod {
 public:
  SegmentMaxMinCPUKernelMod() = default;
  ~SegmentMaxMinCPUKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T1, typename T2>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);

  template <typename T>
  bool GetComputeFunc();

  template <typename T>
  T GetInitValue();

  using SegmentMaxMinFunc =
    std::function<bool(SegmentMaxMinCPUKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, SegmentMaxMinFunc>> func_list_;
  using SegmentComputeFunc = std::function<void(void *output_addr, void *input_addr)>;
  SegmentComputeFunc compute_func_;
  SegmentMaxMinFunc kernel_func_;

  std::string kernel_name_{};
  ShapeVector input_x_shape_;
  ShapeVector segment_ids_shape_;
  ShapeVector output_shape_;
  size_t input_x_num_{0};
  size_t segment_ids_num_{0};
  size_t output_num_{0};
  TypeId input_x_dtype_{kTypeUnknown};
  TypeId segment_ids_dtype_{kTypeUnknown};
  TypeId output_dtype_{kTypeUnknown};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SEGMENT_MAXMIN_CPU_KERNEL_H_
