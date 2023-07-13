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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SLICE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SLICE_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <map>
#include "kernel/kernel_get_value.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "nnacl/base/slice_base.h"
#include "nnacl/kernel/slice.h"

namespace mindspore {
namespace kernel {
class SliceCpuKernelMod : public NativeCpuKernelMod {
 public:
  SliceCpuKernelMod() = default;
  ~SliceCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  void InitSliceParam(const ShapeVector &input_shape, const std::vector<int64_t> &begin,
                      const std::vector<int64_t> &size);
  size_t origin_dim_size_{0};
  int data_size_{4};
  SliceStruct slice_param_;
  size_t output_num_{1};
  TypeId param_dtype_{kNumberTypeInt32};
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> begin_shape_;
  std::vector<int64_t> size_shape_;
  bool is_got_value_{false};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SLICE_CPU_KERNEL_H_
