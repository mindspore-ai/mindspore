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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_TENSOR_COPY_SLICES_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_TENSOR_COPY_SLICES_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "nnacl/fp32/strided_slice_fp32.h"

namespace mindspore {
namespace kernel {
class TensorCopySlicesCpuKernelMod : public NativeCpuKernelMod {
 public:
  TensorCopySlicesCpuKernelMod() = default;
  ~TensorCopySlicesCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {KernelAttr()
                                                     .AddInputAttr(kNumberTypeBool)
                                                     .AddInputAttr(kNumberTypeBool)
                                                     .AddInputAttr(kNumberTypeInt64)
                                                     .AddInputAttr(kNumberTypeInt64)
                                                     .AddInputAttr(kNumberTypeInt64)
                                                     .AddOutputAttr(kNumberTypeBool),
                                                   KernelAttr()
                                                     .AddInputAttr(kNumberTypeInt32)
                                                     .AddInputAttr(kNumberTypeInt32)
                                                     .AddInputAttr(kNumberTypeInt64)
                                                     .AddInputAttr(kNumberTypeInt64)
                                                     .AddInputAttr(kNumberTypeInt64)
                                                     .AddOutputAttr(kNumberTypeInt32),
                                                   KernelAttr()
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeInt64)
                                                     .AddInputAttr(kNumberTypeInt64)
                                                     .AddInputAttr(kNumberTypeInt64)
                                                     .AddOutputAttr(kNumberTypeFloat32),
                                                   KernelAttr()
                                                     .AddInputAttr(kNumberTypeFloat64)
                                                     .AddInputAttr(kNumberTypeFloat64)
                                                     .AddInputAttr(kNumberTypeInt64)
                                                     .AddInputAttr(kNumberTypeInt64)
                                                     .AddInputAttr(kNumberTypeInt64)
                                                     .AddOutputAttr(kNumberTypeFloat64)};
    return support_list;
  }

 private:
  TypeId data_type_;
  size_t offset_{0};
  size_t copy_size_{0};
  bool get_value_before_launch_{false};
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> update_shape_;
  std::vector<int64_t> output_shape_;
  std::vector<int64_t> begin_shape_;
  std::vector<int64_t> end_shape_;
  std::vector<int64_t> stride_shape_;
  void FillSlice(std::vector<int64_t> *begin, std::vector<int64_t> *end);
  void InitOffsetAndCopySize(const std::vector<int64_t> &begin, const std::vector<int64_t> &end,
                             const std::vector<int64_t> &stride);
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_TENSOR_COPY_SLICES_CPU_KERNEL_H_
