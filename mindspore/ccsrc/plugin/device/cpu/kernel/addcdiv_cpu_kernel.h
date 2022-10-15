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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_ADDCDIV_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_ADDCDIV_CPU_KERNEL_H_

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/cpu/kernel/nnacl/arithmetic.h"

namespace mindspore {
namespace kernel {
class AddcdivCpuKernelMod : public NativeCpuKernelMod {
 public:
  AddcdivCpuKernelMod() = default;
  ~AddcdivCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  TypeId dtype_{kTypeUnknown};
  TypeId dtype_value{kTypeUnknown};
  std::vector<int64_t> input_shape0_;
  std::vector<int64_t> input_shape1_;
  std::vector<int64_t> input_shape2_;
  std::vector<int64_t> input_shape3_;
  std::vector<int64_t> output_shape_;
  int64_t data_shape_size_{0};
  int64_t inputx_shape_size_{0};
  int64_t inputy_shape_size_{0};
  int64_t value_shape_size_{0};
  int64_t output_size_{0};
  ArithmeticParameter mul_para_{};

  template <typename T>
  void AddcdivAdd(const T *input1, const T *input2, T *output);
  template <typename T1, typename T2>
  void AddcdivMul(const T1 *input1, const T2 *input2, T1 *output);
  template <typename T>
  void AddcdivDiv(const T *input1, const T *input2, T *output);
  template <typename T>
  bool AddcdivCheck(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
  template <typename T1, typename T2>
  bool AddcdivCompute(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_ADDCDIV_CPU_KERNEL_H_
