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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MINIMUM_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MINIMUM_CPU_KERNEL_H_
#include <map>
#include <vector>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class MinimumCpuKernelMod : public NativeCpuKernelMod {
 public:
  MinimumCpuKernelMod() = default;
  ~MinimumCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

 private:
  bool IsBroadcast() const;
  int64_t Index(const int64_t &index, const int64_t &dim) const;
  void InitTensorBroadcastShape();
  void InitInputTensorAndScalar(size_t max_input_shape_size);
  void InitInputTensors(TypeId input_x_dtype, TypeId input_y_dtype);

  // Broadcast Arithmetic
  template <typename T>
  void BroadcastArithKernel(const int64_t l0, const int64_t l1, const int64_t l2, const int64_t l3, const int64_t l4,
                            const int64_t l5, const int64_t l6, const int64_t r0, const int64_t r1, const int64_t r2,
                            const int64_t r3, const int64_t r4, const int64_t r5, const int64_t r6, const int64_t d0,
                            const int64_t d1, const int64_t d2, const int64_t d3, const int64_t d4, const int64_t d5,
                            const int64_t d6, const T *input_x, const T *input_y, T *output) const;
  template <typename T>
  T MinimumFunc(const T &lhs, const T &rhs) const {
    return lhs < rhs ? lhs : rhs;
  }
  template <typename T>
  void BroadcastArithOneScalarOneTensor(const T *input_x, const T *input_y, T *output) const;
  template <typename T>
  void BroadcastArithTensors(const T *input_x, const T *input_y, T *output) const;
  template <typename T>
  void BroadcastArith(const T *input_x, const T *input_y, T *output) const;
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                    const std::vector<kernel::AddressPtr> &outputs) const;

  using MinimumLaunchFunc = std::function<bool(MinimumCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                                               const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, MinimumLaunchFunc>> func_list_;
  MinimumLaunchFunc kernel_func_;

  bool need_broadcast_{false};
  size_t input_x_num_{1};
  size_t input_y_num_{1};
  size_t output_num_{1};
  std::vector<int64_t> input_x_shape_;
  std::vector<int64_t> input_y_shape_;
  std::vector<int64_t> output_shape_;
  std::vector<int64_t> broadcast_input_x_shape_;
  std::vector<int64_t> broadcast_input_y_shape_;
  std::vector<int64_t> broadcast_output_shape_;
  const size_t max_dims_{7};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MINIMUM_CPU_KERNEL_H_
