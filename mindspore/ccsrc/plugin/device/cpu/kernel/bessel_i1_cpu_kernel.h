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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_BESSEL_I1_CPU_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_BESSEL_I1_CPU_KERNEL_H

#include <cmath>
#include <vector>
#include <tuple>
#include <map>
#include <memory>
#include <string>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class BesselI1CpuKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  BesselI1CpuKernelMod() = default;
  ~BesselI1CpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  static double bessel_i1_func(double x);
  static double chbevl(double x, double array[], int n);
  template <typename T>
  static void BesselI1Func(const T *input, T *output, size_t start, size_t end);

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);
  size_t input_size_;
  std::vector<size_t> input_shape_;
  std::vector<size_t> output_shape_;
  TypeId input_dtype_;
};

class BesselI1eCpuKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  BesselI1eCpuKernelMod() = default;
  ~BesselI1eCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;
  static double bessel_i1e_func(double x);
  static double chbevl(double x, double array[], int n);
  template <typename T>
  static void BesselI1eFunc(const T *input, T *output, size_t start, size_t end);

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);
  size_t input_size_;
  std::vector<size_t> input_shape_;
  std::vector<size_t> output_shape_;
  TypeId input_dtype_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_BESSEL_I1_CPU_KERNEL_H
