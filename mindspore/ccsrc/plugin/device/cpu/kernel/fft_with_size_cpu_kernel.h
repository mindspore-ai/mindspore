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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_FFTWITHSIZE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_FFTWITHSIZE_CPU_KERNEL_H_
#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <utility>
#include <type_traits>
#include <complex>
#include <string>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace mindspore {
constexpr size_t kInputNum = 1;
constexpr size_t kOutputNum = 1;
namespace kernel {
class FFTWithSizeCpuKernelMod : public NativeCpuKernelMod {
 public:
  FFTWithSizeCpuKernelMod() = default;
  ~FFTWithSizeCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T1, typename T2>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);
  using FFTWithSizeFunc = std::function<bool(FFTWithSizeCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                                             const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, FFTWithSizeFunc>> func_list_;
  FFTWithSizeFunc kernel_func_;
  bool real_;
  bool inverse_;
  bool onesided_;
  int64_t signal_ndim_;
  std::string normalized_;
  std::vector<int64_t> raw_checked_signal_size_;
  std::vector<int64_t> x_shape_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_FFTWITHSIZE_CPU_KERNEL_H_
