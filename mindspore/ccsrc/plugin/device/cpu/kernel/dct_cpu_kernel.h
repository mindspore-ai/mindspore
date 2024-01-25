/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_DCT_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_DCT_CPU_KERNEL_H_

#include <Eigen/Dense>
#include <complex>
#include <string>
#include <vector>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "mindspore/core/ops/op_enum.h"

namespace mindspore {
namespace kernel {
class DCTCpuKernelMod : public NativeCpuKernelMod {
 public:
  DCTCpuKernelMod() = default;
  ~DCTCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T1, typename T2, typename T3>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &outputs);

  using DCTFunc = std::function<bool(DCTCpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                                     const std::vector<kernel::KernelTensor *> &)>;
  static std::vector<std::pair<KernelAttr, DCTFunc>> func_list_;
  DCTFunc kernel_func_;
  int64_t type_;
  int64_t n_;
  int64_t axis_;
  ops::NormMode norm_type_;
  std::vector<int64_t> x_shape_;
  int64_t x_rank_;
  bool forward_;
  bool grad_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_DCT_CPU_KERNEL_H_
