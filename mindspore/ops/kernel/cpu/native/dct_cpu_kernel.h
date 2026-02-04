/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include <vector>
#include <complex>
#include <utility>
#include "kernel/cpu/cpu_kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "include/op_enum.h"

namespace mindspore {
namespace kernel {
namespace dct_cpu {
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
  void UpdateParam();

  template <typename T_in, typename T_out>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &outputs);

  template <typename T_in, typename T_out>
  bool LaunchKernelComplex(const std::vector<kernel::KernelTensor *> &inputs,
                           const std::vector<kernel::KernelTensor *> &outputs);

  using DCTFunc =
    std::function<bool(DCTCpuKernelMod *, const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &)>;
  static std::vector<std::pair<KernelAttr, DCTFunc>> func_list_;

  DCTFunc kernel_func_;

  bool forward_;
  int64_t x_rank_;
  int64_t dct_type_;
  int64_t dim_;
  int64_t n_;
  int64_t input_element_nums_;
  int64_t calculate_element_nums_;
  double norm_weight_;
  mindspore::NormMode norm_;
  bool is_ortho_;

  std::vector<int64_t> tensor_shape_;
  std::vector<int64_t> calculate_shape_;
};
}  // namespace dct_cpu
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_DCT_CPU_KERNEL_H_
