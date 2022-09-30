/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_EIGEN_MATMUL_DOULE_CPU_KERNEL_FUNC_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_EIGEN_MATMUL_DOULE_CPU_KERNEL_FUNC_H_

#include <complex>
#include <string>
#include <vector>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"

namespace mindspore {
namespace kernel {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

class MatmulDoubleCpuKernelFunc : public CpuKernelFunc, private NativeCpuKernelMod {
 public:
  MatmulDoubleCpuKernelFunc() = default;
  ~MatmulDoubleCpuKernelFunc() override = default;

  void InitFunc(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                const std::vector<KernelTensorPtr> &outputs) override;

  bool RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
               const std::vector<AddressPtr> &outputs) override;

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

 private:
  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override {
    return true;
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return true;
  }

  template <typename T>
  void MatMul(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  template <typename T>
  void ComputeMatMulOutput(T *a_addr, T *b_addr, T *output_addr);

  size_t batch_{0};
  size_t rank_{0};
  size_t a_row_{0};
  size_t b_row_{0};
  size_t out_row_{0};
  size_t a_col_{0};
  size_t b_col_{0};
  size_t out_col_{0};
  bool trans_a_{false};
  bool trans_b_{false};
  std::string kernel_name_;
  TypeId dtype_{kTypeUnknown};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_EIGEN_MATMUL_DOULE_CPU_KERNEL_FUNC_H_
