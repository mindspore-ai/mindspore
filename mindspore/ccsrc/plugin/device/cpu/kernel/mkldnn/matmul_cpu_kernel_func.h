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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MKLDNN_MATMUL_CPU_KERNEL_FUNC_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MKLDNN_MATMUL_CPU_KERNEL_FUNC_H_

#include <vector>
#include <map>
#include "plugin/device/cpu/kernel/mkldnn/mkl_cpu_kernel.h"

namespace mindspore {
namespace kernel {
class MatMulCpuKernelFunc : public CpuKernelFunc, private MKLCpuKernelMod {
 public:
  MatMulCpuKernelFunc() = default;
  ~MatMulCpuKernelFunc() override = default;

  void InitFunc(const PrimitivePtr &primitive, const std::vector<KernelTensor *> &inputs,
                const std::vector<KernelTensor *> &outputs) override;

  bool RunFunc(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
               const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 private:
  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    return true;
  };

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return true;
  }

  bool with_bias_add_{false};
  bool with_relu_{false};
  bool trans_a_{false};
  bool trans_b_{false};
  PrimitivePtr prim_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MKLDNN_MATMUL_CPU_KERNEL_FUNC_H_
