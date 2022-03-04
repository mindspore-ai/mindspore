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
#include <memory>
#include <map>
#include <string>
#include "plugin/device/cpu/kernel/mkldnn/mkl_cpu_kernel.h"

namespace mindspore {
namespace kernel {
class MatMulCpuKernelFunc : public CpuKernelFunc, private MKLCpuKernelMod {
 public:
  MatMulCpuKernelFunc() = default;
  ~MatMulCpuKernelFunc() override = default;

  void InitFunc(const CNodePtr &kernel_node) override;

  bool RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
               const std::vector<AddressPtr> &outputs) override;

 private:
  void InitKernel(const CNodePtr &kernel_node) override {}
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return true;
  }
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MKLDNN_MATMUL_CPU_KERNEL_FUNC_H_
