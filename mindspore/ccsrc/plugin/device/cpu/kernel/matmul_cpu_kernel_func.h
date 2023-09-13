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
#include <string>
#include "plugin/device/cpu/kernel/matmul_cpu_kernel.h"

#include "nnacl/kernel.h"

namespace mindspore {
namespace kernel {
struct MatmulSlice {
  int row_s_ = 0;
  int row_e_ = 0;
  int col_s_ = 0;
  int col_e_ = 0;
};

class MatMulCpuKernelFunc : public CpuKernelFunc {
 public:
  MatMulCpuKernelFunc() = default;
  ~MatMulCpuKernelFunc() override;

  void InitFunc(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                const std::vector<KernelTensorPtr> &outputs) override;

  bool RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
               const std::vector<AddressPtr> &outputs) override;

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

 private:
  void MatmulAVX512BatchColRowSliceThreadCut();

  std::string kernel_name_{kUnkown};
  bool trans_a_{false};
  bool trans_b_{false};
  bool with_bias_add_{false};
  bool with_relu_{false};
  ExecEnv *exec_env_ = nullptr;
  KernelBase *kernel_ = nullptr;
  TensorC **in_ = nullptr;
  TensorC **out_ = nullptr;
  size_t in_size_ = 0;
  size_t out_size_ = 0;
  OpParameter *op_parameter_ = nullptr;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MKLDNN_MATMUL_CPU_KERNEL_FUNC_H_
