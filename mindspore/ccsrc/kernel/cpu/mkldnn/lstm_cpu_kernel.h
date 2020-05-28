/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_KERNEL_CPU_LSTM_CPU_KERNEL_H
#define MINDSPORE_CCSRC_KERNEL_CPU_LSTM_CPU_KERNEL_H
#include <vector>
#include <memory>
#include "kernel/cpu/mkldnn/mkl_cpu_kernel.h"
namespace mindspore {
namespace kernel {
class LstmCPUKernel : public MKLCPUKernel {
 public:
  LstmCPUKernel() = default;
  ~LstmCPUKernel() override = default;
  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  int weight_size_ = 0;
  int weight_h_size_ = 0;
  int input_size_;
  int hidden_size_;
  int num_layers_;
  int batch_size_;
  int seq_len_;
  int num_directions_;
  bool bidirectional_;
};

MS_REG_CPU_KERNEL(LSTM,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32),
                  LstmCPUKernel);
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_KERNEL_CPU_LSTM_CPU_KERNEL_H
