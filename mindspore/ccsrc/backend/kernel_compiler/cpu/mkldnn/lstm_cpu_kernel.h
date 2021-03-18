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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_LSTM_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_LSTM_CPU_KERNEL_H_
#if defined(__x86_64__) || defined(__amd64__) || defined(_M_IX86) || defined(_M_X64)
#define PLATFORM_86
#endif
#ifdef PLATFORM_86
#include <pmmintrin.h>
#endif
#include <vector>
#include <memory>
#include "backend/kernel_compiler/cpu/mkldnn/mkl_cpu_kernel.h"
namespace mindspore {
namespace kernel {
class LstmCPUKernel : public MKLCPUKernel {
 public:
  LstmCPUKernel() = default;
  ~LstmCPUKernel() override = default;
  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  void InitInputOutputSize(const CNodePtr &kernel_node) override;

 private:
  void CheckParam(const CNodePtr &kernel_node);
  int weight_size_ = 0;
  int weight_h_size_ = 0;
  int input_size_;
  int hidden_size_;
  int num_layers_;
  int batch_size_;
  int seq_len_;
  int num_directions_;
  bool bidirectional_;
  bool has_bias_;
  size_t reserve_size_;
  bool is_training;
  dnnl::memory::dims weights_dims_;
  dnnl::memory::dims weights_h_dims_;
  dnnl::memory::dims bias_dims_;
  dnnl::lstm_forward::primitive_desc prim_desc_;
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
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_LSTM_CPU_KERNEL_H
