/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include <vector>
#include <memory>
#include <map>
#include "plugin/device/cpu/kernel/mkldnn/mkl_cpu_kernel.h"

namespace mindspore {
namespace kernel {
class LstmCpuKernelMod : public MKLCpuKernelMod {
 public:
  LstmCpuKernelMod() = default;
  ~LstmCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {KernelAttr()
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeFloat32)};
    return support_list;
  }

 private:
  void InitOutputSize(const std::vector<KernelTensorPtr> &outputs);

  int64_t weight_size_{0};
  int64_t weight_h_size_{0};
  int64_t weight_r_size_{0};
  int64_t input_size_{0};
  int64_t hidden_size_{0};
  int64_t num_layers_{0};
  int64_t batch_size_{0};
  int64_t seq_len_{0};
  int64_t num_directions_{0};
  int64_t proj_size_{0};
  int64_t real_hidden_size_{0};
  bool bidirectional_{false};
  bool has_bias_{false};
  bool is_training_{false};
  size_t reserve_size_{1};

  dnnl::memory::dims weights_dims_;
  dnnl::memory::dims weights_h_dims_;
  dnnl::memory::dims weights_r_dims_;
  dnnl::memory::dims bias_dims_;
  dnnl::lstm_forward::primitive_desc prim_desc_;
  dnnl::memory::desc bias_desc_;
  dnnl::memory user_weights_memory_;
  dnnl::memory user_weights_h_memory_;
  dnnl::memory user_weights_r_memory_;
  dnnl::memory weights_memory_;
  dnnl::memory weights_h_memory_;
  dnnl::memory weights_r_memory_;
  dnnl::memory bias_memory_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_LSTM_CPU_KERNEL_H_
