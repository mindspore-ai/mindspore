/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_GRU_GRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_GRU_GRAD_CPU_KERNEL_H_

#include <map>
#include <string>
#include <vector>
#include <memory>
#include "plugin/device/cpu/kernel/mkldnn/mkl_cpu_kernel.h"

namespace mindspore {
namespace kernel {
class GRUGradCpuKernelMod : public MKLCpuKernelMod {
 public:
  GRUGradCpuKernelMod() = default;
  ~GRUGradCpuKernelMod() override = default;

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
                                                     .AddInputAttr(kNumberTypeInt32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeFloat32)};
    return support_list;
  }

 private:
  void AddArgumentOp(const dnnl::memory::desc &src_desc, const dnnl::memory::desc &src_c_desc,
                     const dnnl::memory::desc &bias_desc, const dnnl::memory::desc &dst_desc,
                     const dnnl::memory::desc &dst_h_desc, const dnnl::memory::desc &wksp_desc);
  void SetArgumentHandleOp(const std::vector<kernel::AddressPtr> &inputs,
                           const std::vector<kernel::AddressPtr> &outputs);
  void ResetMemory(const dnnl::memory &mem, const string name) const;
  void InitDnnl();

  int weight_size_{0};
  int weight_h_size_{0};
  int input_size_{0};
  int hidden_size_{0};
  int num_layers_{0};
  int batch_size_{0};
  int seq_len_{0};
  int num_directions_{0};
  bool bidirectional_{false};
  bool has_bias_{false};
  size_t reserve_size_{1};

  dnnl::memory::dims weights_dims_;
  dnnl::memory::dims weights_h_dims_;
  dnnl::memory::dims bias_dims_;

  dnnl::gru_backward::primitive_desc prim_backward_desc_;
  dnnl::memory::desc weights_layer_desc_;
  dnnl::memory::desc weights_iter_desc_;
  dnnl::memory::desc bias_desc_;

  dnnl::memory::desc diff_weights_layer_desc_;
  dnnl::memory::desc diff_weights_iter_desc_;
  dnnl::memory::desc diff_bias_desc_;

  dnnl::memory user_weights_memory_;
  dnnl::memory user_weights_h_memory_;
  dnnl::memory weights_memory_;
  dnnl::memory weights_h_memory_;
  dnnl::memory bias_memory_;
  dnnl::memory diff_weights_memory_;
  dnnl::memory diff_weights_h_memory_;
  dnnl::memory diff_bias_memory_;
  dnnl::memory user_diff_weights_memory_;
  dnnl::memory user_diff_weights_h_memory_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_GRU_GRAD_CPU_KERNEL_H_
