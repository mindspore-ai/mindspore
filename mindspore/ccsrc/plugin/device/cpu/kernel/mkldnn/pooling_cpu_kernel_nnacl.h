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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_POOLING_CPU_KERNEL_NNACL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_POOLING_CPU_KERNEL_NNACL_H_

#include <vector>
#include <memory>
#include <utility>
#include <unordered_map>
#include <map>
#include <string>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/nnacl/kernel/pooling.h"
#include "plugin/device/cpu/kernel/nnacl/pooling_parameter.h"

namespace mindspore {
namespace kernel {
constexpr auto kUnkown = "Unknown";

class PoolingCpuKernelNnaclMod : public NativeCpuKernelMod {
 public:
  PoolingCpuKernelNnaclMod() = default;
  explicit PoolingCpuKernelNnaclMod(const std::string &kernel_type) : kernel_type_(kernel_type) {}
  ~PoolingCpuKernelNnaclMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 protected:
  PoolMode pool_mode_;
  std::string format_;
  std::string pad_mode_;
  int64_t batches_{0};
  int64_t channels_{0};
  int64_t input_stride_n_{1};
  int64_t input_stride_c_{1};
  int64_t input_stride_d_{1};
  int64_t input_stride_h_{1};
  int64_t input_stride_w_{1};
  size_t output_num_{1};
  std::vector<int64_t> in_size_;
  std::vector<int64_t> out_size_;
  std::vector<int64_t> kernel_size_;
  std::vector<int64_t> stride_size_;
  std::vector<int64_t> pad_list_;
  std::vector<int64_t> padding_l_;
  std::vector<int64_t> padding_r_;
  bool ceil_mode_{false};
  bool count_include_pad_{true};
  int64_t divisor_override_{0};
  Pooling3DParameter pooling_param_;
  Pooling3DComputeParam pooling_args_;

 private:
  std::string kernel_type_{kUnkown};

  void GetPadList(size_t src_dim, size_t padlist_len);

  void InitPooling3DParams();

  template <typename T>
  CTask KernelAvgPool(T *input_addr, T *output_addr);

  template <typename T>
  CTask KernelMaxPool(T *input_addr, T *output_addr);

  void LaunchPoolingChannelLastFp32(float *input_addr, float *transpose_out, float *pooling_out, float *output_addr);

  void LaunchTransposeFp32(float *input_addr, float *output_addr, int plane, int channel);

  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspaces,
                    const std::vector<kernel::AddressPtr> &outputs);

  TypeId dtype_{kTypeUnknown};
  bool use_channel_last_{false};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_POOLING_CPU_KERNEL_NNACL_H_
