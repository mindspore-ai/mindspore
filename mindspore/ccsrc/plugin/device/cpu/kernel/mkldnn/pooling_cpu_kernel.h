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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_POOLING_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_POOLING_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <unordered_map>
#include <map>
#include <string>
#include "plugin/device/cpu/kernel/mkldnn/mkl_cpu_kernel.h"

namespace mindspore {
namespace kernel {
constexpr auto kUnkown = "Unknown";
constexpr size_t kPoolingDilation = 1;

class PoolingCpuKernelMod : public MKLCpuKernelMod {
 public:
  PoolingCpuKernelMod() = default;
  explicit PoolingCpuKernelMod(const std::string &kernel_type) : kernel_type_(kernel_type) {}
  ~PoolingCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 protected:
  void EliminateInvalidPadding(float *output);
  void ReComputeDivisor(float *output);

  dnnl::algorithm algorithm_{dnnl::algorithm::pooling_max};
  bool ceil_mode_{false};
  int64_t divisor_override_{0};
  std::vector<int64_t> dst_shape_;
  std::vector<int64_t> kernel_;
  std::vector<int64_t> padding_invalid_;
  std::string format_;
  std::string pad_mode_;
  std::vector<int64_t> kernel_include_nc_{};
  std::vector<int64_t> strides_include_nc_{};
  BaseOperatorPtr base_operator_{nullptr};
  std::vector<KernelTensorPtr> inputs_{};
  std::vector<KernelTensorPtr> outputs_{};
  std::map<uint32_t, tensor::TensorPtr> inputs_on_host_{};

 private:
  std::string kernel_type_{kUnkown};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_POOLING_CPU_KERNEL_H_
