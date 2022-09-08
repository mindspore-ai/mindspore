/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_REVERSE_SEQUENCE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_REVERSE_SEQUENCE_CPU_KERNEL_H_

#include <vector>
#include <map>
#include <memory>
#include <algorithm>
#include <utility>
#include <string>
#include <tuple>
#include <complex>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
constexpr auto kUnknown = "Unknown";

class ReverseSequenceCpuKernelMod : public NativeCpuKernelMod {
 public:
  ReverseSequenceCpuKernelMod() = default;
  ~ReverseSequenceCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  int64_t seq_dim_{0};
  int64_t batch_dim_{0};

  // shape correlative
  std::vector<int64_t> input0_shape_;
  std::vector<int64_t> output_shape_;
  int input_stride_[5];
  int output_stride_[5];

  // other parameter
  int ndim_{0};
  int outer_count_{0};
  int outer_stride_{0};
  int inner_count_{0};
  int inner_stride_{0};
  int copy_byte_size_{0};
  int total_data_size_{0};

  void ComputeStrides(const std::vector<int64_t> &shape, int *strides, const int ndim) const;
  int CalcCountPreAxis(const std::vector<int64_t> &shape, int64_t axis) const;
  int CalcCountAfterAxis(const std::vector<int64_t> &shape, int64_t axis) const;
  template <typename T>
  void ResizeKernel(const std::vector<KernelTensorPtr> &inputs, const std::vector<KernelTensorPtr> &outputs);
  template <typename T, typename S>
  void LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);

  using ReverseSequenceFunc = std::function<void(ReverseSequenceCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                                                 const std::vector<kernel::AddressPtr> &)>;
  ReverseSequenceFunc kernel_func_;
  using ResizeFunc = std::function<void(ReverseSequenceCpuKernelMod *, const std::vector<kernel::KernelTensorPtr> &,
                                        const std::vector<kernel::KernelTensorPtr> &)>;
  ResizeFunc resize_func_;
  static std::vector<std::tuple<KernelAttr, ReverseSequenceFunc, ResizeFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_REVERSE_SEQUENCE_CPU_KERNEL_H_
