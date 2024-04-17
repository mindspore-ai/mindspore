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
#include <string>
#include <tuple>
#include <complex>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

class ReverseSequenceCpuKernelMod : public NativeCpuKernelMod {
 public:
  ReverseSequenceCpuKernelMod() = default;
  ~ReverseSequenceCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  int64_t seq_dim_{0};
  int64_t batch_dim_{0};

  // shape correlative
  std::vector<int64_t> input0_shape_;
  std::vector<int64_t> output_shape_;
  int input_stride_[16];
  int output_stride_[16];

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
  void ResizeKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
  template <typename T, typename S>
  void LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &outputs);

  using KernelFunc = std::function<void(ReverseSequenceCpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                                        const std::vector<kernel::KernelTensor *> &)>;
  KernelFunc kernel_func_;
  using ResizeFunc = std::function<void(ReverseSequenceCpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                                        const std::vector<kernel::KernelTensor *> &)>;
  ResizeFunc resize_func_;
  static std::vector<std::tuple<KernelAttr, KernelFunc, ResizeFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_REVERSE_SEQUENCE_CPU_KERNEL_H_
