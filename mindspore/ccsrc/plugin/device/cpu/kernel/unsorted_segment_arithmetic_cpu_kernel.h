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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_UNSORTED_SEGMENT_ARITHMETIC_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_UNSORTED_SEGMENT_ARITHMETIC_CPU_KERNEL_H_

#include <utility>
#include <vector>
#include <memory>
#include <limits>
#include <string>
#include <map>
#include <unordered_map>
#include <algorithm>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "nnacl/base/unsorted_segment_sum_base.h"

namespace mindspore {
namespace kernel {
class UnsortedSegmentArithmeticCpuKernelMod : public NativeCpuKernelMod,
                                              public MatchKernelHelper<UnsortedSegmentArithmeticCpuKernelMod> {
 public:
  UnsortedSegmentArithmeticCpuKernelMod() = default;
  ~UnsortedSegmentArithmeticCpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

  std::vector<size_t> GetLaunchIgnoredInputAddressIdx() const override { return {kIndex2}; }

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &workspace,
                    const std::vector<kernel::KernelTensor *> &outputs);

  template <typename T, typename S>
  bool ComputeFunc(T *input_addr, S *ids_addr, T *output_addr) const;

  size_t comp_size_ = 1;
  size_t loop_size_ = 1;
  size_t out_size_ = 1;
  int64_t batch_rank_ = 0;
  int64_t batch_size_ = 1;
  int64_t in_stride_ = 1;
  int64_t ids_stride_ = 1;
  int64_t out_stride_ = 1;
  int64_t num_segments_ = 0;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_UNSORTED_SEGMENT_ARITHMETIC_CPU_KERNEL_H_
