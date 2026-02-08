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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SCATTER_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SCATTER_CPU_KERNEL_H_

#include <vector>
#include <string>
#include <memory>
#include <map>
#include <utility>
#include "kernel/cpu/cpu_kernel.h"
#include "mindapi/base/types.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "include/utils/utils.h"

namespace mindspore::kernel {
namespace scatter_cpu {
constexpr auto kUnKnown = "Unknown";
constexpr auto kScatter = "Scatter";
constexpr auto kTensorScatterElements = "TensorScatterElements";

class ScatterCpuKernelMod : public NativeCpuKernelMod, public MatchKernelHelper<ScatterCpuKernelMod> {
 public:
  ScatterCpuKernelMod() = default;
  explicit ScatterCpuKernelMod(const std::string &kernel_type) : kernel_type_(kernel_type) {}
  ~ScatterCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

 private:
  template <typename T, typename S, typename ReductionT>
  bool Scatter(const ReductionT &reduction_func, T *output, const S *indices, const T *src);

  template <typename T, typename S>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                    const std::vector<kernel::KernelTensor *> &outputs);

 private:
  std::string kernel_type_{kUnKnown};
  int input_axis_size_{0};
  size_t input_size_{1};
  size_t indices_total_num_{1};
  size_t input_dims_{0};
  int64_t axis_{0};
  std::vector<size_t> output_stride_{};
  std::vector<int> indices_stride_{};
  Reduce reduction_type_{Reduce::REDUCE_NONE};
  std::string input_name_{"input"};
  std::string axis_name_{"dim"};
  std::string index_name_{"index"};
  std::string src_name_{"src"};
  size_t input_idx_ = kIndex0;
  size_t axis_idx_ = kIndex1;
  size_t index_idx_ = kIndex2;
  size_t src_idx_ = kIndex3;
  size_t reduce_idx_ = kIndex4;
};
}  // namespace scatter_cpu
}  // namespace mindspore::kernel

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SCATTER_CPU_KERNEL_H_
