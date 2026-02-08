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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MEDIAN_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MEDIAN_CPU_KERNEL_H_

#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "utils/convert_utils.h"
#include "kernel/cpu/cpu_kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace median_cpu {
class MedianCpuKernelMod : public NativeCpuKernelMod, public MatchKernelHelper<MedianCpuKernelMod> {
 public:
  MedianCpuKernelMod() = default;
  ~MedianCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    MS_EXCEPTION_IF_NULL(kernel_func_);
    return kernel_func_(this, inputs, workspace, outputs);
  }

 protected:
  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  TypeId input_type_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;
  bool global_median_;
  bool keepdim_{false};
  bool ignore_nan_;
  int axis_{0};
  int64_t input_num_elements_;
  size_t output_num_elements_;
  size_t input_dim_;
  bool is_null_input_;
  template <typename T>
  bool GlobalMedianCompute(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
  template <typename T>
  bool MedianCompute(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                     const std::vector<KernelTensor *> &outputs);
  template <typename T>
  bool MedianComputeIgnoreNan(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                              const std::vector<KernelTensor *> &outputs);
  template <typename T, typename... CompFunc>
  inline T CopyAndCalcMedian(const T *const input_begin, int64_t median_pos, CompFunc... comp);
};
}  // namespace median_cpu
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MEDIAN_CPU_KERNEL_H_
