/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_ROI_ALIGN_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_ROI_ALIGN_CPU_KERNEL_H_
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <memory>
#include <utility>
#include "mindspore/core/ops/grad/roi_align_grad.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class ROIAlignGradCpuKernelMod : public NativeCpuKernelMod, public MatchKernelHelper<ROIAlignGradCpuKernelMod> {
 public:
  ROIAlignGradCpuKernelMod() = default;
  ~ROIAlignGradCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); };

 protected:
  using FuncList = std::vector<std::pair<KernelAttr, ROIAlignGradCpuKernelMod::KernelRunFunc>>;

  template <typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &workspace,
                    const std::vector<kernel::KernelTensor *> &outputs);

  void ResetResource() noexcept {
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

  void InitSizeLists() { output_size_list_.push_back(output_size_); }

 private:
  bool is_xdiff_shape_dyn_{false};
  bool get_xdiff_shape_value_{false};
  int pooled_height_{0};
  int pooled_width_{0};
  float spatial_scale_{0.0};
  int sample_num_{0};
  int roi_end_mode_{0};

  int roi_rows_{0};
  int roi_cols_{0};
  int batch_{0};
  int channels_{0};
  int height_{0};
  int width_{0};

  size_t dy_size_{0};
  size_t rois_size_{0};
  size_t output_size_{0};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_ROI_ALIGN_GRAD_CPU_KERNEL_H_
