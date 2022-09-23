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
#include "mindspore/core/ops/roi_align_grad.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class ROIAlignGradCpuKernelMod : public NativeCpuKernelMod, public MatchKernelHelper<ROIAlignGradCpuKernelMod> {
 public:
  ROIAlignGradCpuKernelMod() = default;
  ~ROIAlignGradCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); };

 protected:
  using FuncList = std::vector<std::pair<KernelAttr, ROIAlignGradCpuKernelMod::KernelRunFunc>>;

  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);

  void ResetResource() noexcept {
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

  void InitSizeLists() {
    input_size_list_.push_back(dy_size_);
    input_size_list_.push_back(rois_size_);
    output_size_list_.push_back(output_size_);
  }

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

  std::vector<int64_t> xdiff_shape_;

  size_t dy_size_{0};
  size_t rois_size_{0};
  size_t output_size_{0};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_ROI_ALIGN_GRAD_CPU_KERNEL_H_
