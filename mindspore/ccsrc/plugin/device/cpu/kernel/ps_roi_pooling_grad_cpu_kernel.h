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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_PS_ROI_POOLING_GRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_PS_ROI_POOLING_GRAD_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class PSROIPoolingGradCpuKernelMod : public NativeCpuKernelMod {
 public:
  PSROIPoolingGradCpuKernelMod() = default;
  ~PSROIPoolingGradCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &others = std::map<uint32_t, tensor::TensorPtr>()) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  size_t input_size_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;
  TypeId input_dtype_;

  int32_t batch_size_{-1};
  int32_t output_n_{-1};
  int32_t rois_num_{-1};
  float spatial_scale_{0.0};
  int32_t feature_channels_{-1};
  int32_t height_{-1};
  int32_t width_{-1};
  int32_t pooled_height_{-1};
  int32_t pooled_width_{-1};
  int32_t group_size_{-1};
  int32_t output_channels_{-1};

  static bool IsSupportedDtype(TypeId type_id);
  TypeId data_type_id_{kNumberTypeFloat32};

  template <typename T>
  void PSROIPoolBackward(size_t start, size_t end, const T *input_diff, T *output_diff, T *roi_boxes) const;

  int ResizeCheckInputs(const std::vector<KernelTensorPtr> &inputs) const;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_PS_ROI_POOLING_GRAD_CPU_KERNEL_H_
