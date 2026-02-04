/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_PS_ROI_POOLING_GRAD_V2_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_PS_ROI_POOLING_GRAD_V2_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <functional>
#include <map>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/psroi_pooling_v2_impl.cuh"
namespace mindspore {
namespace kernel {
class PSROIPoolingBackV2GpuKernelMod : public NativeGpuKernelMod {
 public:
  PSROIPoolingBackV2GpuKernelMod() = default;
  ~PSROIPoolingBackV2GpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
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
  int ResizeCheckInputs(const std::vector<KernelTensor *> &inputs);
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_PS_ROI_POOLING_GRAD_V2_GPU_KERNEL_H_
