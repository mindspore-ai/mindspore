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

#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/cuda/nn/roi_align_grad_gpu_kernel.h"

namespace mindspore {
namespace kernel {
bool ROIAlignGradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  // Check input and output numbers
  constexpr size_t kInputNum = 3;
  constexpr size_t kOutputNum = 1;

  if (inputs.size() != kInputNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 3, but got " << inputs.size()
                      << ".";
  }
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }
  // Get primitive args
  pooled_height_ = LongToInt(GetValue<int64_t>(primitive_->GetAttr(ops::kPooledHeight)));
  pooled_width_ = LongToInt(GetValue<int64_t>(primitive_->GetAttr(ops::kPooledWidth)));
  spatial_scale_ = GetValue<float>(primitive_->GetAttr(ops::kSpatialScale));
  sample_num_ = LongToInt(GetValue<int64_t>(primitive_->GetAttr(ops::kSampleNum)));
  return true;
}

int ROIAlignGradGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  std::vector<int64_t> xdiff_shape;
  xdiff_shape = inputs[kIndex2]->GetValueWithCheck<std::vector<int64_t>>();
  // Get the input shapes
  auto dy_shape = inputs[kIndex0]->GetShapeVector();
  auto rois_shape = inputs[kIndex1]->GetShapeVector();
  constexpr size_t kDiffDims = 4;
  constexpr size_t kRoisDims = 2;
  if (dy_shape.size() != kDiffDims) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of dy must be equal to 4, but got " << dy_shape.size()
                  << ".";
    return KRET_RESIZE_FAILED;
  }
  if (rois_shape.size() != kRoisDims) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of rois must be equal to 2, but got "
                  << rois_shape.size() << ".";
    return KRET_RESIZE_FAILED;
  }
  if (xdiff_shape.size() > kDiffDims) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the length of xdiff_shape cannot be greater than 4, but got "
                  << xdiff_shape.size() << ".";
    return KRET_RESIZE_FAILED;
  }
  // Calculate the sizes of inputs and output
  auto dy_type_size = abstract::TypeIdSize(inputs[kIndex0]->dtype_id());
  dy_size_ = dy_shape[kIndex0] * dy_shape[kIndex1] * dy_shape[kIndex2] * dy_shape[kIndex3] * dy_type_size;

  roi_rows_ = rois_shape[kIndex0];
  roi_cols_ = rois_shape[kIndex1];
  auto rois_type_size = abstract::TypeIdSize(inputs[kIndex1]->dtype_id());
  rois_size_ = roi_rows_ * roi_cols_ * rois_type_size;

  batch_ = xdiff_shape[kIndex0];
  channel_ = xdiff_shape[kIndex1];
  height_ = xdiff_shape[kIndex2];
  width_ = xdiff_shape[kIndex3];
  output_size_ = batch_ * channel_ * height_ * width_ * dy_type_size;

  ResetResource();
  InitSizeLists();
  return KRET_OK;
}

const ROIAlignGradGpuKernelMod::FuncList &ROIAlignGradGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, ROIAlignGradGpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &ROIAlignGradGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &ROIAlignGradGpuKernelMod::LaunchKernel<half>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &ROIAlignGradGpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16),
     &ROIAlignGradGpuKernelMod::LaunchKernel<half>},
  };
  return func_list;
}

template <typename T>
bool ROIAlignGradGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &workspace,
                                            const std::vector<KernelTensor *> &outputs) {
  const T *dy = GetDeviceAddress<T>(inputs, 0);
  const T *rois = GetDeviceAddress<T>(inputs, 1);
  T *dx = GetDeviceAddress<T>(outputs, 0);
  T spatial_scale = static_cast<T>(spatial_scale_);
  int64_t roi_end_mode = 1;
  auto status = ROIAlignGrad(dy, rois, batch_, roi_rows_, roi_cols_, dx, spatial_scale, sample_num_, roi_end_mode,
                             channel_, height_, width_, pooled_height_, pooled_width_, device_id_,
                             reinterpret_cast<cudaStream_t>(stream_ptr_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ROIAlignGrad, ROIAlignGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
