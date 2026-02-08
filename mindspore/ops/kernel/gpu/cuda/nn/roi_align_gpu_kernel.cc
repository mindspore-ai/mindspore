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

#include "kernel/gpu/cuda/nn/roi_align_gpu_kernel.h"

namespace mindspore {
namespace kernel {
bool ROIAlignGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  // Check input and output numbers
  constexpr size_t kInputNum = 2;
  constexpr size_t kOutputNum = 1;

  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }
  // Get primitive args
  pooled_height_ = LongToInt(GetValue<int64_t>(primitive_->GetAttr(ops::kPooledHeight)));
  pooled_width_ = LongToInt(GetValue<int64_t>(primitive_->GetAttr(ops::kPooledWidth)));
  spatial_scale_ = GetValue<float>(primitive_->GetAttr(ops::kSpatialScale));
  sample_num_ = LongToInt(GetValue<int64_t>(primitive_->GetAttr(ops::kSampleNum)));
  roi_end_mode_ = LongToInt(GetValue<int64_t>(primitive_->GetAttr(ops::kRoiEndMode)));
  return true;
}

int ROIAlignGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  // Get the input shapes
  auto x_shape = inputs[kIndex0]->GetShapeVector();
  auto rois_shape = inputs[kIndex1]->GetShapeVector();
  constexpr size_t kFeatureDims = 4;
  constexpr size_t kRoisDims = 2;
  if (x_shape.size() > kFeatureDims) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of features cannot be greater than  4, but got "
                  << x_shape.size() << ".";
    return KRET_RESIZE_FAILED;
  }
  if (rois_shape.size() != kRoisDims) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of rois must be equal to 2, but got "
                  << rois_shape.size() << ".";
    return KRET_RESIZE_FAILED;
  }
  // Calculate the sizes of inputs and output
  batch_ = x_shape[kIndex0];
  channel_ = x_shape[kIndex1];
  height_ = x_shape[kIndex2];
  width_ = x_shape[kIndex3];
  roi_rows_ = rois_shape[kIndex0];
  roi_cols_ = rois_shape[kIndex1];
  auto x_type_size = abstract::TypeIdSize(inputs[kIndex0]->dtype_id());
  auto rois_type_size = abstract::TypeIdSize(inputs[kIndex1]->dtype_id());
  x_size_ = batch_ * channel_ * height_ * width_ * x_type_size;
  rois_size_ = roi_rows_ * roi_cols_ * rois_type_size;
  output_size_ = roi_rows_ * channel_ * pooled_height_ * pooled_width_ * rois_type_size;

  ResetResource();
  InitSizeLists();
  return KRET_OK;
}

const ROIAlignGpuKernelMod::FuncList &ROIAlignGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, ROIAlignGpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &ROIAlignGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &ROIAlignGpuKernelMod::LaunchKernel<half>},
  };
  return func_list;
}

template <typename T>
bool ROIAlignGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &workspace,
                                        const std::vector<KernelTensor *> &outputs) {
  const T *x = GetDeviceAddress<T>(inputs, kIndex0);
  const T *rois = GetDeviceAddress<T>(inputs, kIndex1);
  T *out_data = GetDeviceAddress<T>(outputs, kIndex0);
  T spatial_scale = static_cast<T>(spatial_scale_);
  auto status =
    ROIAlign(x, rois, roi_rows_, roi_cols_, out_data, spatial_scale, sample_num_, roi_end_mode_, channel_, height_,
             width_, pooled_height_, pooled_width_, device_id_, reinterpret_cast<cudaStream_t>(stream_ptr_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ROIAlign, ROIAlignGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
