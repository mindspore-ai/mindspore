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

#include "plugin/device/gpu/kernel/nn/roi_align_gpu_kernel.h"

namespace mindspore {
namespace kernel {
bool ROIAlignGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  // Check input and output numbers
  constexpr size_t kInputNum = 2;
  constexpr size_t kOutputNum = 1;
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  // Get primitive args
  auto op = std::dynamic_pointer_cast<ops::ROIAlign>(base_operator);
  pooled_height_ = op->get_pooled_height();
  pooled_width_ = op->get_pooled_width();
  spatial_scale_ = op->get_spatial_scale();
  sample_num_ = op->get_sample_num();
  roi_end_mode_ = op->get_roi_end_mode();
  return true;
}

int ROIAlignGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs,
                                 const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
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
  auto x_type_size = abstract::TypeIdSize(inputs[kIndex0]->GetDtype());
  auto rois_type_size = abstract::TypeIdSize(inputs[kIndex1]->GetDtype());
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
bool ROIAlignGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                        const std::vector<AddressPtr> &outputs) {
  const T *x = GetDeviceAddress<T>(inputs, kIndex0);
  const T *rois = GetDeviceAddress<T>(inputs, kIndex1);
  T *out_data = GetDeviceAddress<T>(outputs, kIndex0);
  T spatial_scale = static_cast<T>(spatial_scale_);
  ROIAlign(x, rois, roi_rows_, roi_cols_, out_data, spatial_scale, sample_num_, roi_end_mode_, channel_, height_,
           width_, pooled_height_, pooled_width_, reinterpret_cast<cudaStream_t>(stream_ptr_));
  return true;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ROIAlign, ROIAlignGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
