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

#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/nn/roi_align_grad_gpu_kernel.h"

namespace mindspore {
namespace kernel {
bool ROIAlignGradGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  // Check input and output numbers
  constexpr size_t kInputNumNoShape = 2;
  constexpr size_t kInputNumWithShape = 3;
  constexpr size_t kOutputNum = 1;
  kernel_name_ = base_operator->name();
  if (inputs.size() != kInputNumNoShape && inputs.size() != kInputNumWithShape) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 2 or 3, but got " << inputs.size()
                      << ".";
  }
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  // Get primitive args
  auto op = std::dynamic_pointer_cast<ops::ROIAlignGrad>(base_operator);
  pooled_height_ = op->get_pooled_height();
  pooled_width_ = op->get_pooled_width();
  spatial_scale_ = op->get_spatial_scale();
  sample_num_ = op->get_sample_num();
  if (inputs.size() == kInputNumWithShape) {
    is_xdiff_shape_dyn_ = true;
    return true;
  }
  xdiff_shape_ = GetValue<std::vector<int64_t>>(base_operator->GetAttr("xdiff_shape"));
  return true;
}

int ROIAlignGradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs,
                                     const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  if (is_xdiff_shape_dyn_) {
    get_xdiff_shape_value_ = GetDynamicAttrIntValue(inputs, kIndex2, inputsOnHost, kernel_name_, &xdiff_shape_);
    if (!get_xdiff_shape_value_) {
      return KRET_OK;
    }
  }
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
  if (xdiff_shape_.size() > kDiffDims) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the length of xdiff_shape cannot be greater than 4, but got "
                  << xdiff_shape_.size() << ".";
    return KRET_RESIZE_FAILED;
  }
  // Calculate the sizes of inputs and output
  auto dy_type_size = abstract::TypeIdSize(inputs[kIndex0]->GetDtype());
  dy_size_ = dy_shape[kIndex0] * dy_shape[kIndex1] * dy_shape[kIndex2] * dy_shape[kIndex3] * dy_type_size;

  roi_rows_ = rois_shape[kIndex0];
  roi_cols_ = rois_shape[kIndex1];
  auto rois_type_size = abstract::TypeIdSize(inputs[kIndex1]->GetDtype());
  rois_size_ = roi_rows_ * roi_cols_ * rois_type_size;

  batch_ = xdiff_shape_[kIndex0];
  channel_ = xdiff_shape_[kIndex1];
  height_ = xdiff_shape_[kIndex2];
  width_ = xdiff_shape_[kIndex3];
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
bool ROIAlignGradGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                            const std::vector<AddressPtr> &workspace,
                                            const std::vector<AddressPtr> &outputs) {
  const T *dy = GetDeviceAddress<T>(inputs, 0);
  const T *rois = GetDeviceAddress<T>(inputs, 1);
  T *dx = GetDeviceAddress<T>(outputs, 0);
  T spatial_scale = static_cast<T>(spatial_scale_);
  int64_t roi_end_mode = 1;
  ROIAlignGrad(dy, rois, batch_, roi_rows_, roi_cols_, dx, spatial_scale, sample_num_, roi_end_mode, channel_, height_,
               width_, pooled_height_, pooled_width_, reinterpret_cast<cudaStream_t>(stream_ptr_));
  return true;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ROIAlignGrad, ROIAlignGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
