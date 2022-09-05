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

#include "plugin/device/gpu/kernel/arrays/crop_and_resize_gpu_kernel.h"
#include "mindspore/core/ops/crop_and_resize.h"

namespace mindspore {
namespace kernel {
bool CropAndResizeGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  constexpr size_t input_num = 4;
  constexpr size_t output_num = 1;
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  // get op parameters
  auto kernel_ptr = std::dynamic_pointer_cast<ops::CropAndResize>(base_operator);
  // suppose use kernel_ptr->get_method(), but the definition in lite is enumeration, not std::string. So we use this
  // for the moment to support dynamic shape.
  std::string method = GetValue<std::string>(kernel_ptr->GetAttr("method"));
  if (method == "bilinear") {
    method_ = kMethodBilinear;
  } else if (method == "nearest") {
    method_ = kMethodNearest;
  } else {  //  bilinear-v2
    method_ = kMethodBilinearV2;
  }
  extrapolation_value_ = kernel_ptr->get_extrapolation_value();
  return true;
}

int CropAndResizeGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  // input image
  auto input_image_shape = inputs[kIndex0]->GetShapeVector();
  auto input_boxes_shape = inputs[kIndex1]->GetShapeVector();
  auto input_box_index_shape = inputs[kIndex2]->GetShapeVector();
  auto input_crop_size_shape = inputs[kIndex3]->GetShapeVector();
  auto output_shape = outputs[kIndex0]->GetShapeVector();
  size_t input_image_shape_len = input_image_shape.size();
  if (input_image_shape_len != kImgDimSize) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of x must be 4, but got " << input_image_shape_len;
    return KRET_RESIZE_FAILED;
  }

  input_height_ = input_image_shape[kImgHIndex];
  input_width_ = input_image_shape[kImgWIndex];
  // input boxes
  size_t input_boxes_shape_len = input_boxes_shape.size();
  if (input_boxes_shape_len != kBoxDimSize) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of boxes must be 2, but got "
                  << input_boxes_shape_len;
    return KRET_RESIZE_FAILED;
  }
  // input box_index
  size_t input_box_index_shape_len = input_box_index_shape.size();
  if (input_box_index_shape_len != 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of box_index must be 1, but got "
                  << input_box_index_shape_len;
    return KRET_RESIZE_FAILED;
  }
  // input crop_size
  size_t input_crop_size_shape_len = input_crop_size_shape.size();
  if (input_crop_size_shape_len != 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of crop_size must be 1, but got "
                  << input_crop_size_shape_len;
    return KRET_RESIZE_FAILED;
  }
  if (input_crop_size_shape[0] != kCropLengthSize) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the length of crop_size must be 2, but got "
                  << input_crop_size_shape[0];
    return KRET_RESIZE_FAILED;
  }
  // output
  auto output_shape_len = output_shape.size();
  if (output_shape_len != kOutputDimSize) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of output must be 4, but got " << output_shape_len;
    return KRET_RESIZE_FAILED;
  }

  // set expected output params
  batch_ = output_shape[kIndexForBatch];
  final_height_ = output_shape[kIndexForHeight];
  final_width_ = output_shape[kIndexForWidth];
  channel_ = output_shape[kIndexForChannel];

  output_size_ = 1;
  for (size_t i = 0; i < output_shape_len; i++) {
    output_size_ *= output_shape[i];
  }
  return KRET_OK;
}

template <typename T>
bool CropAndResizeGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                             const std::vector<AddressPtr> &workspace,
                                             const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  T *input_image = GetDeviceAddress<T>(inputs, 0);
  float *input_boxes = GetDeviceAddress<float>(inputs, 1);
  int *input_box_index = GetDeviceAddress<int>(inputs, 2);
  float *output = GetDeviceAddress<float>(outputs, 0);
  CalCropAndResize(output_size_, input_image, input_boxes, input_box_index, batch_, input_height_, input_width_,
                   final_height_, final_width_, channel_, method_, extrapolation_value_, output,
                   reinterpret_cast<cudaStream_t>(stream_ptr));
  return true;
}

#define CROP_AND_RESIZE_GPU_REG(MS_T, MS_S, T) \
  KernelAttr()                                 \
    .AddInputAttr(MS_T)                        \
    .AddInputAttr(kNumberTypeFloat32)          \
    .AddInputAttr(kNumberTypeInt32)            \
    .AddInputAttr(MS_S)                        \
    .AddOutputAttr(kNumberTypeFloat32),        \
    &CropAndResizeGpuKernelMod::LaunchKernel<T>

std::vector<std::pair<KernelAttr, CropAndResizeGpuKernelMod::CropAndResizeLaunchFunc>>
  CropAndResizeGpuKernelMod::func_list_ = {
    {CROP_AND_RESIZE_GPU_REG(kNumberTypeInt8, kNumberTypeInt32, int8_t)},
    {CROP_AND_RESIZE_GPU_REG(kNumberTypeInt8, kNumberTypeInt64, int8_t)},
    {CROP_AND_RESIZE_GPU_REG(kNumberTypeInt16, kNumberTypeInt32, int16_t)},
    {CROP_AND_RESIZE_GPU_REG(kNumberTypeInt16, kNumberTypeInt64, int16_t)},
    {CROP_AND_RESIZE_GPU_REG(kNumberTypeInt32, kNumberTypeInt32, int32_t)},
    {CROP_AND_RESIZE_GPU_REG(kNumberTypeInt32, kNumberTypeInt64, int32_t)},
    {CROP_AND_RESIZE_GPU_REG(kNumberTypeInt64, kNumberTypeInt32, int64_t)},
    {CROP_AND_RESIZE_GPU_REG(kNumberTypeInt64, kNumberTypeInt64, int64_t)},
    {CROP_AND_RESIZE_GPU_REG(kNumberTypeFloat16, kNumberTypeInt32, half)},
    {CROP_AND_RESIZE_GPU_REG(kNumberTypeFloat16, kNumberTypeInt64, half)},
    {CROP_AND_RESIZE_GPU_REG(kNumberTypeFloat32, kNumberTypeInt32, float)},
    {CROP_AND_RESIZE_GPU_REG(kNumberTypeFloat32, kNumberTypeInt64, float)},
    {CROP_AND_RESIZE_GPU_REG(kNumberTypeFloat64, kNumberTypeInt32, double)},
    {CROP_AND_RESIZE_GPU_REG(kNumberTypeFloat64, kNumberTypeInt64, double)},
    {CROP_AND_RESIZE_GPU_REG(kNumberTypeUInt8, kNumberTypeInt32, uint8_t)},
    {CROP_AND_RESIZE_GPU_REG(kNumberTypeUInt8, kNumberTypeInt64, uint8_t)},
    {CROP_AND_RESIZE_GPU_REG(kNumberTypeUInt16, kNumberTypeInt32, uint16_t)},
    {CROP_AND_RESIZE_GPU_REG(kNumberTypeUInt16, kNumberTypeInt64, uint16_t)},
};

std::vector<KernelAttr> CropAndResizeGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, CropAndResizeGpuKernelMod::CropAndResizeLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, CropAndResize, CropAndResizeGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
