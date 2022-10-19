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

#include "plugin/device/cpu/kernel/crop_and_resize_grad_boxes_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
bool CropAndResizeGradBoxesCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                              const std::vector<KernelTensorPtr> &inputs,
                                              const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNums, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int CropAndResizeGradBoxesCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                               const std::vector<KernelTensorPtr> &inputs,
                                               const std::vector<KernelTensorPtr> &outputs,
                                               const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  //  input grads
  grads_shape_ = inputs[kGrads]->GetShapeVector();
  size_t input_grads_shape_len = grads_shape_.size();
  if (input_grads_shape_len != kGradsShapeLen) {
    MS_LOG(ERROR) << "Grads tensor is " << input_grads_shape_len << "-D, but CropAndResizeGradBoxes supports only "
                  << kGradsShapeLen << "-D for grads tensor.";
    return KRET_RESIZE_FAILED;
  }

  //  input image
  image_shape_ = inputs[kImages]->GetShapeVector();
  size_t input_image_shape_len = image_shape_.size();
  if (input_image_shape_len != kImageShapeLen) {
    MS_LOG(ERROR) << "Images tensor is " << input_image_shape_len << "-D, but CropAndResizeGradBoxes supports only "
                  << kImageShapeLen << "-D for images tensor.";
    return KRET_RESIZE_FAILED;
  }

  //  input boxes
  boxes_shape_ = inputs[kBoxes]->GetShapeVector();
  size_t input_boxes_shape_len = boxes_shape_.size();
  if (input_boxes_shape_len != kBoxesShapeLen) {
    MS_LOG(ERROR) << "Boxes tensor is " << input_boxes_shape_len << "-D, but CropAndResizeGradBoxes supports only "
                  << kBoxesShapeLen << "-D for boxes tensor.";
    return KRET_RESIZE_FAILED;
  }
  if (LongToSize(boxes_shape_[1]) != kCoordinateLen) {
    MS_LOG(ERROR) << "The coordinate size of boxes is " << boxes_shape_[1]
                  << ", but CropAndResizeGradBoxes supports only " << kCoordinateLen << "for boxes.";
    return KRET_RESIZE_FAILED;
  }

  //  input box_index
  box_in_shape_ = inputs[kBoxIndex]->GetShapeVector();
  size_t input_box_index_shape_len = box_in_shape_.size();
  if (input_box_index_shape_len != kBoxIndexShapeLen) {
    MS_LOG(ERROR) << "Box_index tensor is " << input_box_index_shape_len
                  << "-D, but CropAndResizeGradBoxes supports only " << kBoxIndexShapeLen << "-D for box_index.";
    return KRET_RESIZE_FAILED;
  }

  //  output
  output_shape_ = outputs[kOutputIndex]->GetShapeVector();
  auto output_shape_len = output_shape_.size();
  if (output_shape_len != kOutputShapeLen) {
    MS_LOG(ERROR) << "Output tensor is " << output_shape_len << "-D, but CropAndResizeGradBoxes supports only "
                  << kOutputShapeLen << "-D for output tensor.";
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

void CropAndResizeGradBoxesCpuKernelMod::OutputZeroing(const std::vector<kernel::AddressPtr> &outputs) {
  auto *outputDatas = reinterpret_cast<float *>(outputs[0]->addr);
  const int nums_boxes = LongToInt(grads_shape_[0]);
  int num = nums_boxes * SizeToInt(kCoordinateLen);
  float zero_num = static_cast<float>(0);
  for (int i = 0; i < num; i++) {
    *(outputDatas + i) = zero_num;
  }
}

template <typename T>
bool CropAndResizeGradBoxesCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                      const std::vector<kernel::AddressPtr> &outputs) {
  auto *grads = reinterpret_cast<float *>(inputs[kGrads]->addr);
  auto *image = reinterpret_cast<T *>(inputs[kImages]->addr);
  auto *boxes = reinterpret_cast<float *>(inputs[kBoxes]->addr);
  auto *box_ind = reinterpret_cast<int *>(inputs[kBoxIndex]->addr);
  auto *outputDatas = reinterpret_cast<float *>(outputs[0]->addr);
  const int image_batch = LongToInt(image_shape_[kBatch]);
  const int image_height = LongToInt(image_shape_[kHeight]);
  const int image_width = LongToInt(image_shape_[kWidth]);
  const int depth = LongToInt(image_shape_[kDepth]);
  const int nums_boxes = LongToInt(grads_shape_[kNumBoxes]);
  const int crop_height = LongToInt(grads_shape_[kHeight]);
  const int crop_width = LongToInt(grads_shape_[kWidth]);
  const int crop_depth = LongToInt(grads_shape_[kDepth]);
  const int boxesCoordinateNum = LongToInt(boxes_shape_[1]);
  const int num_image2 = image_height * image_width * depth;
  const int num_image3 = image_width * depth;
  const int num_crop2 = crop_height * crop_width * crop_depth;
  const int num_crop3 = crop_width * crop_depth;
  // Output zeroing.
  OutputZeroing(outputs);
  for (int b = 0; b < nums_boxes; b++) {
    const float y1 = *(boxes + b * boxesCoordinateNum + kCoordY1);
    const float x1 = *(boxes + b * boxesCoordinateNum + kCoordX1);
    const float y2 = *(boxes + b * boxesCoordinateNum + kCoordY2);
    const float x2 = *(boxes + b * boxesCoordinateNum + kCoordX2);
    const int b_in = *(box_ind + b);
    if (b_in >= image_batch || b_in < 0) {
      continue;
    }
    const float height_ratio = (crop_height > 1) ? static_cast<float>(image_height - 1) / (crop_height - 1) : 0;
    const float width_ratio = (crop_width > 1) ? static_cast<float>(image_width - 1) / (crop_width - 1) : 0;
    const float height_scale = (crop_height > 1) ? (y2 - y1) * height_ratio : 0;
    const float width_scale = (crop_width > 1) ? (x2 - x1) * width_ratio : 0;
    for (int y = 0; y < crop_height; y++) {
      const float y_in =
        (crop_height > 1) ? y1 * (image_height - 1) + y * height_scale : 0.5 * (y1 + y2) * (image_height - 1);
      if (y_in < 0 || y_in > image_height - 1) {
        continue;
      }
      const int top_y_index = FloatToInt(floorf(y_in));
      const int bottom_y_index = FloatToInt(ceilf(y_in));
      const float y_lerp = y_in - top_y_index;
      for (int x = 0; x < crop_width; x++) {
        const float x_in =
          (crop_width > 1) ? x1 * (image_width - 1) + x * width_scale : 0.5 * (x1 + x2) * (image_width - 1);
        if (x_in < 0 || x_in > image_width - 1) {
          continue;
        }
        const int left_x_ind = FloatToInt(floorf(x_in));
        const int right_x_ind = FloatToInt(ceilf(x_in));
        const float x_lerp = x_in - left_x_ind;
        for (int d = 0; d < depth; d++) {
          const float top_left_value(
            static_cast<float>(*(image + b_in * num_image2 + top_y_index * num_image3 + left_x_ind * depth + d)));
          const float top_right_value(
            static_cast<float>(*(image + b_in * num_image2 + top_y_index * num_image3 + right_x_ind * depth + d)));
          const float bottom_left_value(
            static_cast<float>(*(image + b_in * num_image2 + bottom_y_index * num_image3 + left_x_ind * depth + d)));
          const float bottom_right_value(
            static_cast<float>(*(image + b_in * num_image2 + bottom_y_index * num_image3 + right_x_ind * depth + d)));
          // Compute the image gradient
          float image_ygrad_value =
            (1 - x_lerp) * (bottom_left_value - top_left_value) + x_lerp * (bottom_right_value - top_right_value);
          float image_xgrad_value =
            (1 - y_lerp) * (top_right_value - top_left_value) + y_lerp * (bottom_right_value - bottom_left_value);
          // Modulate the image gradient with the incoming gradient
          const float top_grad = *(grads + b * num_crop2 + y * num_crop3 + x * crop_depth + d);
          image_ygrad_value *= top_grad;
          image_xgrad_value *= top_grad;
          // dy1,dy2
          if (crop_height > 1) {
            *(outputDatas + IntToSize(b) * kCoordinateLen) += image_ygrad_value * (image_height - 1 - y * height_ratio);
            *(outputDatas + IntToSize(b) * kCoordinateLen + kCoordY2) += image_ygrad_value * (y * height_ratio);
          } else {
            *(outputDatas + IntToSize(b) * kCoordinateLen + kCoordY1) += image_ygrad_value * kNum * (image_height - 1);
            *(outputDatas + IntToSize(b) * kCoordinateLen + kCoordY2) += image_ygrad_value * kNum * (image_height - 1);
          }
          // dx1,dx2
          if (crop_width > 1) {
            *(outputDatas + IntToSize(b) * kCoordinateLen + kCoordX1) +=
              image_xgrad_value * (image_width - 1 - x * width_ratio);
            *(outputDatas + IntToSize(b) * kCoordinateLen + kCoordX2) += image_xgrad_value * (x * width_ratio);
          } else {
            *(outputDatas + IntToSize(b) * kCoordinateLen + kCoordX1) += image_xgrad_value * kNum * (image_width - 1);
            *(outputDatas + IntToSize(b) * kCoordinateLen + kCoordX2) += image_xgrad_value * kNum * (image_width - 1);
          }
        }
      }
    }
  }
  return true;
}

std::vector<std::pair<KernelAttr, CropAndResizeGradBoxesCpuKernelMod::CropAndResizeGradBoxesFunc>>
  CropAndResizeGradBoxesCpuKernelMod::func_list_ = {{KernelAttr()
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeUInt8)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeInt32)
                                                       .AddOutputAttr(kNumberTypeFloat32),
                                                     &CropAndResizeGradBoxesCpuKernelMod::LaunchKernel<uint8_t>},
                                                    {KernelAttr()
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeUInt16)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeInt32)
                                                       .AddOutputAttr(kNumberTypeFloat32),
                                                     &CropAndResizeGradBoxesCpuKernelMod::LaunchKernel<uint16_t>},
                                                    {KernelAttr()
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeInt8)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeInt32)
                                                       .AddOutputAttr(kNumberTypeFloat32),
                                                     &CropAndResizeGradBoxesCpuKernelMod::LaunchKernel<int8_t>},
                                                    {KernelAttr()
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeInt16)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeInt32)
                                                       .AddOutputAttr(kNumberTypeFloat32),
                                                     &CropAndResizeGradBoxesCpuKernelMod::LaunchKernel<int16_t>},
                                                    {KernelAttr()
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeInt32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeInt32)
                                                       .AddOutputAttr(kNumberTypeFloat32),
                                                     &CropAndResizeGradBoxesCpuKernelMod::LaunchKernel<int32_t>},
                                                    {KernelAttr()
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeInt64)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeInt32)
                                                       .AddOutputAttr(kNumberTypeFloat32),
                                                     &CropAndResizeGradBoxesCpuKernelMod::LaunchKernel<int64_t>},
                                                    {KernelAttr()
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeInt32)
                                                       .AddOutputAttr(kNumberTypeFloat32),
                                                     &CropAndResizeGradBoxesCpuKernelMod::LaunchKernel<float16>},
                                                    {KernelAttr()
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeInt32)
                                                       .AddOutputAttr(kNumberTypeFloat32),
                                                     &CropAndResizeGradBoxesCpuKernelMod::LaunchKernel<float>},
                                                    {KernelAttr()
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat64)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeInt32)
                                                       .AddOutputAttr(kNumberTypeFloat32),
                                                     &CropAndResizeGradBoxesCpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> CropAndResizeGradBoxesCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, CropAndResizeGradBoxesFunc> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, CropAndResizeGradBoxes, CropAndResizeGradBoxesCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
