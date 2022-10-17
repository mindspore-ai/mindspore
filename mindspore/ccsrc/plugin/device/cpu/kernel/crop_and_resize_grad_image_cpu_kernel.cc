/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/crop_and_resize_grad_image_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace {
constexpr size_t kInputNumsImage = 4;
constexpr size_t kOutNumImage = 1;
constexpr size_t kGradsImage = 0;
constexpr size_t kGradsShapeLenImage = 4;
constexpr size_t kNumBoxesImage = 0;
constexpr size_t kHeightImage = 1;
constexpr size_t kWidthImage = 2;
constexpr size_t kDepthImage = 3;
constexpr size_t kBatchImage = 0;
constexpr size_t kImageSizeImage = 3;
constexpr size_t kImageSizeShapeLenImage = 1;
constexpr size_t kBoxesImage = 1;
constexpr size_t kCoordY1Image = 0;
constexpr size_t kCoordX1Image = 1;
constexpr size_t kCoordY2Image = 2;
constexpr size_t kCoordX2Image = 3;
constexpr size_t kBoxesShapeLenImage = 2;
constexpr size_t kCoordinateLenImage = 4;
constexpr size_t kBoxIndexImage = 2;
constexpr size_t kBoxIndexShapeLenImage = 1;
constexpr size_t kOutputIndexImage = 0;
constexpr size_t kOutputShapeLenImage = 4;
constexpr float kNumImage = 0.5;
}  // namespace

namespace mindspore {
namespace kernel {
void CropAndResizeGradImageCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  cnode_ptr_ = kernel_node;
  attr_method_ = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, "method");

  //  input grads
  grads_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kGradsImage);
  size_t input_grads_shape_len = grads_shape_.size();
  if (input_grads_shape_len != kGradsShapeLenImage) {
    MS_LOG(ERROR) << "Grads tensor is " << input_grads_shape_len << "-D, but CropAndResizeGradImage supports only "
                  << kGradsShapeLenImage << "-D image tensor.";
  }

  //  input image_size
  image_size_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kImageSizeImage);
  size_t input_image_size_shape_len = image_size_shape_.size();
  if (input_image_size_shape_len != kImageSizeShapeLenImage) {
    MS_LOG(ERROR) << "Image_size tensor is " << input_image_size_shape_len
                  << "-D, but CropAndResizeGradImage supports only " << kImageSizeShapeLenImage
                  << "-D image_size tensor.";
  }

  //  input boxes
  boxes_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kBoxesImage);
  size_t input_boxes_shape_len = boxes_shape_.size();
  if (input_boxes_shape_len != kBoxesShapeLenImage) {
    MS_LOG(ERROR) << "Boxes tensor is " << input_boxes_shape_len << ", but CropAndResizeGradImage supports only "
                  << kBoxesShapeLenImage << "-D for boxes.";
  }
  if (boxes_shape_[1] != kCoordinateLenImage) {
    MS_LOG(ERROR) << "The coordinate size of boxes is " << boxes_shape_[1]
                  << ", but CropAndResizeGradImage supports only " << kCoordinateLenImage << "for boxes.";
  }

  //  input box_index
  box_ind_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kBoxIndexImage);
  size_t input_box_index_shape_len = box_ind_shape_.size();
  if (input_box_index_shape_len != kBoxIndexShapeLenImage) {
    MS_LOG(ERROR) << "Box_index tensor is " << input_box_index_shape_len
                  << "-D, but CropAndResizeGradBoxes supports only " << kBoxIndexShapeLenImage << "-D for box_index.";
  }

  //  output
  output_shape_ = common::AnfAlgo::GetOutputInferShape(kernel_node, kOutputIndexImage);
  auto output_shape_len = output_shape_.size();
  if (output_shape_len != kOutputShapeLenImage) {
    MS_LOG(ERROR) << "Output tensor is " << output_shape_len << ", but CropAndResizeGradImage supports only "
                  << kOutputShapeLenImage << "-D for output tensor.";
  }

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "CropAndResizeGradImage does not support this kernel data type: " << kernel_attr;
  }

  kernel_func_ = func_list_[index].second;
}

template <typename T>
bool CropAndResizeGradImageCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                      const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNumsImage, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutNumImage, kernel_name_);
  auto *grads = reinterpret_cast<float *>(inputs[kGradsImage]->addr);
  auto *image_size = reinterpret_cast<int *>(inputs[kImageSizeImage]->addr);
  auto *boxes = reinterpret_cast<float *>(inputs[kBoxesImage]->addr);
  auto *box_ind = reinterpret_cast<int *>(inputs[kBoxIndexImage]->addr);

  const int64_t image_batch = *(image_size + kBatchImage);
  const int64_t image_height = *(image_size + kHeightImage);
  const int64_t image_width = *(image_size + kWidthImage);
  const int64_t image_depth = *(image_size + kDepthImage);
  if (image_height <= 0 || image_width <= 0) {
    MS_EXCEPTION(ValueError) << "image_height and image_width of image_size must be positive.";
  }
  const int64_t nums_boxes = grads_shape_[kNumBoxesImage];
  const int64_t crop_depth = grads_shape_[kDepthImage];
  if (image_depth != crop_depth) {
    MS_EXCEPTION(ValueError) << "shape[3] of grads and image_size[3] must be equal.";
  }
  // output infer
  auto node_ = cnode_ptr_.lock();
  std::vector<TypeId> dtypes{AnfAlgo::GetOutputDeviceDataType(node_, 0)};
  std::vector<int64_t> output_shape = {image_batch, image_height, image_width, image_depth};
  output_shape_ = output_shape;
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, {output_shape}, node_.get());
  const int64_t num_image1 = image_batch * image_height * image_width * image_depth;
  auto *outputDatas = reinterpret_cast<T *>(outputs[0]->addr);
  // set the output data to 0.
  T temp = static_cast<T>(0.0);
  for (int64_t i = 0; i < num_image1; i++) {
    *(outputDatas + i) = temp;
  }
  auto task = [this, &grads, &image_size, &boxes, &box_ind, &outputDatas](size_t start, size_t end) {
    GradOfImageCompute<T>(grads, boxes, box_ind, image_size, outputDatas, start, end);
  };
  CPUKernelUtils::ParallelFor(task, nums_boxes);
  return true;
}

template <typename T>
void CropAndResizeGradImageCpuKernelMod::GradOfImageCompute(const float *grads, const float *boxes, const int *box_ind,
                                                            const int *image_size, T *outputDatas, size_t start,
                                                            size_t end) {
  const int64_t image_batch = *(image_size + kBatchImage);
  const int64_t image_height = *(image_size + kHeightImage);
  const int64_t image_width = *(image_size + kWidthImage);
  const int64_t depth = *(image_size + kDepthImage);

  const int64_t crop_height = grads_shape_[kHeightImage];
  const int64_t crop_width = grads_shape_[kWidthImage];
  const int64_t crop_depth = grads_shape_[kDepthImage];
  const int64_t boxesCoordinateNum = boxes_shape_[1];

  const int64_t num_image2 = image_height * image_width * depth;
  const int64_t num_image3 = image_width * depth;
  const int64_t num_crop2 = crop_height * crop_width * crop_depth;
  const int64_t num_crop3 = crop_width * crop_depth;

  for (size_t b = start; b < end; b++) {
    const float y1 = *(boxes + b * boxesCoordinateNum + kCoordY1Image);
    const float x1 = *(boxes + b * boxesCoordinateNum + kCoordX1Image);
    const float y2 = *(boxes + b * boxesCoordinateNum + kCoordY2Image);
    const float x2 = *(boxes + b * boxesCoordinateNum + kCoordX2Image);
    const int64_t b_in = *(box_ind + b);
    if (b_in < 0 || b_in > image_batch - 1) {
      MS_EXCEPTION(ValueError) << "box_index has values outside [0, batch_size).";
    }
    float height_scale = 0;
    float width_scale = 0;
    if (crop_height > 1) {
      height_scale = (y2 - y1) * (image_height - 1) / (crop_height - 1);
    }
    if (crop_width > 1) {
      width_scale = (x2 - x1) * (image_width - 1) / (crop_width - 1);
    }
    for (int64_t y = 0; y < crop_height; y++) {
      float in_y = kNumImage * (y1 + y2) * (image_height - 1);
      if (crop_height > 1) {
        in_y = y1 * (image_height - 1) + y * height_scale;
      }
      if (in_y < 0 || in_y > image_height - 1) {
        continue;
      }

      const int64_t top_y_index = floorf(in_y);
      const int64_t bottom_y_index = ceilf(in_y);
      const float y_lerp = in_y - top_y_index;
      for (int64_t x = 0; x < crop_width; x++) {
        float in_x = kNumImage * (x1 + x2) * (image_width - 1);
        if (crop_width > 1) {
          in_x = x1 * (image_width - 1) + x * width_scale;
        }
        if (in_x < 0 || in_x > image_width - 1) {
          continue;
        }
        if (attr_method_ == "bilinear") {
          const int64_t left_x_index = floorf(in_x);
          const int64_t right_x_index = ceilf(in_x);
          const float x_lerp = in_x - left_x_index;

          for (int64_t d = 0; d < depth; d++) {
            const float dtop = (*(grads + b * num_crop2 + y * num_crop3 + x * crop_depth + d)) * (1 - y_lerp);
            *(outputDatas + b_in * num_image2 + top_y_index * num_image3 + left_x_index * depth + d) +=
              static_cast<T>((1 - x_lerp) * dtop);
            *(outputDatas + b_in * num_image2 + top_y_index * num_image3 + right_x_index * depth + d) +=
              static_cast<T>(x_lerp * dtop);
            const float dbottom = (*(grads + b * num_crop2 + y * num_crop3 + x * crop_depth + d)) * y_lerp;
            *(outputDatas + b_in * num_image2 + bottom_y_index * num_image3 + left_x_index * depth + d) +=
              static_cast<T>((1 - x_lerp) * dbottom);
            *(outputDatas + b_in * num_image2 + bottom_y_index * num_image3 + right_x_index * depth + d) +=
              static_cast<T>(x_lerp * dbottom);
          }
        } else {
          for (int64_t d = 0; d < depth; d++) {
            const int close_x_index = roundf(in_x);
            const int close_y_index = roundf(in_y);
            *(outputDatas + b_in * num_image2 + close_y_index * num_image3 + close_x_index * depth + d) +=
              static_cast<T>(*(grads + b * num_crop2 + y * num_crop3 + x * crop_depth + d));
          }
        }
      }
    }
  }
}

std::vector<std::pair<KernelAttr, CropAndResizeGradImageCpuKernelMod::CropAndResizeGradImageFunc>>
  CropAndResizeGradImageCpuKernelMod::func_list_ = {{KernelAttr()
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeInt32)
                                                       .AddInputAttr(kNumberTypeInt32)
                                                       .AddOutputAttr(kNumberTypeFloat16),
                                                     &CropAndResizeGradImageCpuKernelMod::LaunchKernel<float16>},
                                                    {KernelAttr()
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeInt32)
                                                       .AddInputAttr(kNumberTypeInt32)
                                                       .AddOutputAttr(kNumberTypeFloat32),
                                                     &CropAndResizeGradImageCpuKernelMod::LaunchKernel<float>},
                                                    {KernelAttr()
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeInt32)
                                                       .AddInputAttr(kNumberTypeInt32)
                                                       .AddOutputAttr(kNumberTypeFloat64),
                                                     &CropAndResizeGradImageCpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> CropAndResizeGradImageCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, CropAndResizeGradImageFunc> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, CropAndResizeGradImage, CropAndResizeGradImageCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
