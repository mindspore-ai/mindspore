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
#include "backend/kernel_compiler/cpu/crop_and_resize_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {

template <typename T>
void CropAndResizeCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 4) {
    MS_LOG(ERROR) << "Input num is " << input_num << ", but CropAndResize needs 4 inputs.";
  }

  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 1) {
    MS_LOG(ERROR) << "Output num is " << output_num << ", but CropAndResize needs 1 output.";
  }

  //  input image
  auto input_image_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  size_t input_image_shape_len = input_image_shape.size();
  if (input_image_shape_len != 4) {
    MS_LOG(ERROR) << "Image tensor is " << input_image_shape_len << "-D, but CropAndResize supports only " << 4
                  << "-D image tensor.";
  }

  input_height_ = input_image_shape[1];
  input_width_ = input_image_shape[2];

  //  input boxes
  auto input_boxes_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  size_t input_boxes_shape_len = input_boxes_shape.size();
  if (input_boxes_shape_len != 2) {
    MS_LOG(ERROR) << "Box is rank " << input_boxes_shape_len << ", but CropAndResize supports only rank " << 2
                  << "for boxes.";
  }

  //  input box_index
  auto input_box_index_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
  size_t input_box_index_shape_len = input_box_index_shape.size();
  if (input_box_index_shape_len != 1) {
    MS_LOG(ERROR) << "Box index is rank " << input_box_index_shape_len << ", but CropAndResize supports only rank " << 1
                  << "for box_index.";
  }

  //  input crop_size
  auto input_crop_size_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 3);
  size_t input_crop_size_shape_len = input_crop_size_shape.size();
  if (input_crop_size_shape_len != 1) {
    MS_LOG(ERROR) << "Crop_size is rank " << input_crop_size_shape_len << "-D, but CropAndResize supports only rank "
                  << 1 << "for Crop_size.";
  }
  if (input_crop_size_shape[0] != 2) {
    MS_LOG(ERROR) << "Crop_size is size " << input_crop_size_shape[0] << "-D, but CropAndResize supports only size "
                  << 2 << "for Crop_size.";
  }

  //  output
  auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  auto output_shape_len = output_shape.size();
  output_size_ = 1;
  for (size_t i = 0; i < output_shape_len; i++) {
    output_size_ *= output_shape[i];
  }

  //  set expected output params
  final_height_ = output_shape[1];
  final_width_ = output_shape[2];
  channel_ = output_shape[3];

  //  get op parameters
  string method = AnfAlgo::GetNodeAttr<string>(kernel_node, "method");
  if (method == "bilinear") {
    method_ = 1;
  } else if (method == "nearest") {
    method_ = 2;
  } else {  //  bilinear-v2
    method_ = 3;
  }
  extrapolation_value_ = AnfAlgo::GetNodeAttr<float>(kernel_node, "extrapolation_value");
}

template <typename T>
bool CropAndResizeCPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  auto *input_image = reinterpret_cast<T *>(inputs[0]->addr);
  auto *input_boxes = reinterpret_cast<float *>(inputs[1]->addr);
  auto *input_box_index = reinterpret_cast<int *>(inputs[2]->addr);
  auto *output = reinterpret_cast<float *>(outputs[0]->addr);

  auto task = [&](size_t start, size_t end) {
    for (size_t pos = start; pos < end; pos++) {
      size_t pos_temp = pos;
      const int pos_channel = pos_temp % channel_;
      pos_temp = pos_temp / channel_;
      const int pos_x = pos_temp % final_width_;
      pos_temp = pos_temp / final_width_;
      const int pos_y = pos_temp % final_height_;
      const int pos_image_idx = pos_temp / final_height_;
      const int box_index = input_box_index[pos_image_idx];

      //  crop values
      const float y1 = input_boxes[4 * pos_image_idx + 0];
      const float x1 = input_boxes[4 * pos_image_idx + 1];
      const float y2 = input_boxes[4 * pos_image_idx + 2];
      const float x2 = input_boxes[4 * pos_image_idx + 3];

      //  set scale and target pixels
      float scale_height = final_height_ > 1 ? (y2 - y1) * (input_height_ - 1) / (final_height_ - 1) : 0;
      float scale_width = final_width_ > 1 ? (x2 - x1) * (input_width_ - 1) / (final_width_ - 1) : 0;
      float target_y =
        final_height_ > 1 ? y1 * (input_height_ - 1) + pos_y * scale_height : 0.5 * (y1 + y2) + (input_height_ - 1);
      float target_x =
        final_width_ > 1 ? x1 * (input_width_ - 1) + pos_x * scale_width : 0.5 * (x1 + x2) + (input_width_ - 1);

      //  use extrapolation value if out of range
      if (((target_x < 0) || (target_x > input_width_ - 1)) || ((target_y < 0) || (target_y > input_height_ - 1))) {
        if ((method_ == 1) || (method_ == 2)) {
          output[pos] = extrapolation_value_;
          continue;
        }
      }

      if (method_ == 1) {
        // Bilinear
        const int top_y_index = floorf(target_y);
        const int bottom_y_index = ceilf(target_y);
        const int left_x_index = floorf(target_x);
        const int right_x_index = ceilf(target_x);
        const float y_lerp = target_y - top_y_index;
        const float x_lerp = target_x - left_x_index;
        const float top_left = static_cast<float>(
          input_image[((box_index * input_height_ + top_y_index) * input_width_ + left_x_index) * channel_ +
                      pos_channel]);
        const float top_right = static_cast<float>(
          input_image[((box_index * input_height_ + top_y_index) * input_width_ + right_x_index) * channel_ +
                      pos_channel]);
        const float bottom_left = static_cast<float>(
          input_image[((box_index * input_height_ + bottom_y_index) * input_width_ + left_x_index) * channel_ +
                      pos_channel]);
        const float bottom_right = static_cast<float>(
          input_image[((box_index * input_height_ + bottom_y_index) * input_width_ + right_x_index) * channel_ +
                      pos_channel]);
        const float top = top_left + (top_right - top_left) * x_lerp;
        const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
        output[pos] = top + (bottom - top) * y_lerp;
      } else if (method_ == 3) {
        int y1h = static_cast<int>(y1 * input_height_);
        int x1w = static_cast<int>(x1 * input_width_);
        int y2h = static_cast<int>(y2 * input_height_);
        int x2w = static_cast<int>(x2 * input_width_);
        int w = ((x2w - x1w + 1) > 1) ? x2w - x1w + 1 : 1;
        int h = ((y2h - y1h + 1) > 1) ? y2h - y1h + 1 : 1;

        float y_point = (pos_y + 0.5) * (h / static_cast<float>(final_height_)) - 0.5;
        int top_y_index = floorf(y_point);
        top_y_index = std::min(std::max(0, top_y_index), h - 1);

        int bottom_y_index = ceilf(y_point);
        bottom_y_index = std::min(std::max(0, bottom_y_index), h - 1);

        float x_point = (pos_x + 0.5) * (w / static_cast<float>(final_width_)) - 0.5;
        int left_x_index = floorf(x_point);
        left_x_index = std::min(std::max(0, left_x_index), w - 1);

        int right_x_index = ceilf(x_point);
        right_x_index = std::min(std::max(0, right_x_index), w - 1);

        const float y_lerp = y_point - top_y_index;
        const float x_lerp = x_point - left_x_index;
        const int y_top_index = box_index * input_height_ + y1h + top_y_index;
        const int y_bottom_index = box_index * input_height_ + y1h + bottom_y_index;

        const float top_left =
          static_cast<float>(input_image[(y_top_index * input_width_ + x1w + left_x_index) * channel_ + pos_channel]);
        const float top_right =
          static_cast<float>(input_image[(y_top_index * input_width_ + x1w + right_x_index) * channel_ + pos_channel]);
        const float bottom_left = static_cast<float>(
          input_image[(y_bottom_index * input_width_ + x1w + left_x_index) * channel_ + pos_channel]);
        const float bottom_right = static_cast<float>(
          input_image[(y_bottom_index * input_width_ + x1w + right_x_index) * channel_ + pos_channel]);

        float ret = top_left * (1 - y_lerp) * (1 - x_lerp) + bottom_right * y_lerp * x_lerp +
                    top_right * (1 - y_lerp) * x_lerp + bottom_left * y_lerp * (1 - x_lerp);
        output[pos] = ret;
      } else {
        // Nearest Neighbour
        const int closest_x_index = roundf(target_x);
        const int closest_y_index = roundf(target_y);
        const float val = static_cast<float>(
          input_image[((box_index * input_height_ + closest_y_index) * input_width_ + closest_x_index) * channel_ +
                      pos_channel]);
        output[pos] = val;
      }
    }
  };
  CPUKernelUtils::ParallelFor(task, output_size_);
  return true;
}

}  // namespace kernel
}  // namespace mindspore
