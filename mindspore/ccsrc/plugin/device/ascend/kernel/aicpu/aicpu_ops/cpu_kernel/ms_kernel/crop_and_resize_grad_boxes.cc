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
#include "crop_and_resize_grad_boxes.h"

#include <cmath>
#include <iostream>
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

#include <chrono>
#include <cstdlib>
#include <vector>
#include "Eigen/Dense"

namespace {
constexpr uint32_t kInputNum = 4;
constexpr uint32_t kOutputNum = 1;
const char *kCropAndResizeGradBoxes = "CropAndResizeGradBoxes";
}  // namespace

namespace aicpu {
uint32_t CropAndResizeGradBoxesCpuKernel::cheakInputTypeAndGetDatas(CpuKernelContext &ctx) {
  Tensor *input_data0 = ctx.Input(0);
  Tensor *input_data1 = ctx.Input(1);
  Tensor *input_data2 = ctx.Input(2);
  Tensor *input_data3 = ctx.Input(3);
  Tensor *output = ctx.Output(0);

  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "CropAndResizeGradBoxes check params failed.");
  image_shape_ = input_data1->GetTensorShape()->GetDimSizes();
  boxes_shape_ = input_data2->GetTensorShape()->GetDimSizes();
  box_in_shape_ = input_data3->GetTensorShape()->GetDimSizes();
  grads_shape_ = input_data0->GetTensorShape()->GetDimSizes();
  output_shape_ = output->GetTensorShape()->GetDimSizes();
  KERNEL_CHECK_FALSE((grads_shape_.size() == 4), KERNEL_STATUS_PARAM_INVALID,
                     "Dim of input[0] must be 4, but the input[0] is %zu.", grads_shape_.size());
  KERNEL_CHECK_FALSE((image_shape_.size() == 4), KERNEL_STATUS_PARAM_INVALID,
                     "Dim of input[1] must be 4, but the input[1] is %zu.", image_shape_.size());

  KERNEL_CHECK_FALSE((image_shape_[1] > 0 && image_shape_[2] > 0), KERNEL_STATUS_PARAM_INVALID,
                     "the height and width of input image of "
                     "CropAndResizeGradBoxes must be over 0.");
  KERNEL_CHECK_FALSE((grads_shape_[1] > 0 && grads_shape_[2] > 0), KERNEL_STATUS_PARAM_INVALID,
                     "the height and width of input grads of "
                     "CropAndResizeGradBoxes must be over 0.");

  KERNEL_CHECK_FALSE((boxes_shape_.size() == 2), KERNEL_STATUS_PARAM_INVALID, "Dim of input[2] must be 2.");

  KERNEL_CHECK_FALSE((box_in_shape_.size() == 1), KERNEL_STATUS_PARAM_INVALID, "Dim of input[3] must be 1.");
  KERNEL_CHECK_FALSE((output_shape_.size() == 2), KERNEL_STATUS_PARAM_INVALID, "Dim of output must be 2.");
  KERNEL_CHECK_FALSE((grads_shape_[0] == boxes_shape_[0]), KERNEL_STATUS_PARAM_INVALID,
                     "boxes and grads must have compatible Batch.");
  data_type_ = input_data1->GetDataType();
  return KERNEL_STATUS_OK;
}

uint32_t CropAndResizeGradBoxesCpuKernel::Compute(CpuKernelContext &ctx) {
  uint32_t res = cheakInputTypeAndGetDatas(ctx);
  KERNEL_CHECK_FALSE((res == KERNEL_STATUS_OK), res, "GetInputAndCheck failed.");

  switch (data_type_) {
    case DT_UINT8:
      res = GradOfBoxesCompute<uint8_t>(ctx);
      break;
    case DT_UINT16:
      res = GradOfBoxesCompute<uint16_t>(ctx);
      break;
    case DT_INT8:
      res = GradOfBoxesCompute<int8_t>(ctx);
      break;
    case DT_INT16:
      res = GradOfBoxesCompute<int16_t>(ctx);
      break;
    case DT_INT32:
      res = GradOfBoxesCompute<int32_t>(ctx);
      break;
    case DT_INT64:
      res = GradOfBoxesCompute<int64_t>(ctx);
      break;
    case DT_FLOAT16:
      res = GradOfBoxesCompute<Eigen::half>(ctx);
      break;
    case DT_FLOAT:
      res = GradOfBoxesCompute<float>(ctx);
      break;
    case DT_DOUBLE:
      res = GradOfBoxesCompute<double>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR("CropAndResizeGradBoxes op doesn't support input tensor types: [%s]",
                       DTypeStr(data_type_).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  KERNEL_CHECK_FALSE((res == KERNEL_STATUS_OK), res, "CropAndResizeGradBoxes Compute failed.");
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t CropAndResizeGradBoxesCpuKernel::GradOfBoxesCompute(CpuKernelContext &ctx) {
  Tensor *grads_tensor = ctx.Input(0);
  Tensor *image_tensor = ctx.Input(1);
  Tensor *boxes_tensor = ctx.Input(2);
  Tensor *box_ind_tensor = ctx.Input(3);
  Tensor *output_tensor = ctx.Output(0);
  const float *grads = reinterpret_cast<float *>(grads_tensor->GetData());
  const T *image = reinterpret_cast<T *>(image_tensor->GetData());
  const float *boxes = reinterpret_cast<float *>(boxes_tensor->GetData());
  float *outputDatas = reinterpret_cast<float *>(output_tensor->GetData());
  const int32_t *box_ind = reinterpret_cast<int32_t *>(box_ind_tensor->GetData());
  const int image_batch = image_shape_[0];
  const int image_height = image_shape_[1];
  const int image_width = image_shape_[2];
  const int depth = image_shape_[3];

  const int nums_boxes = grads_shape_[0];
  const int crop_height = grads_shape_[1];
  const int crop_width = grads_shape_[2];
  const int crop_depth = grads_shape_[3];
  KERNEL_CHECK_FALSE((depth == crop_depth), KERNEL_STATUS_PARAM_INVALID, "boxes and grads must have compatible Depth.");

  const int boxesCoordinateNum = boxes_shape_[1];
  const int num_image2 = image_height * image_width * depth;
  const int num_image3 = image_width * depth;
  const int num_crop2 = crop_height * crop_width * crop_depth;
  const int num_crop3 = crop_width * crop_depth;
  // Output zeroing.
  int num = nums_boxes * 4;
  for (int i = 0; i < num; i++) {
    *(outputDatas + i) = 0;
  }

  for (int b = 0; b < nums_boxes; b++) {
    const float y1 = *(boxes + b * boxesCoordinateNum + 0);
    const float x1 = *(boxes + b * boxesCoordinateNum + 1);
    const float y2 = *(boxes + b * boxesCoordinateNum + 2);
    const float x2 = *(boxes + b * boxesCoordinateNum + 3);
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

      const int top_y_index = floorf(y_in);
      const int bottom_y_index = ceilf(y_in);
      const float y_lerp = y_in - top_y_index;

      for (int x = 0; x < crop_width; x++) {
        const float x_in =
          (crop_width > 1) ? x1 * (image_width - 1) + x * width_scale : 0.5 * (x1 + x2) * (image_width - 1);

        if (x_in < 0 || x_in > image_width - 1) {
          continue;
        }
        const int left_x_index = floorf(x_in);
        const int right_x_index = ceilf(x_in);
        const float x_lerp = x_in - left_x_index;

        for (int d = 0; d < depth; d++) {
          const float top_left_value(
            static_cast<float>(*(image + b_in * num_image2 + top_y_index * num_image3 + left_x_index * depth + d)));
          const float top_right_value(
            static_cast<float>(*(image + b_in * num_image2 + top_y_index * num_image3 + right_x_index * depth + d)));
          const float bottom_left_value(
            static_cast<float>(*(image + b_in * num_image2 + bottom_y_index * num_image3 + left_x_index * depth + d)));
          const float bottom_right_value(
            static_cast<float>(*(image + b_in * num_image2 + bottom_y_index * num_image3 + right_x_index * depth + d)));
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
            *(outputDatas + b * 4 + 0) += image_ygrad_value * (image_height - 1 - y * height_ratio);
            *(outputDatas + b * 4 + 2) += image_ygrad_value * (y * height_ratio);
          } else {
            *(outputDatas + b * 4 + 0) += image_ygrad_value * 0.5 * (image_height - 1);
            *(outputDatas + b * 4 + 2) += image_ygrad_value * 0.5 * (image_height - 1);
          }
          // dx1,dx2
          if (crop_width > 1) {
            *(outputDatas + b * 4 + 1) += image_xgrad_value * (image_width - 1 - x * width_ratio);
            *(outputDatas + b * 4 + 3) += image_xgrad_value * (x * width_ratio);
          } else {
            *(outputDatas + b * 4 + 1) += image_xgrad_value * 0.5 * (image_width - 1);
            *(outputDatas + b * 4 + 3) += image_xgrad_value * 0.5 * (image_width - 1);
          }
        }
      }
    }
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kCropAndResizeGradBoxes, CropAndResizeGradBoxesCpuKernel);
}  // namespace aicpu
