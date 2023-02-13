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
#include "crop_and_resize_grad_image.h"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

#include <cmath>
#include <iostream>

namespace {
constexpr uint32_t kInputNum = 4;
constexpr uint32_t kOutputNum = 1;
const char *kCropAndResizeGradImage = "CropAndResizeGradImage";
}  // namespace

namespace aicpu {
uint32_t CropAndResizeGradImageCpuKernel::cheakInputTypeAndGetDatas(CpuKernelContext &ctx) {
  Tensor *grads = ctx.Input(0);
  Tensor *boxes = ctx.Input(1);
  Tensor *box_index = ctx.Input(2);
  Tensor *image_size = ctx.Input(3);
  Tensor *output = ctx.Output(0);
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "CropAndResizeGradImage check params failed.");
  grads_shape_ = grads->GetTensorShape()->GetDimSizes();
  boxes_shape_ = boxes->GetTensorShape()->GetDimSizes();
  box_ind_shape_ = box_index->GetTensorShape()->GetDimSizes();
  image_size_shape_ = image_size->GetTensorShape()->GetDimSizes();
  output_shape_ = output->GetTensorShape()->GetDimSizes();

  KERNEL_CHECK_FALSE((grads_shape_.size() == 4), KERNEL_STATUS_PARAM_INVALID,
                     "Dim of grads must be 4, but the grads is %zu.", grads_shape_.size());
  KERNEL_CHECK_FALSE((boxes_shape_.size() == 2), KERNEL_STATUS_PARAM_INVALID,
                     "Dim of boxes must be 2, but the boxes is %zu.", boxes_shape_.size());

  KERNEL_CHECK_FALSE((box_ind_shape_.size() == 1), KERNEL_STATUS_PARAM_INVALID, "Dim of box_index must be 1.");

  KERNEL_CHECK_FALSE((image_size_shape_.size() == 1 && image_size_shape_[0] == 4), KERNEL_STATUS_PARAM_INVALID,
                     "the input of image_size must be 1D and have 4 elements.");
  KERNEL_CHECK_FALSE((output_shape_.size() == 4), KERNEL_STATUS_PARAM_INVALID, "Dim of output must be 4.");

  KERNEL_CHECK_FALSE((grads_shape_[1] > 0 && grads_shape_[2] > 0), KERNEL_STATUS_PARAM_INVALID,
                     "grads dimensions must be positive.");
  KERNEL_CHECK_FALSE((grads_shape_[0] == boxes_shape_[0]), KERNEL_STATUS_PARAM_INVALID,
                     "boxes and grads have incompatible shape.");
  data_type_ = output->GetDataType();
  return KERNEL_STATUS_OK;
}

uint32_t CropAndResizeGradImageCpuKernel::Compute(CpuKernelContext &ctx) {
  uint32_t res = cheakInputTypeAndGetDatas(ctx);
  KERNEL_CHECK_FALSE((res == KERNEL_STATUS_OK), res, "GetInputAndCheck failed.");
  switch (data_type_) {
    case DT_FLOAT16:
      res = GradOfImageComputeShared<Eigen::half>(ctx);
      break;
    case DT_FLOAT:
      res = GradOfImageComputeShared<float>(ctx);
      break;
    case DT_DOUBLE:
      res = GradOfImageComputeShared<double>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR("CropAndResizeGradImage op doesn't support input tensor types: [%s]",
                       DTypeStr(data_type_).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  KERNEL_CHECK_FALSE((res == KERNEL_STATUS_OK), res, "CropAndResizeGradImage Compute failed.");
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t CropAndResizeGradImageCpuKernel::GradOfImageCompute(CpuKernelContext &ctx, int64_t start, int64_t end) {
  Tensor *grads_tensor = ctx.Input(0);
  Tensor *boxes_tensor = ctx.Input(1);
  Tensor *box_index_tensor = ctx.Input(2);
  Tensor *image_size_tensor = ctx.Input(3);
  Tensor *output_tensor = ctx.Output(0);
  float *grads = reinterpret_cast<float *>(grads_tensor->GetData());
  int32_t *image_size = reinterpret_cast<int32_t *>(image_size_tensor->GetData());
  float *boxes = reinterpret_cast<float *>(boxes_tensor->GetData());
  T *outputDatas = reinterpret_cast<T *>(output_tensor->GetData());
  int32_t *box_index = reinterpret_cast<int32_t *>(box_index_tensor->GetData());

  const int64_t image_batch = *(image_size + 0);
  const int64_t image_height = *(image_size + 1);
  const int64_t image_width = *(image_size + 2);
  const int64_t depth = *(image_size + 3);
  const int64_t crop_height = grads_shape_[1];
  const int64_t crop_width = grads_shape_[2];
  const int64_t crop_depth = grads_shape_[3];
  const int64_t boxesCoordinateNum = boxes_shape_[1];
  const int64_t num_image2 = image_height * image_width * depth;
  const int64_t num_image3 = image_width * depth;
  const int64_t num_crop2 = crop_height * crop_width * crop_depth;
  const int64_t num_crop3 = crop_width * crop_depth;

  for (int64_t b = start; b < end; b++) {
    const float y1 = *(boxes + b * boxesCoordinateNum + 0);
    const float x1 = *(boxes + b * boxesCoordinateNum + 1);
    const float y2 = *(boxes + b * boxesCoordinateNum + 2);
    const float x2 = *(boxes + b * boxesCoordinateNum + 3);
    const int64_t b_in = *(box_index + b);
    if (b_in < 0 || b_in > image_batch - 1) {
      continue;
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
      float in_y = 0.5 * (y1 + y2) * (image_height - 1);
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
        float in_x = 0.5 * (x1 + x2) * (image_width - 1);
        if (crop_width > 1) {
          in_x = x1 * (image_width - 1) + x * width_scale;
        }
        if (in_x < 0 || in_x > image_width - 1) {
          continue;
        }
        AttrValue *attr = ctx.GetAttr("method");
        std::string str = attr->GetString();
        if (str == "bilinear") {
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
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t CropAndResizeGradImageCpuKernel::GradOfImageComputeShared(CpuKernelContext &ctx) {
  Tensor *image_size_tensor = ctx.Input(3);
  Tensor *output_tensor = ctx.Output(0);
  int32_t *image_size = reinterpret_cast<int32_t *>(image_size_tensor->GetData());
  T *outputDatas = reinterpret_cast<T *>(output_tensor->GetData());

  const int64_t image_height = *(image_size + 1);
  const int64_t image_width = *(image_size + 2);
  const int64_t depth = *(image_size + 3);
  KERNEL_CHECK_FALSE((image_height > 0 && image_width > 0), KERNEL_STATUS_PARAM_INVALID,
                     "image dimensions must be positive.");
  const int64_t nums_boxes = grads_shape_[0];
  const int64_t crop_depth = grads_shape_[3];
  KERNEL_CHECK_FALSE((depth == crop_depth), KERNEL_STATUS_PARAM_INVALID, "image_size and grads are incompatible.");
  const int64_t num_image1 = nums_boxes * image_height * image_width * depth;

  // Set the output data to 0.
  T temp = static_cast<T>(0.0);
  for (int i = 0; i < num_image1; i++) {
    *(outputDatas + i) = temp;
  }

  auto shared_CropAndResizeGradImage = [&](size_t start, size_t end) {
    uint32_t res = GradOfImageCompute<T>(ctx, start, end);
    return res;
  };
  CpuKernelUtils::ParallelFor(ctx, nums_boxes, 1, shared_CropAndResizeGradImage);

  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kCropAndResizeGradImage, CropAndResizeGradImageCpuKernel);
}  // namespace aicpu