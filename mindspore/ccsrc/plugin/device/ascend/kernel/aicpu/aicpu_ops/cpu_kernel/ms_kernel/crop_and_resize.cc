/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "ms_kernel/crop_and_resize.h"

#include <securec.h>
#include <algorithm>
#include <limits>
#include <vector>

#include "common/cpu_kernel_utils.h"
#include "common/kernel_log.h"
#include "common/status.h"
#include "inc/cpu_types.h"
#include "utils/kernel_util.h"
#include "utils/sparse_tensor.h"

namespace {
const char *kCropAndResize = "CropAndResize";
const size_t kInputNum = 4;
const size_t kOutputNum = 1;
const size_t kXDim = 4;
const size_t kBoxesDim = 2;
const size_t kBoxIndexDim = 1;

inline int FloatToInt(float u) {
  if (u > static_cast<float>((std::numeric_limits<int>::max)())) {
    KERNEL_LOG_ERROR("The float value(%.16f) exceeds the maximum value of int.", u);
  }
  return static_cast<int>(u);
}

inline float IntToFloat(int32_t v) { return static_cast<float>(v); }

int input_batch_;
int input_height_;
int input_width_;
int final_height_;
int final_width_;
int channel_;

template <typename T>
void BilinearResize(T *input_image, float target_x, float target_y, size_t pos, int box_index, int pos_channel,
                    float *output) {
  const int top_y_index = FloatToInt(floorf(target_y));
  const int bottom_y_index = FloatToInt(ceilf(target_y));
  const int left_x_index = FloatToInt(floorf(target_x));
  const int right_x_index = FloatToInt(ceilf(target_x));

  const float top_left = static_cast<float>(
    input_image[((box_index * input_height_ + top_y_index) * input_width_ + left_x_index) * channel_ + pos_channel]);
  const float top_right = static_cast<float>(
    input_image[((box_index * input_height_ + top_y_index) * input_width_ + right_x_index) * channel_ + pos_channel]);
  const float bottom_left = static_cast<float>(
    input_image[((box_index * input_height_ + bottom_y_index) * input_width_ + left_x_index) * channel_ + pos_channel]);
  const float bottom_right = static_cast<float>(
    input_image[((box_index * input_height_ + bottom_y_index) * input_width_ + right_x_index) * channel_ +
                pos_channel]);
  const float top = top_left + (top_right - top_left) * (target_x - left_x_index);
  const float bottom = bottom_left + (bottom_right - bottom_left) * (target_x - left_x_index);
  output[pos] = top + (bottom - top) * (target_y - top_y_index);
}

template <typename T>
void BilinearV2Resize(T *input_image, float y1, float x1, float y2, float x2, int pos_y, int pos_x, size_t pos,
                      int box_index, int pos_channel, float *output) {
  const float HALF = 0.5;
  int y1h = FloatToInt(y1 * input_height_);
  int x1w = FloatToInt(x1 * input_width_);
  int y2h = FloatToInt(y2 * input_height_);
  int x2w = FloatToInt(x2 * input_width_);
  int w = ((x2w - x1w + 1) > 1) ? x2w - x1w + 1 : 1;
  int h = ((y2h - y1h + 1) > 1) ? y2h - y1h + 1 : 1;

  float y_point = (pos_y + HALF) * (h / IntToFloat(final_height_)) - HALF;
  int top_y_index = std::min(std::max(0, FloatToInt(floorf(y_point))), h - 1);
  int bottom_y_index = std::min(std::max(0, FloatToInt(ceilf(y_point))), h - 1);

  float x_point = (pos_x + HALF) * (w / IntToFloat(final_width_)) - HALF;
  int left_x_index = std::min(std::max(0, FloatToInt(floorf(x_point))), w - 1);
  int right_x_index = std::min(std::max(0, FloatToInt(ceilf(x_point))), w - 1);

  const float y_lerp = y_point - top_y_index;
  const float x_lerp = x_point - left_x_index;

  const int y_top_index = std::max(0, y1h + top_y_index);
  const int y_bottom_index = std::max(0, y1h + bottom_y_index);
  const int x_left_index = std::max(0, x1w + left_x_index);
  const int x_right_index = std::max(0, x1w + right_x_index);

  const float top_left = static_cast<float>(
    input_image[((box_index * input_height_ + y_top_index) * input_width_ + x_left_index) * channel_ + pos_channel]);
  const float top_right = static_cast<float>(
    input_image[((box_index * input_height_ + y_top_index) * input_width_ + x_right_index) * channel_ + pos_channel]);
  const float bottom_left = static_cast<float>(
    input_image[((box_index * input_height_ + y_bottom_index) * input_width_ + x_left_index) * channel_ + pos_channel]);
  const float bottom_right = static_cast<float>(
    input_image[((box_index * input_height_ + y_bottom_index) * input_width_ + x_right_index) * channel_ +
                pos_channel]);

  output[pos] = top_left * (1 - y_lerp) * (1 - x_lerp) + bottom_right * y_lerp * x_lerp +
                top_right * (1 - y_lerp) * x_lerp + bottom_left * y_lerp * (1 - x_lerp);
}
}  // namespace

namespace aicpu {
uint32_t CropAndResizeCpuKernel::GetInputAndCheck(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "ResizeBicubic check params failed.");
  Tensor *image = ctx.Input(0);
  Tensor *out = ctx.Output(0);
  in_shape_ = image->GetTensorShape()->GetDimSizes();
  KERNEL_CHECK_FALSE((in_shape_.size() == kXDim), KERNEL_STATUS_PARAM_INVALID, "Dim of x must be 4, but got[%zu].",
                     in_shape_.size());
  auto boxes_shape = ctx.Input(1)->GetTensorShape()->GetDimSizes();
  KERNEL_CHECK_FALSE((boxes_shape.size() == kBoxesDim), KERNEL_STATUS_PARAM_INVALID,
                     "Dim of boxes must be 2, but got[%zu].", boxes_shape.size());
  auto box_index_shape = ctx.Input(2)->GetTensorShape()->GetDimSizes();
  KERNEL_CHECK_FALSE((box_index_shape.size() == kBoxIndexDim), KERNEL_STATUS_PARAM_INVALID,
                     "Dim of box_index must be 1, but got[%zu].", box_index_shape.size());
  out_shape_ = out->GetTensorShape()->GetDimSizes();
  auto out_h = out_shape_[1];
  auto out_w = out_shape_[2];
  auto out_channel = out_shape_[3];
  KERNEL_CHECK_FALSE(out_h > 0 && out_w > 0 && out_channel > 0, KERNEL_STATUS_PARAM_INVALID,
                     "output dimensions and channel must be positive but got height %lld, width %lld, channel %lld.",
                     out_h, out_w, out_channel);
  dtype_ = DataType(image->GetDataType());
  return KERNEL_STATUS_OK;
}

uint32_t CropAndResizeCpuKernel::Compute(CpuKernelContext &ctx) {
  uint32_t res = GetInputAndCheck(ctx);
  KERNEL_CHECK_FALSE((res == KERNEL_STATUS_OK), res, "GetInputAndCheck failed.");
  KERNEL_LOG_ERROR("MindSpore CropAndResize aicpu kernel.");
  if (dtype_ == DT_FLOAT16) {
    res = DoCompute<Eigen::half, float>(ctx);
  } else if (dtype_ == DT_FLOAT) {
    res = DoCompute<float, float>(ctx);
  } else if (dtype_ == DT_INT8) {
    res = DoCompute<int8_t, float>(ctx);
  } else if (dtype_ == DT_UINT8) {
    res = DoCompute<uint8_t, float>(ctx);
  } else if (dtype_ == DT_INT16) {
    res = DoCompute<int16_t, float>(ctx);
  } else if (dtype_ == DT_UINT16) {
    res = DoCompute<uint16_t, float>(ctx);
  } else if (dtype_ == DT_INT32) {
    res = DoCompute<int32_t, float>(ctx);
  } else if (dtype_ == DT_INT64) {
    res = DoCompute<int64_t, float>(ctx);
  } else if (dtype_ == DT_DOUBLE) {
    res = DoCompute<double, float>(ctx);
  } else {
    KERNEL_LOG_ERROR("ResizeBicubic doesn't support input tensor types: [%s]", DTypeStr(dtype_).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  KERNEL_CHECK_FALSE((res == KERNEL_STATUS_OK), res, "ResizeBicubic Compute failed.");
  return KERNEL_STATUS_OK;
}

template <typename T1, typename T2>
uint32_t CropAndResizeCpuKernel::DoCompute(const CpuKernelContext &ctx) {
  Tensor *image_ori = ctx.Input(0);
  Tensor *boxes_ori = ctx.Input(1);
  Tensor *box_index_ori = ctx.Input(2);
  Tensor *output_ori = ctx.Output(0);

  auto *input_image = reinterpret_cast<T1 *>(image_ori->GetData());
  auto *input_boxes = reinterpret_cast<float *>(boxes_ori->GetData());
  auto *input_box_index = reinterpret_cast<int *>(box_index_ori->GetData());
  auto *output = reinterpret_cast<T2 *>(output_ori->GetData());

  auto method_ = ctx.GetAttr("method")->GetString();
  float extrapolation_value_ = ctx.GetAttr("extrapolation_value")->GetFloat();

  input_batch_ = in_shape_[0];
  input_height_ = in_shape_[1];
  input_width_ = in_shape_[2];

  int64_t output_size = output_ori->NumElements();
  final_height_ = out_shape_[1];
  final_width_ = out_shape_[2];
  channel_ = out_shape_[3];

  auto num_boxes = out_shape_[0];
  for (int64_t b = 0; b < num_boxes; ++b) {
    auto box_idx = input_box_index[b];
    KERNEL_CHECK_FALSE(box_idx >= 0 && box_idx < static_cast<int>(input_batch_), KERNEL_STATUS_PARAM_INVALID,
                       "Invalid box_index[%lld] value: [%d], should be in [0, %lld]!", b, box_idx, b);
  }

  auto task = [&](size_t start, size_t end) {
    const float HALF = 0.5;
    for (size_t pos = start; pos < end; pos++) {
      int pos_temp = SizeToInt(pos);
      const int pos_channel = pos_temp % channel_;
      pos_temp = pos_temp / channel_;
      const int pos_x = pos_temp % final_width_;
      pos_temp = pos_temp / final_width_;
      const int pos_y = pos_temp % final_height_;
      const int pos_image_idx = pos_temp / final_height_;
      const int box_index = input_box_index[pos_image_idx];

      //  crop values
      const float y1 = input_boxes[4 * pos_image_idx];
      const float x1 = input_boxes[4 * pos_image_idx + 1];
      const float y2 = input_boxes[4 * pos_image_idx + 2];
      const float x2 = input_boxes[4 * pos_image_idx + 3];

      //  set scale and target pixels
      float scale_height = final_height_ > 1 ? (y2 - y1) * (input_height_ - 1) / (final_height_ - 1) : 0;
      float scale_width = final_width_ > 1 ? (x2 - x1) * (input_width_ - 1) / (final_width_ - 1) : 0;

      double ty = static_cast<double>(y1 * (input_height_ - 1)) + static_cast<double>(pos_y * scale_height);
      float target_y = final_height_ > 1 ? static_cast<float>(ty) : HALF * (y1 + y2) * (input_height_ - 1);

      double tx = static_cast<double>(x1 * (input_width_ - 1)) + static_cast<double>(pos_x * (scale_width));
      float target_x = final_width_ > 1 ? static_cast<float>(tx) : HALF * (x1 + x2) * (input_width_ - 1);

      //  use extrapolation value if out of range
      if (((target_x < 0) || (target_x > input_width_ - 1)) || ((target_y < 0) || (target_y > input_height_ - 1))) {
        if ((method_ == "bilinear") || (method_ == "nearest")) {
          output[pos] = extrapolation_value_;
          continue;
        }
      }

      if (method_ == "bilinear") {
        // Bilinear
        BilinearResize(input_image, target_x, target_y, pos, box_index, pos_channel, output);
      } else if (method_ == "bilinear_v2") {
        BilinearV2Resize(input_image, y1, x1, y2, x2, pos_y, pos_x, pos, box_index, pos_channel, output);
        // BilinearV2
      } else {
        // Nearest Neighbour
        const int closest_x_index = FloatToInt(roundf(target_x));
        const int closest_y_index = FloatToInt(roundf(target_y));
        const float val = static_cast<float>(
          input_image[((box_index * input_height_ + closest_y_index) * input_width_ + closest_x_index) * channel_ +
                      pos_channel]);
        output[pos] = val;
      }
    }
  };

  uint32_t min_core_num = 1;
  int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
  KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, output_size, output_size / max_core_num, task),
                      "CropAndResize compute failed.");
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kCropAndResize, CropAndResizeCpuKernel);
}  // namespace aicpu
