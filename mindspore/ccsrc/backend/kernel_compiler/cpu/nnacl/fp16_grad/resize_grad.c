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
#include "nnacl/fp16_grad/resize_grad.h"
#include <math.h>
#include "nnacl/infer/common_infer.h"

int ResizeNearestNeighborFp16Grad(float16_t *in_addr, float16_t *out_addr, int batch_size, int channel, int format,
                                  ResizeFp16GradParameter *param) {
  bool align_corners = param->align_corners_;
  size_t in_hw_size = param->in_width_ * param->in_height_;
  size_t out_hw_size = param->out_width_ * param->out_height_;

  if (format == Format_NHWC) {
    NNACL_CHECK_ZERO_RETURN_ERR(param->in_width_);
    for (int32_t b = 0; b < batch_size; ++b) {
      for (size_t i = 0; i < in_hw_size; ++i) {
        size_t in_y = i / param->in_width_;
        size_t in_x = i % param->in_width_;
        for (int32_t c = 0; c < channel; ++c) {
          size_t out_y = MSMIN(
            (align_corners) ? (size_t)roundf(in_y * param->height_scale_) : (size_t)floorf(in_y * param->height_scale_),
            param->out_height_ - 1);
          size_t out_x = MSMIN(
            (align_corners) ? (size_t)roundf(in_x * param->width_scale_) : (size_t)floorf(in_x * param->width_scale_),
            param->out_width_ - 1);
          size_t out_offset = out_y * (param->out_width_ * channel) + (out_x * channel) + c;
          size_t in_offset = in_y * (param->in_width_ * channel) + (in_x * channel) + c;
          out_addr[out_offset] += in_addr[in_offset];
        }
      }
      out_addr += out_hw_size * channel;
      in_addr += in_hw_size * channel;
    }
  } else if (format == Format_NCHW) {
    for (int32_t b = 0; b < batch_size; ++b) {
      for (int32_t c = 0; c < channel; ++c) {
        for (size_t h = 0; h < param->in_height_; ++h) {
          size_t out_y =
            MSMIN((align_corners) ? (size_t)roundf(h * param->height_scale_) : (size_t)floorf(h * param->height_scale_),
                  param->out_height_ - 1);
          for (size_t w = 0; w < param->in_width_; ++w) {
            size_t out_x =
              MSMIN((align_corners) ? (size_t)roundf(w * param->width_scale_) : (size_t)floorf(w * param->width_scale_),
                    param->out_width_ - 1);
            out_addr[out_y * param->out_width_ + out_x] += in_addr[h * param->in_width_ + w];
          }
        }
        out_addr += out_hw_size;
        in_addr += in_hw_size;
      }
    }
  }
  return NNACL_OK;
}

int ResizeBiLinearFp16Grad(float16_t *in_addr, float16_t *out_addr, int batch_size, int channel, int format,
                           ResizeFp16GradParameter *param) {
  size_t in_hw_size = param->in_width_ * param->in_height_;
  size_t out_hw_size = param->out_width_ * param->out_height_;

  if (format == Format_NHWC) {
    NNACL_CHECK_ZERO_RETURN_ERR(param->in_width_);
    for (int32_t b = 0; b < batch_size; ++b) {
      for (size_t i = 0; i < in_hw_size; ++i) {
        size_t h = i / param->in_width_;
        size_t w = i % param->in_width_;
        for (int32_t c = 0; c < channel; ++c) {
          float16_t in_y = (float16_t)h * param->height_scale_;
          size_t top_y_index = MSMAX((size_t)(floorf(in_y)), (size_t)(0));
          size_t bottom_y_index = MSMIN((size_t)(ceilf(in_y)), param->out_height_ - 1);
          float16_t y_lerp = in_y - floorf(in_y);
          const float16_t inverse_y_lerp = 1.0 - y_lerp;

          float16_t in_x = (float16_t)w * param->width_scale_;
          size_t left_x_index = MSMAX((size_t)(floorf(in_x)), (size_t)(0));
          size_t right_x_index = MSMIN((size_t)(ceilf(in_x)), param->out_width_ - 1);
          float16_t x_lerp = in_x - floorf(in_x);
          const float16_t inverse_x_lerp = 1.0 - x_lerp;

          size_t in_offset = h * (param->in_width_ * channel) + (w * channel) + c;
          size_t out_offset_top_y_left_x = top_y_index * (param->out_width_ * channel) + (left_x_index * channel) + c;
          size_t out_offset_top_y_right_x = top_y_index * (param->out_width_ * channel) + (right_x_index * channel) + c;
          size_t out_offset_bottom_y_left_x =
            bottom_y_index * (param->out_width_ * channel) + (left_x_index * channel) + c;
          size_t out_offset_bottom_y_right_x =
            bottom_y_index * (param->out_width_ * channel) + (right_x_index * channel) + c;

          out_addr[out_offset_top_y_left_x] += in_addr[in_offset] * (float16_t)(inverse_y_lerp * inverse_x_lerp);
          out_addr[out_offset_top_y_right_x] += in_addr[in_offset] * (float16_t)(inverse_y_lerp * x_lerp);
          out_addr[out_offset_bottom_y_left_x] += in_addr[in_offset] * (float16_t)(y_lerp * inverse_x_lerp);
          out_addr[out_offset_bottom_y_right_x] += in_addr[in_offset] * (float16_t)(y_lerp * x_lerp);
        }
      }
      out_addr += out_hw_size * channel;
      in_addr += in_hw_size * channel;
    }
  } else if (format == Format_NCHW) {
    size_t in_height = param->in_height_;
    size_t in_width = param->in_width_;
    size_t out_height = param->out_height_;
    size_t out_width = param->out_width_;

    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t c = 0; c < channel; ++c) {
        for (size_t h = 0; h < in_height; ++h) {
          const float16_t in_y = (float16_t)(h)*param->height_scale_;
          const size_t top_y_index = MSMAX((size_t)floorf(in_y), 0);
          const size_t bottom_y_index = MSMIN((size_t)ceilf(in_y), out_height - 1);
          const float16_t y_lerp = in_y - floorf(in_y);
          const float16_t inverse_y_lerp = 1.0 - y_lerp;
          for (size_t w = 0; w < in_width; ++w) {
            const float16_t in_x = (float16_t)(w)*param->width_scale_;
            const size_t left_x_index = MSMAX((size_t)floorf(in_x), 0);
            const size_t right_x_index = MSMIN((size_t)ceilf(in_x), out_width - 1);
            const float16_t x_lerp = in_x - floorf(in_x);
            const float16_t inverse_x_lerp = 1.0 - x_lerp;
            out_addr[top_y_index * out_width + left_x_index] +=
              in_addr[h * in_width + w] * (float16_t)(inverse_y_lerp * inverse_x_lerp);
            out_addr[top_y_index * out_width + right_x_index] +=
              in_addr[h * in_width + w] * (float16_t)(inverse_y_lerp * x_lerp);
            out_addr[bottom_y_index * out_width + left_x_index] +=
              in_addr[h * in_width + w] * (float16_t)(y_lerp * inverse_x_lerp);
            out_addr[bottom_y_index * out_width + right_x_index] +=
              in_addr[h * in_width + w] * (float16_t)(y_lerp * x_lerp);
          }
        }
        out_addr += out_hw_size;
        in_addr += in_hw_size;
      }
    }
  }
  return NNACL_OK;
}
