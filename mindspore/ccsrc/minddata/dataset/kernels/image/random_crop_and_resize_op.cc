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
#include "minddata/dataset/kernels/image/random_crop_and_resize_op.h"
#include <limits>
#include <random>

#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
const float RandomCropAndResizeOp::kDefScaleLb = 0.08;
const float RandomCropAndResizeOp::kDefScaleUb = 1.0;
const float RandomCropAndResizeOp::kDefAspectLb = 0.75;
const float RandomCropAndResizeOp::kDefAspectUb = 1.333333;
const InterpolationMode RandomCropAndResizeOp::kDefInterpolation = InterpolationMode::kLinear;
const int32_t RandomCropAndResizeOp::kDefMaxIter = 10;

RandomCropAndResizeOp::RandomCropAndResizeOp(int32_t target_height, int32_t target_width, float scale_lb,
                                             float scale_ub, float aspect_lb, float aspect_ub,
                                             InterpolationMode interpolation, int32_t max_attempts)
    : target_height_(target_height),
      target_width_(target_width),
      rnd_scale_(scale_lb, scale_ub),
      rnd_aspect_(log(aspect_lb), log(aspect_ub)),
      interpolation_(interpolation),
      aspect_lb_(aspect_lb),
      aspect_ub_(aspect_ub),
      max_iter_(max_attempts) {
  rnd_.seed(GetSeed());
  is_deterministic_ = false;
}

Status RandomCropAndResizeOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  if (input.size() != 1) {
    for (size_t i = 0; i < input.size() - 1; i++) {
      if (input[i]->Rank() != 2 && input[i]->Rank() != 3) {
        std::string err_msg = "RandomCropAndResizeOp: image shape is not <H,W,C> or <H, W>, but got rank:" +
                              std::to_string(input[i]->Rank());
        RETURN_STATUS_UNEXPECTED(err_msg);
      }
      if (input[i]->shape()[0] != input[i + 1]->shape()[0] || input[i]->shape()[1] != input[i + 1]->shape()[1]) {
        RETURN_STATUS_UNEXPECTED("RandomCropAndResizeOp: Input images must have the same size.");
      }
    }
  }
  const int output_count = input.size();
  output->resize(output_count);
  int x = 0;
  int y = 0;
  int crop_height = 0;
  int crop_width = 0;
  for (size_t i = 0; i < input.size(); i++) {
    RETURN_IF_NOT_OK(ValidateImageRank("RandomCropAndResize", input[i]->shape().Size()));
    int h_in = input[i]->shape()[0];
    int w_in = input[i]->shape()[1];
    if (i == 0) {
      RETURN_IF_NOT_OK(GetCropBox(h_in, w_in, &x, &y, &crop_height, &crop_width));
    }
    RETURN_IF_NOT_OK(CropAndResize(input[i], &(*output)[i], x, y, crop_height, crop_width, target_height_,
                                   target_width_, interpolation_));
  }
  return Status::OK();
}

Status RandomCropAndResizeOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  TensorShape out = TensorShape{target_height_, target_width_};
  if (inputs[0].Rank() == 2) {
    (void)outputs.emplace_back(out);
  }
  if (inputs[0].Rank() == 3) {
    (void)outputs.emplace_back(out.AppendDim(inputs[0][2]));
  }
  if (!outputs.empty()) {
    return Status::OK();
  }
  return Status(StatusCode::kMDUnexpectedError,
                "RandomCropAndResize: invalid input shape, expected 2D or 3D input, but got input dimension is: " +
                  std::to_string(inputs[0].Rank()));
}
Status RandomCropAndResizeOp::GetCropBox(int h_in, int w_in, int *x, int *y, int *crop_height, int *crop_width) {
  CHECK_FAIL_RETURN_UNEXPECTED(crop_height != nullptr, "crop_height is nullptr.");
  CHECK_FAIL_RETURN_UNEXPECTED(crop_width != nullptr, "crop_width is nullptr.");
  *crop_width = w_in;
  *crop_height = h_in;
  CHECK_FAIL_RETURN_UNEXPECTED(w_in != 0, "RandomCropAndResize: Width cannot be 0.");
  CHECK_FAIL_RETURN_UNEXPECTED(h_in != 0, "RandomCropAndResize: Height cannot be 0.");
  CHECK_FAIL_RETURN_UNEXPECTED(aspect_lb_ > 0, "RandomCropAndResize: aspect lower bound must be greater than zero.");
  for (int32_t i = 0; i < max_iter_; i++) {
    double const sample_scale = rnd_scale_(rnd_);
    // In case of non-symmetrical aspect ratios, use uniform distribution on a logarithmic sample_scale.
    // Note rnd_aspect_ is already a random distribution of the input aspect ratio in logarithmic sample_scale.
    double const sample_aspect = exp(rnd_aspect_(rnd_));

    CHECK_FAIL_RETURN_UNEXPECTED(
      (std::numeric_limits<int32_t>::max() / h_in) > w_in,
      "RandomCropAndResizeOp: multiplication out of bounds, check image width and image height first.");
    CHECK_FAIL_RETURN_UNEXPECTED(
      (std::numeric_limits<int32_t>::max() / h_in / w_in) > sample_scale,
      "RandomCropAndResizeOp: multiplication out of bounds, check image width, image height and sample scale first.");
    CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int32_t>::max() / h_in / w_in / sample_scale) > sample_aspect,
                                 "RandomCropAndResizeOp: multiplication out of bounds, check image width, image "
                                 "height, sample scale and sample aspect first.");
    *crop_width = static_cast<int32_t>(std::round(std::sqrt(h_in * w_in * sample_scale * sample_aspect)));
    *crop_height = static_cast<int32_t>(std::round(*crop_width / sample_aspect));

    // forbidden crop_width or crop_height is zero
    if (*crop_width <= 0) {
      *crop_width = 1;
    }
    if (*crop_height <= 0) {
      *crop_height = 1;
    }

    if (*crop_width <= w_in && *crop_height <= h_in) {
      std::uniform_int_distribution<> rd_x(0, w_in - *crop_width);
      std::uniform_int_distribution<> rd_y(0, h_in - *crop_height);
      *x = rd_x(rnd_);
      *y = rd_y(rnd_);
      return Status::OK();
    }
  }
  double const img_aspect = static_cast<double>(w_in) / h_in;
  if (img_aspect < aspect_lb_) {
    *crop_width = w_in;
    *crop_height = static_cast<int32_t>(std::round(*crop_width / static_cast<double>(aspect_lb_)));
  } else {
    if (img_aspect > aspect_ub_) {
      *crop_height = h_in;
      *crop_width = static_cast<int32_t>(std::round(*crop_height * static_cast<double>(aspect_ub_)));
    } else {
      *crop_width = w_in;
      *crop_height = h_in;
    }
  }
  constexpr float crop_ratio = 2.0;
  // forbidden crop_width or crop_height is zero
  if (*crop_width <= 0) {
    *crop_width = 1;
  }
  if (*crop_height <= 0) {
    *crop_height = 1;
  }

  *x = static_cast<int32_t>(std::round((w_in - *crop_width) / crop_ratio));
  *y = static_cast<int32_t>(std::round((h_in - *crop_height) / crop_ratio));
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
