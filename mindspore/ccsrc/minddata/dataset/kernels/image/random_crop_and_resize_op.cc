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
#include "minddata/dataset/kernels/image/random_crop_and_resize_op.h"

#include <limits>
#include <random>
#include <vector>

#include "minddata/dataset/kernels/data/data_utils.h"
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

  for (size_t i = 0; i < input.size(); i++) {
    if (input[i]->Rank() < kMinImageRank) {
      RETURN_STATUS_UNEXPECTED("RandomResizedCrop: input tensor should have at least 2 dimensions, but got: " +
                               std::to_string(input[i]->Rank()));
    }
    if (i < input.size() - 1) {
      std::vector<dsize_t> size;
      std::vector<dsize_t> next_size;
      RETURN_IF_NOT_OK(ImageSize(input[i], &size));
      RETURN_IF_NOT_OK(ImageSize(input[i + 1], &next_size));
      if (size[0] != next_size[0] || size[1] != next_size[1]) {
        RETURN_STATUS_UNEXPECTED(
          "RandomCropAndResizeOp: Input tensor in different columns of each row must have the same size.");
      }
    }
  }
  output->resize(input.size());
  int x = 0;
  int y = 0;
  int crop_height = 0;
  int crop_width = 0;
  for (size_t i = 0; i < input.size(); i++) {
    auto input_shape = input[i]->shape();
    std::vector<dsize_t> size;
    RETURN_IF_NOT_OK(ImageSize(input[i], &size));
    int h_in = size[0];
    int w_in = size[1];
    if (i == 0) {
      RETURN_IF_NOT_OK(GetCropBox(h_in, w_in, &x, &y, &crop_height, &crop_width));
    }
    if (input[i]->Rank() <= kDefaultImageRank) {
      RETURN_IF_NOT_OK(CropAndResize(input[i], &(*output)[i], x, y, crop_height, crop_width, target_height_,
                                     target_width_, interpolation_));
    } else if (input[i]->Rank() > kDefaultImageRank) {
      dsize_t num_batch = input[i]->Size() / (input_shape[-3] * input_shape[-2] * input_shape[-1]);
      TensorShape new_shape({num_batch, input_shape[-3], input_shape[-2], input_shape[-1]});
      RETURN_IF_NOT_OK(input[i]->Reshape(new_shape));
      // split [N, H, W, C] to N [H, W, C], and Resize N [H, W, C]
      std::vector<std::shared_ptr<Tensor>> input_vector_hwc, output_vector_hwc;
      RETURN_IF_NOT_OK(BatchTensorToTensorVector(input[i], &input_vector_hwc));
      for (auto input_hwc : input_vector_hwc) {
        std::shared_ptr<Tensor> output_img;
        RETURN_IF_NOT_OK(CropAndResize(input_hwc, &output_img, x, y, crop_height, crop_width, target_height_,
                                       target_width_, interpolation_));
        output_vector_hwc.push_back(output_img);
      }
      RETURN_IF_NOT_OK(TensorVectorToBatchTensor(output_vector_hwc, &(*output)[i]));
      auto output_shape = ComputeOutputShape(input_shape, target_height_, target_width_);
      RETURN_IF_NOT_OK((*output)[i]->Reshape(output_shape));
    }
  }
  return Status::OK();
}

TensorShape RandomCropAndResizeOp::ComputeOutputShape(const TensorShape &input, int32_t target_height,
                                                      int32_t target_width) {
  auto out_shape_vec = input.AsVector();
  auto size = out_shape_vec.size();
  int32_t kHeightIdx = -3;
  int32_t kWidthIdx = -2;
  out_shape_vec[size + kHeightIdx] = target_height_;
  out_shape_vec[size + kWidthIdx] = target_width_;
  TensorShape out = TensorShape(out_shape_vec);
  return out;
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
  } else if (inputs[0].Rank() > kDefaultImageRank) {
    auto out_shape = ComputeOutputShape(inputs[0], target_height_, target_width_);
    (void)outputs.emplace_back(out_shape);
  }
  if (!outputs.empty()) {
    return Status::OK();
  }
  return Status(StatusCode::kMDUnexpectedError,
                "RandomCropAndResize: input tensor should have at least 2 dimensions, "
                "but got: " +
                  std::to_string(inputs[0].Rank()));
}
Status RandomCropAndResizeOp::GetCropBox(int h_in, int w_in, int *x, int *y, int *crop_height, int *crop_width) {
  CHECK_FAIL_RETURN_UNEXPECTED(crop_height != nullptr, "crop_height is nullptr.");
  CHECK_FAIL_RETURN_UNEXPECTED(crop_width != nullptr, "crop_width is nullptr.");
  *crop_width = w_in;
  *crop_height = h_in;
  CHECK_FAIL_RETURN_UNEXPECTED(w_in != 0, "RandomCropAndResize: Width of input cannot be 0.");
  CHECK_FAIL_RETURN_UNEXPECTED(h_in != 0, "RandomCropAndResize: Height of input cannot be 0.");
  CHECK_FAIL_RETURN_UNEXPECTED(
    aspect_lb_ > 0,
    "RandomCropAndResize: 'ratio'(aspect) lower bound must be greater than 0, but got:" + std::to_string(aspect_lb_));
  for (int32_t i = 0; i < max_iter_; i++) {
    double const sample_scale = rnd_scale_(rnd_);
    // In case of non-symmetrical aspect ratios, use uniform distribution on a logarithmic sample_scale.
    // Note rnd_aspect_ is already a random distribution of the input aspect ratio in logarithmic sample_scale.
    double const sample_aspect = exp(rnd_aspect_(rnd_));

    CHECK_FAIL_RETURN_UNEXPECTED(
      (std::numeric_limits<int32_t>::max() / h_in) > w_in,
      "RandomCropAndResizeOp: multiplication out of bounds, check image width and image height first.");
    CHECK_FAIL_RETURN_UNEXPECTED(
      static_cast<double>((std::numeric_limits<int32_t>::max() / h_in) / w_in) > sample_scale,
      "RandomCropAndResizeOp: multiplication out of bounds, check image width, image height and sample scale first.");
    CHECK_FAIL_RETURN_UNEXPECTED(
      (static_cast<double>((std::numeric_limits<int32_t>::max() / h_in) / w_in) / sample_scale) > sample_aspect,
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
