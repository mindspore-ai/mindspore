/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/image/random_crop_op.h"

#include <random>

#include "minddata/dataset/kernels/data/data_utils.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
const int32_t RandomCropOp::kDefPadTop = 0;
const int32_t RandomCropOp::kDefPadBottom = 0;
const int32_t RandomCropOp::kDefPadLeft = 0;
const int32_t RandomCropOp::kDefPadRight = 0;
const BorderType RandomCropOp::kDefBorderType = BorderType::kConstant;
const bool RandomCropOp::kDefPadIfNeeded = false;
const uint8_t RandomCropOp::kDefFillR = 0;
const uint8_t RandomCropOp::kDefFillG = 0;
const uint8_t RandomCropOp::kDefFillB = 0;

RandomCropOp::RandomCropOp(int32_t crop_height, int32_t crop_width, int32_t pad_top, int32_t pad_bottom,
                           int32_t pad_left, int32_t pad_right, bool pad_if_needed, BorderType padding_mode,
                           uint8_t fill_r, uint8_t fill_g, uint8_t fill_b)
    : crop_height_(crop_height),
      crop_width_(crop_width),
      pad_top_(pad_top),
      pad_bottom_(pad_bottom),
      pad_left_(pad_left),
      pad_right_(pad_right),
      pad_if_needed_(pad_if_needed),
      border_type_(padding_mode),
      fill_r_(fill_r),
      fill_g_(fill_g),
      fill_b_(fill_b) {
  rnd_.seed(GetSeed());
  is_deterministic_ = false;
}

Status RandomCropOp::ImagePadding(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *pad_image,
                                  int32_t *t_pad_top, int32_t *t_pad_bottom, int32_t *t_pad_left, int32_t *t_pad_right,
                                  int32_t *padded_image_w, int32_t *padded_image_h, bool *crop_further) {
  *t_pad_top = pad_top_;
  *t_pad_bottom = pad_bottom_;
  *t_pad_left = pad_left_;
  *t_pad_right = pad_right_;

  constexpr int64_t max_ratio = 3;
  CHECK_FAIL_RETURN_UNEXPECTED(
    pad_top_ < input->shape()[0] * max_ratio && pad_bottom_ < input->shape()[0] * max_ratio &&
      pad_left_ < input->shape()[1] * max_ratio && pad_right_ < input->shape()[1] * max_ratio,
    "RandomCrop: padding size is three times bigger than the image size, padding top: " + std::to_string(pad_top_) +
      ", padding bottom: " + std::to_string(pad_bottom_) + ", padding pad_left_: " + std::to_string(pad_left_) +
      ", padding padding right:" + std::to_string(pad_right_) + ", image shape: " + std::to_string(input->shape()[0]) +
      ", " + std::to_string(input->shape()[1]));

  RETURN_IF_NOT_OK(
    Pad(input, pad_image, pad_top_, pad_bottom_, pad_left_, pad_right_, border_type_, fill_r_, fill_g_, fill_b_));
  CHECK_FAIL_RETURN_UNEXPECTED(
    (*pad_image)->shape().Size() >= 2,
    "RandomCrop: invalid shape of image after pad, got rank: " + std::to_string((*pad_image)->shape().Size()));

  *padded_image_h = static_cast<int32_t>((*pad_image)->shape()[0]);
  *padded_image_w = static_cast<int32_t>((*pad_image)->shape()[1]);

  if (*padded_image_h == crop_height_ && *padded_image_w == crop_width_) {
    *crop_further = false;  //  no need for further crop
    return Status::OK();
  } else if (pad_if_needed_) {
    // check the dimensions of the image for padding, if we do need padding, then we change the pad values
    if (*padded_image_h < crop_height_) {
      RETURN_IF_NOT_OK(Pad(*pad_image, pad_image, crop_height_ - *padded_image_h, crop_height_ - *padded_image_h, 0, 0,
                           border_type_, fill_r_, fill_g_, fill_b_));

      // update pad total above/below
      t_pad_top += ((ptrdiff_t)crop_height_ - *padded_image_h);
      t_pad_bottom += ((ptrdiff_t)crop_height_ - *padded_image_h);
    }
    if (*padded_image_w < crop_width_) {
      RETURN_IF_NOT_OK(Pad(*pad_image, pad_image, 0, 0, crop_width_ - *padded_image_w, crop_width_ - *padded_image_w,
                           border_type_, fill_r_, fill_g_, fill_b_));
      // update pad total left/right
      t_pad_left += ((ptrdiff_t)crop_width_ - *padded_image_w);
      t_pad_right += ((ptrdiff_t)crop_width_ - *padded_image_w);
    }
    *padded_image_h = static_cast<int32_t>((*pad_image)->shape()[0]);
    *padded_image_w = static_cast<int32_t>((*pad_image)->shape()[1]);
  }

  if (crop_height_ == 0 || crop_width_ == 0) {
    RETURN_STATUS_ERROR(StatusCode::kMDShapeMisMatch,
                        "RandomCrop: invalid crop size, crop width or crop height is not allowed to be zero.");
  }
  if (*padded_image_h < crop_height_ || *padded_image_w < crop_width_ || crop_height_ == 0 || crop_width_ == 0) {
    RETURN_STATUS_ERROR(StatusCode::kMDShapeMisMatch,
                        "RandomCrop: invalid crop size, crop size is bigger than the image dimensions, "
                        "got crop height: " +
                          std::to_string(crop_height_) + ", crop width: " + std::to_string(crop_width_));
  }
  return Status::OK();
}

void RandomCropOp::GenRandomXY(int32_t *x, int32_t *y, int32_t padded_image_w, int32_t padded_image_h) {
  // GenCropPoints for cropping
  *x = std::uniform_int_distribution<int>(0, padded_image_w - crop_width_)(rnd_);
  *y = std::uniform_int_distribution<int>(0, padded_image_h - crop_height_)(rnd_);
}

Status RandomCropOp::RandomCropImg(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t *x,
                                   int32_t *y, int32_t index) {
  std::shared_ptr<Tensor> pad_image = nullptr;
  int32_t t_pad_top = 0;
  int32_t t_pad_bottom = 0;
  int32_t t_pad_left = 0;
  int32_t t_pad_right = 0;
  int32_t padded_image_w = 0;
  int32_t padded_image_h = 0;
  bool crop_further = true;  // whether image needs further cropping based on new size & requirements

  RETURN_IF_NOT_OK(  // error code sent back directly
    ImagePadding(input, &pad_image, &t_pad_top, &t_pad_bottom, &t_pad_left, &t_pad_right, &padded_image_w,
                 &padded_image_h, &crop_further));
  if (!crop_further) {
    *output = pad_image;
    return Status::OK();
  }
  if (index == 0) {
    GenRandomXY(x, y, padded_image_w, padded_image_h);
  }
  RETURN_IF_NOT_OK(Crop(pad_image, output, *x, *y, crop_width_, crop_height_));

  return Status::OK();
}

Status RandomCropOp::ConstructShape(const TensorShape &in_shape, std::shared_ptr<TensorShape> *out_shape) const {
  auto in_shape_vec = in_shape.AsVector();
  const int h_index = -3;
  const int w_index = -2;
  in_shape_vec[in_shape_vec.size() + h_index] = crop_height_;
  in_shape_vec[in_shape_vec.size() + w_index] = crop_width_;

  *out_shape = std::make_shared<TensorShape>(in_shape_vec);

  return Status::OK();
}

Status RandomCropOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);

  for (const auto &image : input) {
    if (image->shape().Rank() < kMinImageRank) {
      std::string err_msg =
        "RandomCropOp: input tensor should have at least 2 dimensions, but got: " + std::to_string(image->Rank());
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
  }

  if (input.size() > 1) {
    for (size_t i = 0; i < input.size() - 1; i++) {
      if (input[i]->shape()[0] != input[i + 1]->shape()[0] || input[i]->shape()[1] != input[i + 1]->shape()[1]) {
        RETURN_STATUS_UNEXPECTED(
          "RandomCrop: Input images in different column must have the same shape, check the output shape in "
          "specified 'input_columns' before call this operation.");
      }
    }
  }

  const auto output_count = input.size();
  output->resize(output_count);
  int32_t x = 0;
  int32_t y = 0;
  for (size_t i = 0; i < input.size(); i++) {
    if (input[i]->shape().Rank() <= kDefaultImageRank) {  // keep original logic untained
      RETURN_IF_NOT_OK(RandomCropImg(input[i], &(*output)[i], &x, &y, i));
    } else {  // deal with videos
      // reshape input to hwc
      auto input_shape = input[i]->shape();
      dsize_t num_batch = input[i]->Size() / (input_shape[-3] * input_shape[-2] * input_shape[-1]);
      TensorShape new_shape({num_batch, input_shape[-3], input_shape[-2], input_shape[-1]});
      RETURN_IF_NOT_OK(input[i]->Reshape(new_shape));

      // split [N, H, W, C] to N [H, W, C], and center crop N [H, W, C]
      std::vector<std::shared_ptr<Tensor>> input_vector_hwc, output_vector_hwc;
      RETURN_IF_NOT_OK(BatchTensorToTensorVector(input[i], &input_vector_hwc));

      // perform randomCrop
      for (int32_t idx = 0; idx < num_batch; idx++) {
        std::shared_ptr<Tensor> random_crop;
        RETURN_IF_NOT_OK(RandomCropImg(input_vector_hwc[idx], &random_crop, &x, &y, i));
        output_vector_hwc.push_back(random_crop);
      }

      // integrate N [H, W, C] to [N, H, W, C], and reshape [..., H, W, C]
      RETURN_IF_NOT_OK(TensorVectorToBatchTensor(output_vector_hwc, &(*output)[i]));

      // reshape output before return, only height and width are changed
      std::shared_ptr<TensorShape> output_shape_new;
      RETURN_IF_NOT_OK(ConstructShape(input_shape, &output_shape_new));
      RETURN_IF_NOT_OK((*output)[i]->Reshape(*output_shape_new));
    }
  }

  return Status::OK();
}

Status RandomCropOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  TensorShape out = TensorShape{crop_height_, crop_width_};
  if (inputs[0].Rank() == kMinImageRank) {
    (void)outputs.emplace_back(out);
  } else if (inputs[0].Rank() == kDefaultImageRank) {
    (void)outputs.emplace_back(out.AppendDim(inputs[0][kChannelIndexHWC]));
  } else if (inputs[0].Rank() > kDefaultImageRank) {
    std::shared_ptr<TensorShape> output_shape_new;
    RETURN_IF_NOT_OK(ConstructShape(inputs[0], &output_shape_new));
    (void)outputs.emplace_back(*output_shape_new);
  }
  if (!outputs.empty()) {
    return Status::OK();
  }
  RETURN_STATUS_UNEXPECTED("RandomCrop: invalid input shape, expected 2D or 3D input, but got input dimension is:" +
                           std::to_string(inputs[0].Rank()));
}
}  // namespace dataset
}  // namespace mindspore
