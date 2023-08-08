/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/image/resized_crop_op.h"

#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
Status ResizedCropOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  // input output tensor shape check
  IO_CHECK(input, output);
  RETURN_IF_NOT_OK(ValidateImageRank("ResizedCropOp", input->Rank()));

  std::vector<dsize_t> input_size;
  RETURN_IF_NOT_OK(ImageSize(input, &input_size));
  auto input_h = static_cast<int32_t>(input_size[kHeightIndex]);
  auto input_w = static_cast<int32_t>(input_size[kWidthIndex]);
  int32_t size1 = size_[kHeightIndex];
  int32_t size2 = size_[kWidthIndex];

  // crop check
  CHECK_FAIL_RETURN_UNEXPECTED(top_ + height_ <= input_h,
                               "ResizedCrop: the sum of top and height: " + std::to_string(top_ + height_) +
                                 " exceeds image height: " + std::to_string(input_h));
  CHECK_FAIL_RETURN_UNEXPECTED(left_ + width_ <= input_w,
                               "ResizedCrop: the sum of left and width: " + std::to_string(left_ + width_) +
                                 " exceeds image width: " + std::to_string(input_w));

  int32_t output_h;
  int32_t output_w;
  if (size2 == 0) {
    if (input_h < input_w) {
      output_h = size1;
      output_w = static_cast<int>(
        roundf(static_cast<float>(width_) / static_cast<float>(height_) * static_cast<float>(output_h)));
    } else {
      output_w = size1;
      output_h = static_cast<int>(
        roundf(static_cast<float>(height_) / static_cast<float>(width_) * static_cast<float>(output_w)));
    }
  } else {
    output_h = size1;
    output_w = size2;
  }

  RETURN_IF_NOT_OK(CropAndResize(input, output, left_, top_, height_, width_, output_h, output_w, interpolation_));

  return Status::OK();
}

Status ResizedCropOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();

  int32_t size1 = size_[kHeightIndex];
  int32_t size2 = size_[kWidthIndex];
  int32_t output_h = 0;
  int32_t output_w = 0;

  if (size2 != 0) {
    output_h = size1;
    output_w = size2;
  }

  TensorShape out = TensorShape{output_h, output_w};
  const TensorShape &input_shape = inputs.front();
  if (input_shape.Rank() == kMinImageRank) {
    (void)outputs.emplace_back(out);
  }
  if (input_shape.Rank() == kDefaultImageRank) {
    (void)outputs.emplace_back(out.AppendDim(input_shape[input_shape.Size() - 1]));
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!outputs.empty(),
                               "ResizedCrop: input tensor is not in shape of <H,W> or <H,W,C>, but got rank: " +
                                 std::to_string(input_shape.Rank()));
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
