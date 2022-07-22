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
#include "minddata/dataset/kernels/image/pad_to_size_op.h"

#include "minddata/dataset/kernels/image/image_utils.h"

namespace mindspore {
namespace dataset {
PadToSizeOp::PadToSizeOp(const std::vector<int32_t> &size, const std::vector<int32_t> &offset,
                         const std::vector<uint8_t> &fill_value, BorderType padding_mode)
    : size_(size), offset_(offset), fill_value_(fill_value), boarder_type_(padding_mode) {}

template <typename T>
std::string SizeToString(const std::vector<T> &size) {
  std::string init;
  std::string err_msg = std::accumulate(size.begin(), size.end(), init, [](const std::string &str, T val) {
    if (str.empty()) {
      return std::to_string(val);
    } else {
      return str + ", " + std::to_string(val);
    }
  });
  return "(" + err_msg + ")";
}

Status PadToSizeOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  RETURN_IF_NOT_OK(ValidateImage(input, "PadToSize", {1, 2, 3, 4, 5, 6, 10, 11, 12}, {2, 3}, {1, 3}));
  std::vector<dsize_t> image_size;
  RETURN_IF_NOT_OK(ImageSize(input, &image_size));
  CHECK_FAIL_RETURN_SYNTAX_ERROR(
    image_size[0] <= size_[0] && image_size[1] <= size_[1],
    "PadToSize: the target size to pad should be no less than the original image size, but got target size " +
      SizeToString(size_) + " and original size " + SizeToString(image_size) + ".");
  int32_t pad_top, pad_bottom, pad_left, pad_right;
  if (offset_.empty()) {
    pad_top = static_cast<double>((size_[0] - image_size[0])) * kHalf;
    pad_left = static_cast<double>((size_[1] - image_size[1])) * kHalf;
  } else if (offset_.size() == 1) {
    pad_top = offset_[0];
    pad_left = offset_[0];
  } else {
    pad_top = offset_[0];
    pad_left = offset_[1];
  }
  pad_bottom = size_[0] - image_size[0] - pad_top;
  pad_right = size_[1] - image_size[1] - pad_left;
  CHECK_FAIL_RETURN_SYNTAX_ERROR(pad_bottom >= 0 && pad_right >= 0,
                                 "PadToSize: the sum of offset and original image size should be no more than the "
                                 "target size to pad, but got offset " +
                                   SizeToString(std::vector<int32_t>{pad_top, pad_left}) + " plus original size " +
                                   SizeToString(image_size) + " bigger than " + SizeToString(size_));
  return Pad(input, output, pad_top, pad_bottom, pad_left, pad_right, boarder_type_, fill_value_[kRIndex],
             fill_value_[kGIndex], fill_value_[kBIndex]);
}

Status PadToSizeOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  TensorShape out({size_[0], size_[1]});
  if (inputs[0].Rank() == kMinImageRank) {
    outputs = {out};
  } else if (inputs[0].Rank() == kDefaultImageRank) {
    outputs = {out.AppendDim(kDefaultImageChannel)};
  } else {
    RETURN_STATUS_UNEXPECTED("PadToSize: input tensor should be in shape of <H,W,C> or <H, W>, but got dimension: " +
                             std::to_string(inputs[0].Rank()));
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
