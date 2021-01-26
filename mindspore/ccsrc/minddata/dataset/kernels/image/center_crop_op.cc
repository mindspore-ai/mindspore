/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/image/center_crop_op.h"
#include <string>
#include "utils/ms_utils.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/image_utils.h"
#else
#include "minddata/dataset/kernels/image/lite_image_utils.h"
#endif

#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
const int32_t CenterCropOp::kDefWidth = 0;

Status CenterCropOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  std::string err_msg;
  std::string err_head = "CenterCrop: ";
  dsize_t rank = input->shape().Rank();
  err_msg += (rank < 2 || rank > 3) ? "image shape is not <H,W,C> or <H,W> \t" : "";
  err_msg += (crop_het_ <= 0 || crop_wid_ <= 0) ? "crop size needs to be positive integers\t" : "";

  if (err_msg.length() != 0) RETURN_STATUS_UNEXPECTED(err_head + err_msg);

  int32_t top = crop_het_ - input->shape()[0];  // number of pixels to pad (top and bottom)
  int32_t left = crop_wid_ - input->shape()[1];
  std::shared_ptr<Tensor> pad_image;

  CHECK_FAIL_RETURN_UNEXPECTED((top < input->shape()[0] * 3 && left < input->shape()[1] * 3),
                               "CenterCrop: CenterCropOp padding size is more than 3 times the original size.");

  if (top > 0 && left > 0) {  // padding only
    return Pad(input, output, top / 2 + top % 2, top / 2, left / 2 + left % 2, left / 2, BorderType::kConstant);
  } else if (top > 0) {
    RETURN_IF_NOT_OK(Pad(input, &pad_image, top / 2 + top % 2, top / 2, 0, 0, BorderType::kConstant));
    return Crop(pad_image, output, (static_cast<int32_t>(pad_image->shape()[1]) - crop_wid_) / 2,
                (static_cast<int32_t>(pad_image->shape()[0]) - crop_het_) / 2, crop_wid_, crop_het_);
  } else if (left > 0) {
    RETURN_IF_NOT_OK(Pad(input, &pad_image, 0, 0, left / 2 + left % 2, left / 2, BorderType::kConstant));
    return Crop(pad_image, output, (static_cast<int32_t>(pad_image->shape()[1]) - crop_wid_) / 2,
                (static_cast<int32_t>(pad_image->shape()[0]) - crop_het_) / 2, crop_wid_, crop_het_);
  }
  return Crop(input, output, (input->shape()[1] - crop_wid_) / 2, (input->shape()[0] - crop_het_) / 2, crop_wid_,
              crop_het_);
}

void CenterCropOp::Print(std::ostream &out) const {
  out << "CenterCropOp: "
      << "cropWidth: " << crop_wid_ << "cropHeight: " << crop_het_ << "\n";
}
Status CenterCropOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  TensorShape out = TensorShape{crop_het_, crop_wid_};
  if (inputs[0].Rank() == 2) outputs.emplace_back(out);
  if (inputs[0].Rank() == 3) outputs.emplace_back(out.AppendDim(inputs[0][2]));
  if (!outputs.empty()) return Status::OK();
  return Status(StatusCode::kMDUnexpectedError, "CenterCrop: invalid input shape.");
}
}  // namespace dataset
}  // namespace mindspore
