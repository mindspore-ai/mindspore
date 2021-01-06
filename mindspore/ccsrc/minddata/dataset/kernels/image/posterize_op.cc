/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/kernels/image/posterize_op.h"

#include <opencv2/imgcodecs.hpp>

namespace mindspore {
namespace dataset {

const uint8_t PosterizeOp::kBit = 8;

PosterizeOp::PosterizeOp(uint8_t bit) : bit_(bit) {}

Status PosterizeOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  uint8_t mask_value = ~((uint8_t)(1 << (8 - bit_)) - 1);
  std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
  if (!input_cv->mat().data) {
    RETURN_STATUS_UNEXPECTED("Posterize: load image failed.");
  }
  if (input_cv->Rank() != 3 && input_cv->Rank() != 2) {
    RETURN_STATUS_UNEXPECTED("Posterize: input image is not in shape of <H,W,C> or <H,W>");
  }
  std::vector<uint8_t> lut_vector;
  for (std::size_t i = 0; i < 256; i++) {
    lut_vector.push_back(i & mask_value);
  }
  cv::Mat in_image = input_cv->mat();
  cv::Mat output_img;
  CHECK_FAIL_RETURN_UNEXPECTED(in_image.depth() == CV_8U || in_image.depth() == CV_8S,
                               "Posterize: input image data type can not be float, "
                               "but got " +
                                 input->type().ToString());
  cv::LUT(in_image, lut_vector, output_img);
  std::shared_ptr<CVTensor> result_tensor;
  RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, &result_tensor));
  *output = std::static_pointer_cast<Tensor>(result_tensor);
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
