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
#include "minddata/dataset/kernels/image/solarize_op.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

// only supports RGB images
const uint8_t kPixelValue = 255;

Status SolarizeOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);

  uint8_t threshold_min_ = threshold_[0], threshold_max_ = threshold_[1];

  CHECK_FAIL_RETURN_UNEXPECTED(threshold_min_ <= threshold_max_,
                               "Solarize: threshold_min must be smaller or equal to threshold_max.");

  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    cv::Mat input_img = input_cv->mat();
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("Solarize: load image failed.");
    }

    std::shared_ptr<CVTensor> mask_mat_tensor;
    std::shared_ptr<CVTensor> output_cv_tensor;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(input_cv->mat(), &mask_mat_tensor));

    RETURN_IF_NOT_OK(CVTensor::CreateEmpty(input_cv->shape(), input_cv->type(), &output_cv_tensor));
    RETURN_UNEXPECTED_IF_NULL(mask_mat_tensor);
    RETURN_UNEXPECTED_IF_NULL(output_cv_tensor);

    if (threshold_min_ == threshold_max_) {
      mask_mat_tensor->mat().setTo(0, ~(input_cv->mat() >= threshold_min_));
    } else {
      mask_mat_tensor->mat().setTo(0, ~((input_cv->mat() >= threshold_min_) & (input_cv->mat() <= threshold_max_)));
    }

    // solarize desired portion
    output_cv_tensor->mat() = cv::Scalar::all(255) - mask_mat_tensor->mat();
    input_cv->mat().copyTo(output_cv_tensor->mat(), mask_mat_tensor->mat() == 0);
    input_cv->mat().copyTo(output_cv_tensor->mat(), input_cv->mat() < threshold_min_);

    *output = std::static_pointer_cast<Tensor>(output_cv_tensor);
  }

  catch (const cv::Exception &e) {
    const char *cv_err_msg = e.what();
    std::string err_message = "Solarize: ";
    err_message += cv_err_msg;
    RETURN_STATUS_UNEXPECTED(err_message);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
