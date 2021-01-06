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
#include "minddata/dataset/kernels/image/invert_op.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

// only supports RGB images

Status InvertOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);

  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    cv::Mat input_img = input_cv->mat();
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("Invert: load image failed.");
    }

    if (input_cv->Rank() != 3) {
      RETURN_STATUS_UNEXPECTED("Invert: image shape is not <H,W,C>");
    }
    int num_channels = input_cv->shape()[2];
    if (num_channels != 3) {
      RETURN_STATUS_UNEXPECTED("Invert: image shape is incorrect: num of channels != 3");
    }
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateEmpty(input_cv->shape(), input_cv->type(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);

    output_cv->mat() = cv::Scalar::all(255) - input_img;
    *output = std::static_pointer_cast<Tensor>(output_cv);
  }

  catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Invert: " + std::string(e.what()));
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
