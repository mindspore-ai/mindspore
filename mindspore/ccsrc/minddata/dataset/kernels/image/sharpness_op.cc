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

#include "minddata/dataset/kernels/image/sharpness_op.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

const float SharpnessOp::kDefAlpha = 1.0;

Status SharpnessOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);

  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    cv::Mat input_img = input_cv->mat();
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("Sharpness: load image failed.");
    }

    if (input_cv->Rank() == 1 || input_cv->mat().dims > 2) {
      RETURN_STATUS_UNEXPECTED("Sharpness: input tensor is not in shape of <H,W,C> or <H,W>.");
    }

    /// creating a smoothing filter. 1, 1, 1,
    ///                              1, 5, 1,
    ///                              1, 1, 1

    float filterSum = 13.0;
    cv::Mat filter = cv::Mat(3, 3, CV_32F, cv::Scalar::all(1.0 / filterSum));
    filter.at<float>(1, 1) = 5.0 / filterSum;

    /// applying filter on channels
    cv::Mat result = cv::Mat();
    cv::filter2D(input_img, result, -1, filter);

    int height = input_cv->shape()[0];
    int width = input_cv->shape()[1];

    /// restoring the edges
    input_img.row(0).copyTo(result.row(0));
    input_img.row(height - 1).copyTo(result.row(height - 1));
    input_img.col(0).copyTo(result.col(0));
    input_img.col(width - 1).copyTo(result.col(width - 1));

    /// blend based on alpha : (alpha_ *input_img) +  ((1.0-alpha_) * result);
    cv::addWeighted(input_img, alpha_, result, 1.0 - alpha_, 0.0, result);

    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(result, &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);

    *output = std::static_pointer_cast<Tensor>(output_cv);
  }

  catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Sharpness: " + std::string(e.what()));
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
