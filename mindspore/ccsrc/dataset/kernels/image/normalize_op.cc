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
#include "dataset/kernels/image/normalize_op.h"

#include <random>

#include "dataset/core/cv_tensor.h"
#include "dataset/kernels/image/image_utils.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
NormalizeOp::NormalizeOp(float mean_r, float mean_g, float mean_b, float std_r, float std_g, float std_b) {
  int size[] = {3};
  cv::Mat mean_cv(1, size, CV_32F);
  mean_cv.at<float>(0) = mean_r;
  mean_cv.at<float>(1) = mean_g;
  mean_cv.at<float>(2) = mean_b;
  mean_ = std::make_shared<CVTensor>(mean_cv);
  mean_->Squeeze();

  cv::Mat std_cv(1, size, CV_32F);
  std_cv.at<float>(0) = std_r;
  std_cv.at<float>(1) = std_g;
  std_cv.at<float>(2) = std_b;
  std_ = std::make_shared<CVTensor>(std_cv);
  std_->Squeeze();
}

Status NormalizeOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // Doing the normalization
  return Normalize(input, output, mean_, std_);
}

void NormalizeOp::Print(std::ostream &out) const {
  out << "NormalizeOp, mean: " << mean_->mat().at<float>(0) << ", " << mean_->mat().at<float>(1) << ", "
      << mean_->mat().at<float>(2) << "std: " << std_->mat().at<float>(0) << ", " << std_->mat().at<float>(1) << ", "
      << std_->mat().at<float>(2) << std::endl;
}
}  // namespace dataset
}  // namespace mindspore
