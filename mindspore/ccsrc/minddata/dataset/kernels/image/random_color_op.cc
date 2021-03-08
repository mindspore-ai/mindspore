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

#include "minddata/dataset/kernels/image/random_color_op.h"
#include "minddata/dataset/core/cv_tensor.h"
namespace mindspore {
namespace dataset {

RandomColorOp::RandomColorOp(float t_lb, float t_ub) : rnd_(GetSeed()), dist_(t_lb, t_ub), t_lb_(t_lb), t_ub_(t_ub) {
  is_deterministic_ = false;
}

Status RandomColorOp::Compute(const std::shared_ptr<Tensor> &in, std::shared_ptr<Tensor> *out) {
  IO_CHECK(in, out);
  if (in->Rank() != 3 || in->shape()[2] != 3) {
    RETURN_STATUS_UNEXPECTED("RandomColor: image shape is not <H,W,C> or channel is not 3.");
  }
  // 0.5 pixel precision assuming an 8 bit image
  const auto eps = 0.00195;
  const auto t = dist_(rnd_);
  if (abs(t - 1.0) < eps) {
    // Just return input? Can we do it given that input would otherwise get consumed in CVTensor constructor anyway?
    *out = in;
    return Status::OK();
  }
  auto cvt_in = CVTensor::AsCVTensor(in);
  auto m1 = cvt_in->mat();
  cv::Mat gray;
  // gray is allocated without using the allocator
  cv::cvtColor(m1, gray, cv::COLOR_RGB2GRAY);
  // luminosity is not preserved, consider using weights.
  cv::Mat temp[3] = {gray, gray, gray};
  cv::Mat cv_out;
  cv::merge(temp, 3, cv_out);
  std::shared_ptr<CVTensor> cvt_out;
  CVTensor::CreateFromMat(cv_out, &cvt_out);
  if (abs(t - 0.0) < eps) {
    // return grayscale
    *out = std::static_pointer_cast<Tensor>(cvt_out);
    return Status::OK();
  }
  // return blended image. addWeighted takes care of overflow for uint8_t
  cv::addWeighted(m1, t, cvt_out->mat(), 1 - t, 0, cvt_out->mat());
  *out = std::static_pointer_cast<Tensor>(cvt_out);
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
