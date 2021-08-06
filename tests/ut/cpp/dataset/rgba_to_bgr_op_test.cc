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

#include <opencv2/imgcodecs.hpp>
#include "common/common.h"
#include "common/cvop_common.h"
#include "minddata/dataset/kernels/image/rgba_to_bgr_op.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

class MindDataTestRgbaToBgrOp : public UT::CVOP::CVOpCommon {
 protected:
  MindDataTestRgbaToBgrOp() : CVOpCommon() {}

  std::shared_ptr<Tensor> output_tensor_;
};

TEST_F(MindDataTestRgbaToBgrOp, TestOp1) {
  MS_LOG(INFO) << "Doing testRGBA2BGR.";
  std::unique_ptr<RgbaToBgrOp> op(new RgbaToBgrOp());
  EXPECT_TRUE(op->OneToOne());
  // prepare 4 channel image
  cv::Mat rgba_image;
  // First create the image with alpha channel
  cv::cvtColor(raw_cv_image_, rgba_image, cv::COLOR_BGR2RGBA);
  std::vector<cv::Mat>channels(4);
  cv::split(rgba_image, channels);
  channels[3] = cv::Mat::zeros(rgba_image.rows, rgba_image.cols, CV_8UC1);
  cv::merge(channels, rgba_image);
  // create new tensor to test conversion
  std::shared_ptr<Tensor> rgba_input;
  std::shared_ptr<CVTensor> input_cv_tensor;
  CVTensor::CreateFromMat(rgba_image, 3, &input_cv_tensor);
  rgba_input = std::dynamic_pointer_cast<Tensor>(input_cv_tensor);

  Status s = op->Compute(rgba_input, &output_tensor_);
  size_t actual = 0;
  if (s == Status::OK()) {
    actual = output_tensor_->shape()[0] * output_tensor_->shape()[1] * output_tensor_->shape()[2];
  }
  EXPECT_EQ(actual, input_tensor_->shape()[0] * input_tensor_->shape()[1] * 3);
  EXPECT_EQ(s, Status::OK());
}

