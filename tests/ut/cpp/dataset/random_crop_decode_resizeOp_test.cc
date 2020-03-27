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
#include <fstream>
#include "common/common.h"
#include "common/cvop_common.h"
#include "dataset/kernels/image/decode_op.h"
#include "dataset/kernels/image/random_crop_and_resize_op.h"
#include "dataset/kernels/image/random_crop_decode_resize_op.h"
#include "dataset/core/config_manager.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

class MindDataTestRandomCropDecodeResizeOp : public UT::CVOP::CVOpCommon {
 public:
  MindDataTestRandomCropDecodeResizeOp() : CVOpCommon() {}
};

TEST_F(MindDataTestRandomCropDecodeResizeOp, TestOp2) {
  MS_LOG(INFO) << "Doing testRandomCropDecodeResizeOp Test";

  std::shared_ptr<Tensor> output_tensor1;
  std::shared_ptr<Tensor> output_tensor2;

  int target_height = 884;
  int target_width = 718;
  float scale_lb = 0.08;
  float scale_ub = 1.0;
  float aspect_lb = 0.75;
  float aspect_ub = 1.333333;
  InterpolationMode interpolation = InterpolationMode::kLinear;
  uint32_t max_iter = 10;
  std::unique_ptr<RandomCropAndResizeOp> op1(new RandomCropAndResizeOp(
    target_height, target_width, scale_lb, scale_ub, aspect_lb, aspect_ub, interpolation, max_iter));
  EXPECT_TRUE(op1->OneToOne());
  std::unique_ptr<RandomCropDecodeResizeOp> op2(new RandomCropDecodeResizeOp(
    target_height, target_width, scale_lb, scale_ub, aspect_lb, aspect_ub, interpolation, max_iter));
  EXPECT_TRUE(op2->OneToOne());
  Status s1, s2;

  for (int i = 0; i < 100; i++) {
    s1 = op1->Compute(input_tensor_, &output_tensor1);
    s2 = op2->Compute(raw_input_tensor_, &output_tensor2);
    cv::Mat output1(target_height, target_width, CV_8UC3, output_tensor1->StartAddr());
    cv::Mat output2(target_height, target_width, CV_8UC3, output_tensor2->StartAddr());
    long int mse_sum = 0;
    long int count = 0;
    int a, b;
    for (int i = 0; i < target_height; i++) {
      for (int j = 0; j < target_width; j++) {
        a = (int)output1.at<cv::Vec3b>(i, j)[1];
        b = (int)output2.at<cv::Vec3b>(i, j)[1];
        mse_sum += sqrt((a - b) * (a - b));
        if (a != b) {
          count++;
        };
      }
    }
    double mse;
    if (count > 0) {
      mse = (double) mse_sum / count;
    } else {
      mse = mse_sum;
    }
    std::cout << "mse: " << mse << std::endl;
  }
  MS_LOG(INFO) << "MindDataTestRandomCropDecodeResizeOp end!";
}

TEST_F(MindDataTestRandomCropDecodeResizeOp, TestOp1) {
  MS_LOG(INFO) << "Doing MindDataTestRandomCropDecodeResizeOp";
  const unsigned int h = 884;
  const unsigned int w = 718;
  const float scale_lb = 0.1;
  const float scale_ub = 1;
  const float aspect_lb = 0.1;
  const float aspect_ub = 10;

  std::shared_ptr<Tensor> decoded, decoded_and_cropped, cropped_and_decoded;
  std::mt19937 rd;
  std::uniform_real_distribution<float> rd_scale(scale_lb, scale_ub);
  std::uniform_real_distribution<float> rd_aspect(aspect_lb, aspect_ub);
  DecodeOp op(true);
  op.Compute(raw_input_tensor_, &decoded);
  Status s1, s2;
  float scale, aspect;
  int crop_width, crop_height;
  bool crop_success = false;
  unsigned int mse_sum, m1, m2, count;
  float mse;

  for (unsigned int k = 0; k < 100; ++k) {
    mse_sum = 0;
    count = 0;
    for (auto i = 0; i < 100; i++) {
      scale = rd_scale(rd);
      aspect = rd_aspect(rd);
      crop_width = std::round(std::sqrt(h * w * scale / aspect));
      crop_height = std::round(crop_width * aspect);
      if (crop_width <= w && crop_height <= h) {
        crop_success = true;
        break;
      }
    }
    if (crop_success == false) {
      aspect = static_cast<float>(h) / w;
      scale = rd_scale(rd);
      crop_width = std::round(std::sqrt(h * w * scale / aspect));
      crop_height = std::round(crop_width * aspect);
      crop_height = (crop_height > h) ? h : crop_height;
      crop_width = (crop_width > w) ? w : crop_width;
    }
    std::uniform_int_distribution<> rd_x(0, w - crop_width);
    std::uniform_int_distribution<> rd_y(0, h - crop_height);
    int x = rd_x(rd);
    int y = rd_y(rd);

    op.Compute(raw_input_tensor_, &decoded);
    s1 = Crop(decoded, &decoded_and_cropped, x, y, crop_width, crop_height);
    s2 = JpegCropAndDecode(raw_input_tensor_, &cropped_and_decoded, x, y, crop_width, crop_height);
    {
      cv::Mat M1(crop_height, crop_width, CV_8UC3, decoded_and_cropped->StartAddr());
      cv::Mat M2(crop_height, crop_width, CV_8UC3, cropped_and_decoded->StartAddr());
      for (unsigned int i = 0; i < crop_height; ++i) {
        for (unsigned int j = 0; j < crop_width; ++j) {
          m1 = M1.at<cv::Vec3b>(i, j)[1];
          m2 = M2.at<cv::Vec3b>(i, j)[1];
          mse_sum += sqrt((m1 - m2) * (m1 - m2));
          if (m1 != m2) {
            count++;
          }
        }
      }
    }

    mse = (count == 0) ? mse_sum : static_cast<float>(mse_sum) / count;
    std::cout << "mse: " << mse << std::endl;
  }
  MS_LOG(INFO) << "MindDataTestRandomCropDecodeResizeOp end!";
}
