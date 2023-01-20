/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/image/decode_op.h"
#include "minddata/dataset/kernels/image/random_crop_and_resize_op.h"
#include "minddata/dataset/kernels/image/random_crop_decode_resize_op.h"
#include "minddata/dataset/core/config_manager.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
constexpr double kMseThreshold = 2.5;

class MindDataTestRandomCropDecodeResizeOp : public UT::CVOP::CVOpCommon {
 public:
  MindDataTestRandomCropDecodeResizeOp() : CVOpCommon() {}
};

/// Feature: RandomCropDecodeResize op
/// Description: Test RandomCropDecodeResizeOp basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestRandomCropDecodeResizeOp, TestOp2) {
  MS_LOG(INFO) << "starting RandomCropDecodeResizeOp test 1";

  std::shared_ptr<Tensor> crop_and_decode_output;

  constexpr int target_height = 884;
  constexpr int target_width = 718;
  constexpr float scale_lb = 0.08;
  constexpr float scale_ub = 1.0;
  constexpr float aspect_lb = 0.75;
  constexpr float aspect_ub = 1.333333;
  const InterpolationMode interpolation = InterpolationMode::kLinear;
  constexpr uint32_t max_iter = 10;

  auto crop_and_decode = RandomCropDecodeResizeOp(target_height, target_width, scale_lb, scale_ub, aspect_lb, aspect_ub,
                                                  interpolation, max_iter);
  auto crop_and_decode_copy = crop_and_decode;
  auto decode_and_crop = static_cast<RandomCropAndResizeOp>(crop_and_decode_copy);
  GlobalContext::config_manager()->set_seed(42);
  for (int k = 0; k < 10; k++) {
    TensorRow input_tensor_row_decode;
    TensorRow output_tensor_row_decode;
    std::shared_ptr<Tensor> input1;
    Tensor::CreateFromTensor(raw_input_tensor_, &input1);
    input_tensor_row_decode.push_back(input1);
    TensorRow input_tensor_row;
    TensorRow output_tensor_row;
    std::shared_ptr<Tensor> input2;
    Tensor::CreateFromTensor(input_tensor_, &input2);
    input_tensor_row.push_back(input2);
    (void)crop_and_decode.Compute(input_tensor_row_decode, &output_tensor_row_decode);
    (void)decode_and_crop.Compute(input_tensor_row, &output_tensor_row);
    cv::Mat output1 = CVTensor::AsCVTensor(output_tensor_row_decode[0])->mat().clone();
    cv::Mat output2 = CVTensor::AsCVTensor(output_tensor_row[0])->mat().clone();

    long int mse_sum = 0;
    long int count = 0;
    int a, b;
    for (int i = 0; i < target_height; i++) {
      for (int j = 0; j < target_width; j++) {
        a = static_cast<int>(output1.at<cv::Vec3b>(i, j)[1]);
        b = static_cast<int>(output2.at<cv::Vec3b>(i, j)[1]);
        mse_sum += sqrt((a - b) * (a - b));
        if (a != b) {
          count++;
        };
      }
    }
    double mse;
    mse = count > 0 ? static_cast<double>(mse_sum) / count : mse_sum;
    MS_LOG(INFO) << "mse: " << mse << std::endl;
    EXPECT_LT(mse, kMseThreshold);
  }

  MS_LOG(INFO) << "RandomCropDecodeResizeOp test 1 finished";
}

/// Feature: RandomCropDecodeResize op
/// Description: Test by applying individual ops: Decode op and Crop op, and JpegCropAndDecode op
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestRandomCropDecodeResizeOp, TestOp1) {
  MS_LOG(INFO) << "starting RandomCropDecodeResizeOp test 2";
  constexpr int h = 884;
  constexpr int w = 718;
  constexpr float scale_lb = 0.1;
  constexpr float scale_ub = 1;
  constexpr float aspect_lb = 0.1;
  constexpr float aspect_ub = 10;

  std::shared_ptr<Tensor> decoded, decoded_and_cropped, cropped_and_decoded;
  std::mt19937 rd;
  std::uniform_real_distribution<float> rd_scale(scale_lb, scale_ub);
  std::uniform_real_distribution<float> rd_aspect(aspect_lb, aspect_ub);
  DecodeOp op(true);
  std::shared_ptr<Tensor> raw_input1;
  Tensor::CreateFromTensor(raw_input_tensor_, &raw_input1);
  op.Compute(raw_input1, &decoded);
  Status crop_and_decode_status, decode_and_crop_status;
  float scale, aspect;
  int crop_width, crop_height;
  bool crop_success = false;
  int mse_sum, m1, m2, count;
  double mse;

  for (int k = 0; k < 10; ++k) {
    mse_sum = 0;
    count = 0;
    for (auto i = 0; i < 10; i++) {
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

    std::shared_ptr<Tensor> raw_input2;
    Tensor::CreateFromTensor(raw_input_tensor_, &raw_input2);
    op.Compute(raw_input2, &decoded);
    crop_and_decode_status = Crop(decoded, &decoded_and_cropped, x, y, crop_width, crop_height);
    std::shared_ptr<Tensor> raw_input3;
    Tensor::CreateFromTensor(raw_input_tensor_, &raw_input3);
    decode_and_crop_status = JpegCropAndDecode(raw_input3, &cropped_and_decoded, x, y, crop_width, crop_height);
    {
      cv::Mat M1 = CVTensor::AsCVTensor(decoded_and_cropped)->mat().clone();
      cv::Mat M2 = CVTensor::AsCVTensor(cropped_and_decoded)->mat().clone();
      for (int i = 0; i < crop_height; ++i) {
        for (int j = 0; j < crop_width; ++j) {
          m1 = M1.at<cv::Vec3b>(i, j)[1];
          m2 = M2.at<cv::Vec3b>(i, j)[1];
          mse_sum += sqrt((m1 - m2) * (m1 - m2));
          if (m1 != m2) {
            count++;
          }
        }
      }
    }

    mse = count > 0 ? static_cast<double>(mse_sum) / count : mse_sum;
    MS_LOG(INFO) << "mse: " << mse << std::endl;
    EXPECT_LT(mse, kMseThreshold);
  }
  MS_LOG(INFO) << "RandomCropDecodeResizeOp test 2 finished";
}
