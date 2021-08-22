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

#include "common/common.h"
#include "common/cvop_common.h"
#include "minddata/dataset/kernels/image/random_color_op.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;

class MindDataTestRandomColorOp : public UT::CVOP::CVOpCommon {
 public:
  MindDataTestRandomColorOp() : CVOpCommon(), shape({3, 3, 3}) {
    std::shared_ptr<Tensor> in;
    std::shared_ptr<Tensor> gray;

    (void)Tensor::CreateEmpty(shape, DataType(DataType::DE_UINT8), &in);
    (void)Tensor::CreateEmpty(shape, DataType(DataType::DE_UINT8), &input_tensor);
    Status s = in->Fill<uint8_t>(42);
    s = input_tensor->Fill<uint8_t>(42);
    cvt_in = CVTensor::AsCVTensor(in);
    cv::Mat m2;
    auto m1 = cvt_in->mat();
    cv::cvtColor(m1, m2, cv::COLOR_RGB2GRAY);
    cv::Mat temp[3] = {m2 , m2 , m2 };
    cv::Mat cv_out;
    cv::merge(temp, 3, cv_out);
    std::shared_ptr<CVTensor> cvt_out;
    CVTensor::CreateFromMat(cv_out, 3, &cvt_out);
    gray_tensor = std::static_pointer_cast<Tensor>(cvt_out);
  }
  TensorShape shape;
  std::shared_ptr<Tensor> input_tensor;
  std::shared_ptr<CVTensor> cvt_in;
  std::shared_ptr<Tensor> gray_tensor;
};

int64_t Compare(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2) {
  auto shape = t1->shape();
  int64_t sum = 0;
  for (auto i = 0; i < shape[0]; i++) {
    for (auto j = 0; j < shape[1]; j++) {
      for (auto k = 0; k < shape[2]; k++) {
        uint8_t value1;
        uint8_t value2;
        (void)t1->GetItemAt<uint8_t>(&value1, {i, j, k});
        (void)t2->GetItemAt<uint8_t>(&value2, {i, j, k});
        sum += abs(static_cast<int>(value1) - static_cast<int>(value2));
      }
    }
  }
  return sum;
}

// these tests are tautological, write better tests when the requirements for the output are determined
// e. g. how do we want to convert to gray and what does it mean to blend with a gray image (pre- post- gamma corrected,
// what weights).
TEST_F(MindDataTestRandomColorOp, TestOp1) {
  std::shared_ptr<Tensor> output_tensor;
  auto op = RandomColorOp(1, 1);
  auto s = op.Compute(input_tensor, &output_tensor);
  auto res = Compare(input_tensor, output_tensor);
  EXPECT_EQ(0, res);
}

TEST_F(MindDataTestRandomColorOp, TestOp2) {
  std::shared_ptr<Tensor> output_tensor;
  auto op = RandomColorOp(0, 0);
  auto s = op.Compute(input_tensor, &output_tensor);
  EXPECT_TRUE(s.IsOk());
  auto res = Compare(output_tensor, gray_tensor);
  EXPECT_EQ(res, 0);
}

TEST_F(MindDataTestRandomColorOp, TestOp3) {
  std::shared_ptr<Tensor> output_tensor;
  auto op = RandomColorOp(0.0, 1.0);
  for (auto i = 0; i < 1; i++) {
    auto s = op.Compute(input_tensor, &output_tensor);
    EXPECT_TRUE(s.IsOk());
  }
}
