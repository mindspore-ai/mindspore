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
#include "minddata/dataset/kernels/image/random_posterize_op.h"
#include "minddata/dataset/include/dataset/execute.h"
#include "minddata/dataset/include/dataset/vision.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;

class MindDataTestRandomPosterizeOp : public UT::CVOP::CVOpCommon {
};

TEST_F(MindDataTestRandomPosterizeOp, TestOp1) {
  MS_LOG(INFO) << "Doing testRandomPosterize.";

  std::shared_ptr<Tensor> output_tensor;
  std::unique_ptr<RandomPosterizeOp> op(new RandomPosterizeOp({1, 1}));
  EXPECT_TRUE(op->OneToOne());
  Status s = op->Compute(input_tensor_, &output_tensor);
  EXPECT_TRUE(s.IsOk());
  CheckImageShapeAndData(output_tensor, kRandomPosterize);
}

TEST_F(MindDataTestRandomPosterizeOp, TestOp2) {
   // Test Eager RandomPosterize image = (h, w, c)
  MS_LOG(INFO) << "Doing VisionRandomPosterizeTest.";
  std::shared_ptr<Tensor> de_tensor;
  std::string dataset_root_path = "data/dataset";
  Tensor::CreateFromFile(dataset_root_path + "/testPK/data/class1/0.jpg", &de_tensor);
  auto image = mindspore::MSTensor(std::make_shared<DETensor>(de_tensor));
  std::shared_ptr<TensorTransform>  decode_op = std::make_shared<vision::Decode>();
  std::shared_ptr<TensorTransform> randomposterize_op(new vision::RandomPosterize({3, 5}));
  auto transform = Execute({decode_op, randomposterize_op});
  Status rc = transform(image, &image);
  EXPECT_TRUE(rc.IsOk());
  EXPECT_EQ(image.Shape().size(), 3);
  EXPECT_EQ(image.Shape()[2], 3);
}

TEST_F(MindDataTestRandomPosterizeOp, TestOp3) {
  // Test Eager RandomSolarize image.size = {2, 2, 2, 1}
  MS_LOG(INFO) << "Doing VisionRandomSolarizeTest.";

  std::shared_ptr<Tensor> de_tensor;
  Tensor::CreateFromVector(std::vector<uint8_t>({0, 25, 120, 0, 38, 2, 10, 13}), TensorShape({2, 2, 2, 1}), &de_tensor);
  auto image = mindspore::MSTensor(std::make_shared<DETensor>(de_tensor));
  std::shared_ptr<TensorTransform> randomsolarize_op(new vision::RandomSolarize({12, 25}));
  auto transform = Execute({randomsolarize_op});
  Status rc = transform(image, &image);
  EXPECT_TRUE(rc.IsError());
}
