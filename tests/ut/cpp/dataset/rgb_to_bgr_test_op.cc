/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <opencv2/opencv.hpp>
#include "common/common.h"
#include "common/cvop_common.h"
#include "include/dataset/datasets.h"
#include "include/dataset/transforms.h"
#include "include/dataset/vision.h"
#include "include/dataset/execute.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/kernels/image/rgb_to_bgr_op.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "utils/log_adapter.h"

using namespace std;
using namespace mindspore::dataset;
using mindspore::dataset::CVTensor;
using mindspore::dataset::BorderType;
using mindspore::dataset::Tensor;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;


class MindDataTestRgbToBgrOp : public UT::DatasetOpTesting {
 protected:
};


TEST_F(MindDataTestRgbToBgrOp, TestOp1) {
  // Eager
  MS_LOG(INFO) << "Doing MindDataTestGaussianBlur-TestGaussianBlurEager.";

  // Read images
  auto image = ReadFileToTensor("data/dataset/apple.jpg");

  // Transform params
  auto decode = vision::Decode();
  auto rgb2bgr_op = vision::RGB2BGR();

  auto transform = Execute({decode, rgb2bgr_op});
  Status rc = transform(image, &image);

  EXPECT_EQ(rc, Status::OK());
}


TEST_F(MindDataTestRgbToBgrOp, TestOp2) {
  // pipeline
  MS_LOG(INFO) << "Basic Function Test.";
  // create two imagenet dataset
  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds1 = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds1, nullptr);
  std::shared_ptr<Dataset> ds2 = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds2, nullptr);

  auto rgb2bgr_op = vision::RGB2BGR();
  
  ds1 = ds1->Map({rgb2bgr_op});
  EXPECT_NE(ds1, nullptr);

  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  EXPECT_NE(iter1, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row1;
  iter1->GetNextRow(&row1);

  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  EXPECT_NE(iter2, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row2;
  iter2->GetNextRow(&row2);

  uint64_t i = 0;
  while (row1.size() != 0) {
    i++;
    auto image =row1["image"];
    iter1->GetNextRow(&row1);
    iter2->GetNextRow(&row2);
  }
  EXPECT_EQ(i, 2);

  iter1->Stop();
  iter2->Stop();
}
