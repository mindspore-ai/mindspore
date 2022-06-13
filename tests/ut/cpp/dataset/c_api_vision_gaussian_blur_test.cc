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
#include "common/common.h"
#include "minddata/dataset/include/dataset/datasets.h"
#include "minddata/dataset/include/dataset/execute.h"
#include "minddata/dataset/include/dataset/vision.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;

class MindDataTestGaussianBlur : public UT::DatasetOpTesting {
 protected:
};

/// Feature: GaussianBlur op
/// Description: Test GaussianBlur op with invalid parameters
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestGaussianBlur, TestGaussianBlurParamCheck) {
  MS_LOG(INFO) << "Doing MindDataTestGaussianBlur-TestGaussianBlurParamCheck with invalid parameters.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Case 1: Kernel size is not positive
  // Create objects for the tensor ops
  auto gaussian_blur1 = std::make_shared<vision::GaussianBlur>(std::vector<int32_t>{-1});
  auto ds1 = ds->Map({gaussian_blur1});
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid kernel_size for GaussianBlur
  EXPECT_EQ(iter1, nullptr);

  // Case 2: Kernel size is not odd
  // Create objects for the tensor ops
  auto gaussian_blur2 = std::make_shared<vision::GaussianBlur>(std::vector<int32_t>{2, 2}, std::vector<float>{3, 3});
  auto ds2 = ds->Map({gaussian_blur2});
  EXPECT_NE(ds2, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid kernel_size for GaussianBlur
  EXPECT_EQ(iter2, nullptr);

  // Case 3: Sigma is not positive
  // Create objects for the tensor ops
  auto gaussian_blur3 = std::make_shared<vision::GaussianBlur>(std::vector<int32_t>{3}, std::vector<float>{-3});
  auto ds3 = ds->Map({gaussian_blur3});
  EXPECT_NE(ds3, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid sigma for GaussianBlur
  EXPECT_EQ(iter3, nullptr);
}

/// Feature: GaussianBlur op
/// Description: Test GaussianBlur op in pipeline mode
/// Expectation: Runs successfully
TEST_F(MindDataTestGaussianBlur, TestGaussianBlurPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestGaussianBlur-TestGaussianBlurPipeline.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto gaussian_blur = std::make_shared<vision::GaussianBlur>(std::vector<int32_t>{3, 3}, std::vector<float>{5, 5});

  // Create a Map operation on ds
  ds = ds->Map({gaussian_blur});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: GaussianBlur op
/// Description: Test GaussianBlur op in eager mode
/// Expectation: Runs successfully
TEST_F(MindDataTestGaussianBlur, TestGaussianBlurEager) {
  MS_LOG(INFO) << "Doing MindDataTestGaussianBlur-TestGaussianBlurEager.";

  // Read images
  auto image = ReadFileToTensor("data/dataset/apple.jpg");

  // Transform params
  auto decode = vision::Decode();
  auto gaussian_blur = vision::GaussianBlur({7}, {3.5});

  auto transform = Execute({decode, gaussian_blur});
  Status rc = transform(image, &image);

  EXPECT_EQ(rc, Status::OK());
}
