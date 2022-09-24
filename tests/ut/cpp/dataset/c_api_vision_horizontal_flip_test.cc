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

class MindDataTestHorizontalFlip : public UT::DatasetOpTesting {
 protected:
};

/// Feature: HorizontalFlip op
/// Description: Test HorizontalFlip op in pipeline mode
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestHorizontalFlip, TestHorizontalFlipPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestHorizontalFlip-TestHorizontalFlipPipeline.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto horizontal_flip = std::make_shared<vision::HorizontalFlip>();

  // Create a Map operation on ds
  ds = ds->Map({horizontal_flip});
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

/// Feature: HorizontalFlip op
/// Description: Test HorizontalFlip op in eager mode
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestHorizontalFlip, TestHorizontalFlipEager) {
  MS_LOG(INFO) << "Doing MindDataTestHorizontalFlip-TestHorizontalFlipEager.";

  // Read images
  auto image = ReadFileToTensor("data/dataset/apple.jpg");

  // Transform params
  auto decode = vision::Decode();
  auto horizontal_flip = vision::HorizontalFlip();

  auto transform = Execute({decode, horizontal_flip});
  Status rc = transform(image, &image);

  EXPECT_EQ(rc, Status::OK());
}

/// Feature: HorizontalFlip op
/// Description: Test HorizontalFlip op by processing tensor with dim more than 3
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestHorizontalFlip, TestHorizontalFlipBatch) {
  MS_LOG(INFO) << "Doing MindDataTestHorizontalFlip-TestHorizontalFlipBatch.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 5));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 3;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds, choose batch size 3 to test high dimension input
  int32_t batch_size = 3;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto horizontal_flip = std::make_shared<vision::HorizontalFlip>();

  // Create a Map operation on ds
  ds = ds->Map({horizontal_flip});
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

  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();
}
