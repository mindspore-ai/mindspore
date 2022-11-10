/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/include/dataset/vision.h"

namespace common = mindspore::common;

using namespace mindspore::dataset;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};


/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on SkipOp
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedPipelineSkipOp) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedPipelineSkipOp.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Skip operation on ds
  int32_t count = 3;
  ds = ds->Skip(count);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_EQ(row.size(), 2);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row[0];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }
  MS_LOG(INFO) << "Number of rows: " << i;

  // Expect 10 - 3 = 7 rows
  EXPECT_EQ(i, 7);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on SkipOp with count larger than number of rows
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedPipelineSkipOpLargeCount) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedPipelineSkipOpLargeCount.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Skip operation on ds
  int32_t count = 30;
  ds = ds->Skip(count);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_EQ(row.size(), 0);
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on TakeOp with default count=-1
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestGetNextPullBasedPipelineTakeOpDefault) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedPipelineTakeOpDefault.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 7));
  EXPECT_NE(ds, nullptr);

  // Create a Take operation on ds, default count = -1
  ds = ds->Take();
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row[0];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }
  MS_LOG(INFO) << "Number of rows: " << i;

  // Expect 7 rows
  EXPECT_EQ(i, 7);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on TakeOp with count = 5
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestGetNextPullBasedPipelineTakeOpCount5) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedPipelineTakeOpCount5.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 7));
  EXPECT_NE(ds, nullptr);

  // Create a Take operation on ds, count = 5
  ds = ds->Take(5);
  EXPECT_EQ(ds->GetDatasetSize(), 5);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_EQ(row.size(), 2);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row[0];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }
  MS_LOG(INFO) << "Number of rows: " << i;

  // Expect 5 rows from take(5).
  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Take op
/// Description: Test Take op with invalid count input
/// Expectation: Error message is logged, and CreatePullBasedIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestTakeDatasetError1CreatePullBasedIterator) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTakeDatasetError1CreatePullBasedIterator.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Take operation on ds with invalid count input
  int32_t count = -5;
  auto ds1 = ds->Take(count);
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds1->CreatePullBasedIterator();
  // Expect failure: invalid Op input
  EXPECT_EQ(iter, nullptr);

  // Create a Take operation on ds with invalid count input
  count = 0;
  auto ds2 = ds->Take(count);
  EXPECT_NE(ds2, nullptr);

  // Create an iterator over the result of the above dataset
  iter = ds2->CreatePullBasedIterator();
  // Expect failure: invalid Op input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on TakeOp with invalid count = -5
/// Expectation: Error message is logged, and CreatePullBasedIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestGetNextPullBasedPipelineTakeOpError) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedPipelineTakeOpError.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Take operation on ds with invalid count input
  int32_t count = -5;
  auto ds1 = ds->Take(count);
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<PullIterator> iter = ds1->CreatePullBasedIterator();
  std::vector<mindspore::MSTensor> row;
  // Expect failure: invalid Op input
  EXPECT_EQ(iter, nullptr);

  // Create a Take operation on ds with invalid count input
  count = 0;
  auto ds2 = ds->Take(count);
  EXPECT_NE(ds2, nullptr);

  // Create an iterator over the result of the above dataset
  iter = ds2->CreatePullBasedIterator();
  // Expect failure: invalid Op input
  EXPECT_EQ(iter, nullptr);
}
