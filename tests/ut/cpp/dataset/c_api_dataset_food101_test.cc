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

using namespace mindspore::dataset;
using mindspore::dataset::DataType;
using mindspore::dataset::Tensor;
using mindspore::dataset::TensorShape;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

/// Feature: Food101Dataset
/// Description: Test basic usage of Food101Dataset
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestFood101TestDataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFood101TestDataset.";

  // Create a Food101 Test Dataset
  std::string folder_path = datasets_root_path_ + "/testFood101Data/";
  std::shared_ptr<Dataset> ds = Food101(folder_path, "test", true, std::make_shared<RandomSampler>(false, 4));

  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("image"), row.end());
  EXPECT_NE(row.find("label"), row.end());

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Food101Dataset
/// Description: Test Food101Dataset in pipeline mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestFood101TestDatasetWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFood101TestDatasetWithPipeline.";

  std::string folder_path = datasets_root_path_ + "/testFood101Data/";

  // Create two Food101 Test Dataset
  std::shared_ptr<Dataset> ds1 = Food101(folder_path, "test", true, std::make_shared<RandomSampler>(false, 4));
  std::shared_ptr<Dataset> ds2 = Food101(folder_path, "test", true,std::make_shared<RandomSampler>(false, 4));
  EXPECT_NE(ds1, nullptr);
  EXPECT_NE(ds2, nullptr);

  // Create two Repeat operation on ds
  int32_t repeat_num = 1;
  ds1 = ds1->Repeat(repeat_num);
  EXPECT_NE(ds1, nullptr);
  repeat_num = 1;
  ds2 = ds2->Repeat(repeat_num);
  EXPECT_NE(ds2, nullptr);

  // Create two Project operation on ds
  std::vector<std::string> column_project = {"image", "label"};
  ds1 = ds1->Project(column_project);
  EXPECT_NE(ds1, nullptr);
  ds2 = ds2->Project(column_project);
  EXPECT_NE(ds2, nullptr);

  // Create a Concat operation on the ds
  ds1 = ds1->Concat({ds2});
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds1->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("image"), row.end());
  EXPECT_NE(row.find("label"), row.end());

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 8);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Food101Dataset
/// Description: Test Food101Dataset GetDatasetSize
/// Expectation: Correct size of dataset
TEST_F(MindDataTestPipeline, TestGetFood101TestDatasetSize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetFood101TestDatasetSize.";

  std::string folder_path = datasets_root_path_ + "/testFood101Data/";

  // Create a Food101 Test Dataset
  std::shared_ptr<Dataset> ds = Food101(folder_path, "all");
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 8);
}

/// Feature: Food101Dataset
/// Description: Test Food101Dataset Getters method
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestFood101TestDatasetGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFood101TestDatasetGetters.";

  // Create a Food101 Test Dataset
  std::string folder_path = datasets_root_path_ + "/testFood101Data/";
  std::shared_ptr<Dataset> ds = Food101(folder_path, "test");
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 4);
  std::vector<DataType> types = ToDETypes(ds->GetOutputTypes());
  std::vector<TensorShape> shapes = ToTensorShapeVec(ds->GetOutputShapes());
  std::vector<std::string> column_names = {"image", "label"};
  int64_t num_classes = ds->GetNumClasses();
  EXPECT_EQ(types.size(), 2);
  EXPECT_EQ(types[0].ToString(), "uint8");
  EXPECT_EQ(types[1].ToString(), "string");
  EXPECT_EQ(shapes.size(), 2);
  EXPECT_EQ(shapes[1].ToString(), "<>");
  EXPECT_EQ(num_classes, -1);
  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);

  EXPECT_EQ(ds->GetDatasetSize(), 4);
  EXPECT_EQ(ToDETypes(ds->GetOutputTypes()), types);
  EXPECT_EQ(ToTensorShapeVec(ds->GetOutputShapes()), shapes);
  EXPECT_EQ(ds->GetNumClasses(), -1);

  EXPECT_EQ(ds->GetColumnNames(), column_names);
  EXPECT_EQ(ds->GetDatasetSize(), 4);
  EXPECT_EQ(ToDETypes(ds->GetOutputTypes()), types);
  EXPECT_EQ(ToTensorShapeVec(ds->GetOutputShapes()), shapes);
  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);
  EXPECT_EQ(ds->GetNumClasses(), -1);
  EXPECT_EQ(ds->GetDatasetSize(), 4);
}

/// Feature: Food101Dataset
/// Description: Test iterator of Food101Dataset with wrong column
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestFood101IteratorWrongColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFood101IteratorOneColumn.";
  // Create a Food101 Dataset
  std::string folder_path = datasets_root_path_ + "/testFood101Data/";
  std::shared_ptr<Dataset> ds = Food101(folder_path, "all", false, std::make_shared<RandomSampler>(false, 4));
  EXPECT_NE(ds, nullptr);

  // Pass wrong column name
  std::vector<std::string> columns = {"digital"};
  std::shared_ptr<ProjectDataset> project_ds = ds->Project(columns);
  std::shared_ptr<Iterator> iter = project_ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Food101Dataset
/// Description: Test Food101Dataset with empty string as the folder path
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestFood101DatasetFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFood101DatasetFail.";

  // Create a Food101 Dataset
  std::shared_ptr<Dataset> ds = Food101("", "train", false, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Food101 input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Food101Dataset
/// Description: Test Food101Dataset with invalid usage
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestFood101DatasetWithInvalidUsageFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFood101DatasetWithInvalidUsageFail.";

  // Create a Food101 Dataset
  std::string folder_path = datasets_root_path_ + "/testFood101Data/";
  std::shared_ptr<Dataset> ds = Food101(folder_path, "validation");
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Food101 input, validation is not a valid usage
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Food101Dataset
/// Description: Test Food101Dataset with null sampler
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestFood101DatasetWithNullSamplerFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFood101UDatasetWithNullSamplerFail.";

  // Create a Food101 Dataset
  std::string folder_path = datasets_root_path_ + "/testFood101Data/";
  std::shared_ptr<Dataset> ds = Food101(folder_path, "all", false, nullptr);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Food101 input, sampler cannot be nullptr
  EXPECT_EQ(iter, nullptr);
}
