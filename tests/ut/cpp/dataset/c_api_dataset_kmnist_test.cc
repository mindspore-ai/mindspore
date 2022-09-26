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

using namespace mindspore::dataset;
using mindspore::dataset::DataType;
using mindspore::dataset::Tensor;
using mindspore::dataset::TensorShape;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

/// Feature: KMnistDataset
/// Description: Test basic usage of KMnistDataset
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestKMnistTestDataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestKMnistTestDataset.";

  // Create a KMnist Test Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = KMnist(folder_path, "test", std::make_shared<RandomSampler>(false, 10));

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

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: KMnistDataset
/// Description: Test KMnistDataset in pipeline mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestKMnistTestDatasetWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestKMnistTestDatasetWithPipeline.";

  std::string folder_path = datasets_root_path_ + "/testMnistData/";

  // Create two KMnist Test Dataset
  std::shared_ptr<Dataset> ds1 = KMnist(folder_path, "test", std::make_shared<RandomSampler>(false, 10));
  std::shared_ptr<Dataset> ds2 = KMnist(folder_path, "test", std::make_shared<RandomSampler>(false, 10));
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

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: KMnistDataset
/// Description: Test iterator of KMnistDataset with only the image column
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestKMnistDatasetIteratorOneColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestKMnistIteratorOneColumn.";
  // Create a KMnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = KMnist(folder_path, "all", std::make_shared<RandomSampler>(false, 4));
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 2;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // Only select "image" column and drop others
  std::vector<std::string> columns = {"image"};
  std::shared_ptr<ProjectDataset> project_ds = ds->Project(columns);
  std::shared_ptr<Iterator> iter = project_ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  std::vector<int64_t> expect_image = {2, 28, 28, 1};

  uint64_t i = 0;
  while (row.size() != 0) {
    for (auto &v : row) {
      MS_LOG(INFO) << "image shape:" << v.Shape();
      EXPECT_EQ(expect_image, v.Shape());
    }
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: KMnistDataset
/// Description: Test iterator of KMnistDataset with wrong column
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestKMnistDatasetIteratorWrongColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestKMnistDatasetIteratorWrongColumn.";
  // Create a KMnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = KMnist(folder_path, "all", std::make_shared<RandomSampler>(false, 4));
  EXPECT_NE(ds, nullptr);

  // Pass wrong column name
  std::vector<std::string> columns = {"digital"};
  std::shared_ptr<ProjectDataset> project_ds = ds->Project(columns);
  std::shared_ptr<Iterator> iter = project_ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: KMnistDataset
/// Description: Test KMnistDataset GetDatasetSize
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestGetKMnistTestDatasetSize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetKMnistTestDatasetSize.";

  std::string folder_path = datasets_root_path_ + "/testMnistData/";

  // Create a KMnist Test Dataset
  std::shared_ptr<Dataset> ds = KMnist(folder_path, "test");
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 10000);
}

/// Feature: KMnistDataset
/// Description: Test KMnistDataset Getters method
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestKMnistTestDatasetGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestKMnistTestDatasetGetters.";

  // Create a KMnist Test Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = KMnist(folder_path, "test");
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 10000);
  std::vector<DataType> types = ToDETypes(ds->GetOutputTypes());
  std::vector<TensorShape> shapes = ToTensorShapeVec(ds->GetOutputShapes());
  std::vector<std::string> column_names = {"image", "label"};
  int64_t num_classes = ds->GetNumClasses();
  EXPECT_EQ(types.size(), 2);
  EXPECT_EQ(types[0].ToString(), "uint8");
  EXPECT_EQ(types[1].ToString(), "uint32");
  EXPECT_EQ(shapes.size(), 2);
  EXPECT_EQ(shapes[0].ToString(), "<28,28,1>");
  EXPECT_EQ(shapes[1].ToString(), "<>");
  EXPECT_EQ(num_classes, -1);
  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);

  EXPECT_EQ(ds->GetDatasetSize(), 10000);
  EXPECT_EQ(ToDETypes(ds->GetOutputTypes()), types);
  EXPECT_EQ(ToTensorShapeVec(ds->GetOutputShapes()), shapes);
  EXPECT_EQ(ds->GetNumClasses(), -1);

  EXPECT_EQ(ds->GetColumnNames(), column_names);
  EXPECT_EQ(ds->GetDatasetSize(), 10000);
  EXPECT_EQ(ToDETypes(ds->GetOutputTypes()), types);
  EXPECT_EQ(ToTensorShapeVec(ds->GetOutputShapes()), shapes);
  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);
  EXPECT_EQ(ds->GetNumClasses(), -1);
  EXPECT_EQ(ds->GetDatasetSize(), 10000);
}

/// Feature: KMnistDataset
/// Description: Test KMnistDataset with wrong column
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestKMnistIteratorWrongColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestKMnistIteratorOneColumn.";
  // Create a KMnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = KMnist(folder_path, "all", std::make_shared<RandomSampler>(false, 4));
  EXPECT_NE(ds, nullptr);

  // Pass wrong column name
  std::vector<std::string> columns = {"digital"};
  std::shared_ptr<ProjectDataset> project_ds = ds->Project(columns);
  std::shared_ptr<Iterator> iter = project_ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: KMnistDataset
/// Description: Test KMnistDataset with empty string as folder path
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestKMnistDatasetFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestKMnistDatasetFail.";

  // Create a KMnist Dataset
  std::shared_ptr<Dataset> ds = KMnist("", "train", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid KMnist input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: KMnistDataset
/// Description: Test KMnistDataset with invalid usage
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestKMnistDatasetWithInvalidUsageFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestKMnistDatasetWithInvalidUsageFail.";

  // Create a KMnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = KMnist(folder_path, "validation");
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid KMnist input, validation is not a valid usage
  EXPECT_EQ(iter, nullptr);
}

/// Feature: KMnistDataset
/// Description: Test KMnistDataset with null sampler
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestKMnistDatasetWithNullSamplerFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestKMnistUDatasetWithNullSamplerFail.";

  // Create a KMnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = KMnist(folder_path, "all", nullptr);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid KMnist input, sampler cannot be nullptr
  EXPECT_EQ(iter, nullptr);
}
