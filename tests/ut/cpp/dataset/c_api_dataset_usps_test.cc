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

/// Feature: USPSDataset
/// Description: Test basic usage of USPSDataset with train dataset
/// Expectation: Get correct number of data
TEST_F(MindDataTestPipeline, TestUSPSTrainDataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUSPSTrainDataset.";

  // Create a USPS Train Dataset
  std::string folder_path = datasets_root_path_ + "/testUSPSDataset/";
  std::shared_ptr<Dataset> ds = USPS(folder_path, "train");
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

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: USPSDataset
/// Description: Test basic usage of USPSDataset with test dataset
/// Expectation: Get correct number of data
TEST_F(MindDataTestPipeline, TestUSPSTestDataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUSPSTestDataset.";

  // Create a USPS Test Dataset
  std::string folder_path = datasets_root_path_ + "/testUSPSDataset/";
  std::shared_ptr<Dataset> ds = USPS(folder_path, "test");
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

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: USPSDataset
/// Description: Test basic usage of USPSDataset with all dataset
/// Expectation: Get correct number of data
TEST_F(MindDataTestPipeline, TestUSPSAllDataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUSPSAllDataset.";

  // Create a USPS Test Dataset
  std::string folder_path = datasets_root_path_ + "/testUSPSDataset/";
  std::shared_ptr<Dataset> ds = USPS(folder_path, "all");
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

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: USPSDataset
/// Description: Test usage of USPSDataset with pipeline mode
/// Expectation: Get correct number of data
TEST_F(MindDataTestPipeline, TestUSPSDatasetWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUSPSTrainDatasetWithPipeline.";

  // Create two USPS Train Dataset
  std::string folder_path = datasets_root_path_ + "/testUSPSDataset/";
  std::shared_ptr<Dataset> ds1 = USPS(folder_path, "train");
  std::shared_ptr<Dataset> ds2 = USPS(folder_path, "train");
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

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: USPSDataset
/// Description: Test iterator of USPSDataset with only the "image" column
/// Expectation: Get correct data
TEST_F(MindDataTestPipeline, TestUSPSIteratorOneColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUSPSIteratorOneColumn.";
  // Create a USPS Dataset
  std::string folder_path = datasets_root_path_ + "/testUSPSDataset/";
  std::shared_ptr<Dataset> ds = USPS(folder_path, "train");
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
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
  std::vector<int64_t> expect_image = {1, 16, 16, 1};

  uint64_t i = 0;
  while (row.size() != 0) {
    for (auto &v : row) {
      MS_LOG(INFO) << "image shape:" << v.Shape();
      EXPECT_EQ(expect_image, v.Shape());
    }
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: USPSDataset
/// Description: Test iterator of USPSDataset with wrong column
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestUSPSIteratorWrongColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUSPSIteratorWrongColumn.";
  // Create a USPS Dataset
  std::string folder_path = datasets_root_path_ + "/testUSPSDataset/";
  std::shared_ptr<Dataset> ds = USPS(folder_path, "train");
  EXPECT_NE(ds, nullptr);
  
  // Pass wrong column name
  std::vector<std::string> columns = {"digital"};
  std::shared_ptr<ProjectDataset> project_ds = ds->Project(columns);
  std::shared_ptr<Iterator> iter = project_ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: USPSDataset
/// Description: Test GetDatasetSize of USPSDataset
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestGetUSPSDatasetSize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetUSPSTrainDatasetSize.";

  // Create a USPS Train Dataset
  std::string folder_path = datasets_root_path_ + "/testUSPSDataset/";
  std::shared_ptr<Dataset> ds = USPS(folder_path, "train");
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 3);
}

/// Feature: USPSDataset
/// Description: Test usage of getters USPSDataset
/// Expectation: Get correct number of data and correct tensor shape
TEST_F(MindDataTestPipeline, TestUSPSDatasetGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUSPSTrainDatasetGetters.";

  // Create a USPS Train Dataset
  std::string folder_path = datasets_root_path_ + "/testUSPSDataset/";
  std::shared_ptr<Dataset> ds = USPS(folder_path, "train");
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 3);
  std::vector<DataType> types = ToDETypes(ds->GetOutputTypes());
  std::vector<TensorShape> shapes = ToTensorShapeVec(ds->GetOutputShapes());
  std::vector<std::string> column_names = {"image", "label"};
  EXPECT_EQ(types.size(), 2);
  EXPECT_EQ(types[0].ToString(), "uint8");
  EXPECT_EQ(types[1].ToString(), "uint32");
  EXPECT_EQ(shapes.size(), 2);
  EXPECT_EQ(shapes[0].ToString(), "<16,16,1>");
  EXPECT_EQ(shapes[1].ToString(), "<>");
  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);

  EXPECT_EQ(ds->GetDatasetSize(), 3);
  EXPECT_EQ(ToDETypes(ds->GetOutputTypes()), types);
  EXPECT_EQ(ToTensorShapeVec(ds->GetOutputShapes()), shapes);

  EXPECT_EQ(ds->GetColumnNames(), column_names);
  EXPECT_EQ(ds->GetDatasetSize(), 3);
  EXPECT_EQ(ToDETypes(ds->GetOutputTypes()), types);
  EXPECT_EQ(ToTensorShapeVec(ds->GetOutputShapes()), shapes);
  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);
  EXPECT_EQ(ds->GetDatasetSize(), 3);
}

/// Feature: USPSDataset
/// Description: Test failure of USPSDataset with empty string as folder path
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestUSPSDatasetFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUSPSDatasetFail.";

  // Create a USPS Dataset
  std::shared_ptr<Dataset> ds = USPS("", "train");
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid USPS input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: USPSDataset
/// Description: Test failure of USPSDataset with invalid usage
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestUSPSDatasetWithInvalidUsageFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUSPSDatasetWithInvalidUsageFail.";

  // Create a USPS Dataset
  std::string folder_path = datasets_root_path_ + "/testUSPSDataset/";
  std::shared_ptr<Dataset> ds = USPS(folder_path, "validation");
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid USPS input, validation is not a valid usage
  EXPECT_EQ(iter, nullptr);
}
