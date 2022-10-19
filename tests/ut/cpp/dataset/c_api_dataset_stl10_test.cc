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

/// Feature: STL10Dataset
/// Description: Test basic usage of STL10Dataset with train dataset
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestSTL10TrainDataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSTL10TrainDataset.";

  // Create a STL10 Dataset
  std::string folder_path = datasets_root_path_ + "/testSTL10Data/";
  std::shared_ptr<Dataset> ds = STL10(folder_path, "train", std::make_shared<RandomSampler>(false, 1));
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

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: STL10Dataset
/// Description: Test basic usage of STL10Dataset with test dataset
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestSTL10TestDataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSTL10TestDataset.";

  // Create a STL10 Dataset
  std::string folder_path = datasets_root_path_ + "/testSTL10Data/";
  std::shared_ptr<Dataset> ds = STL10(folder_path, "test", std::make_shared<RandomSampler>(false, 1));
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

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: STL10Dataset
/// Description: Test basic usage of STL10Dataset with unlabeled dataset
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestSTL10UnlabeledDataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSTL10UnlabeledDataset.";

  // Create a STL10 Dataset
  std::string folder_path = datasets_root_path_ + "/testSTL10Data/";
  std::shared_ptr<Dataset> ds = STL10(folder_path, "unlabeled", std::make_shared<RandomSampler>(false, 1));
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

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: STL10Dataset
/// Description: Test basic usage of STL10Dataset with train+unlabeled dataset
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestSTL10TrainUnlabeledDataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSTL10TrainUnlabeledDataset.";

  // Create a STL10 Dataset
  std::string folder_path = datasets_root_path_ + "/testSTL10Data/";
  std::shared_ptr<Dataset> ds = STL10(folder_path, "train+unlabeled", std::make_shared<RandomSampler>(false, 2));
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

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: STL10Dataset
/// Description: Test basic usage of STL10Dataset with all dataset
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestSTL10AllDataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSTL10AllDataset.";

  // Create a STL10 Dataset
  std::string folder_path = datasets_root_path_ + "/testSTL10Data/";
  std::shared_ptr<Dataset> ds = STL10(folder_path, "all", std::make_shared<RandomSampler>(false, 2));
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

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: STL10Dataset
/// Description: Test usage of STL10Dataset with pipeline mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestSTL10TrainDatasetWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSTL10TrainDatasetWithPipeline.";

  // Create two STL10 Dataset
  std::string folder_path = datasets_root_path_ + "/testSTL10Data/";
  std::shared_ptr<Dataset> ds1 = STL10(folder_path, "train", std::make_shared<RandomSampler>(false, 2));
  std::shared_ptr<Dataset> ds2 = STL10(folder_path, "train", std::make_shared<RandomSampler>(false, 2));
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

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: STL10Dataset
/// Description: Test iterator of STL10Dataset with only the image column
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestSTL10DatasetIteratorOneColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSTL10DatasetIteratorOneColumn.";
  // Create a STL10 Dataset
  std::string folder_path = datasets_root_path_ + "/testSTL10Data/";
  std::shared_ptr<Dataset> ds = STL10(folder_path, "train", std::make_shared<RandomSampler>(false, 1));
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
  std::vector<int64_t> expect_shape = {1, 96, 96, 3};

  uint64_t i = 0;
  while (row.size() != 0) {
    for (auto &v : row) {
      MS_LOG(INFO) << "image shape:" << v.Shape();
      EXPECT_EQ(expect_shape, v.Shape());
    }
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: STL10Dataset
/// Description: Test iterator of STL10Dataset with wrong column
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestSTL10DatasetIteratorWrongColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSTL10DatasetIteratorWrongColumn.";
  // Create a STL10 Dataset
  std::string folder_path = datasets_root_path_ + "/testSTL10Data/";
  std::shared_ptr<Dataset> ds = STL10(folder_path, "train", std::make_shared<RandomSampler>(false, 1));
  EXPECT_NE(ds, nullptr);

  // Pass wrong column name
  std::vector<std::string> columns = {"digital"};
  std::shared_ptr<ProjectDataset> project_ds = ds->Project(columns);
  std::shared_ptr<Iterator> iter = project_ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: STL10Dataset
/// Description: Test usage of STL10Dataset with train dataset GetDatasetSize
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestSTL10GetTrainDatasetSize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSTL10GetTrainDatasetSize.";

  // Create a STL10 Dataset
  std::string folder_path = datasets_root_path_ + "/testSTL10Data/";
  std::shared_ptr<Dataset> ds = STL10(folder_path, "train");
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 1);
}

/// Feature: STL10Dataset
/// Description: Test STL10Dataset with test dataset GetDatasetSize
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestSTL10GetTestDatasetSize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSTL10GetTestDatasetSize.";

  // Create a STL10 Dataset
  std::string folder_path = datasets_root_path_ + "/testSTL10Data/";
  std::shared_ptr<Dataset> ds = STL10(folder_path, "test");
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 1);
}

/// Feature: STL10Dataset
/// Description: Test STL10Dataset with unlabeled dataset GetDatasetSize
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestSTL10GetUnlabeledDatasetSize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSTL10GetUnlabeledDatasetSize.";

  // Create a STL10 Dataset
  std::string folder_path = datasets_root_path_ + "/testSTL10Data/";
  std::shared_ptr<Dataset> ds = STL10(folder_path, "unlabeled");
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 1);
}

/// Feature: STL10Dataset
/// Description: Test STL10Dataset with train+unlabeled dataset GetDatasetSize
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestSTL10GetTrainUnlabeledDatasetSize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSTL10GetTrainUnlabeledDatasetSize.";

  // Create a STL10 Dataset
  std::string folder_path = datasets_root_path_ + "/testSTL10Data/";
  std::shared_ptr<Dataset> ds = STL10(folder_path, "train+unlabeled");
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 2);
}

/// Feature: STL10Dataset
/// Description: Test STL10Dataset with all dataset GetDatasetSize
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestSTL10GetAllDatasetSize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSTL10GetAllDatasetSize.";

  // Create a STL10 Dataset
  std::string folder_path = datasets_root_path_ + "/testSTL10Data/";
  std::shared_ptr<Dataset> ds = STL10(folder_path, "all");
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 3);
}

/// Feature: STL10Dataset
/// Description: Test STL10Dataset Getters method
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestSTL10TrainGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSTL10TrainGetter.";

  // Create a STL10 Dataset
  std::string folder_path = datasets_root_path_ + "/testSTL10Data/";
  std::shared_ptr<Dataset> ds = STL10(folder_path, "train");
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 1);
  std::vector<DataType> types = ToDETypes(ds->GetOutputTypes());
  std::vector<TensorShape> shapes = ToTensorShapeVec(ds->GetOutputShapes());
  std::vector<std::string> column_names = {"image", "label"};
  int64_t num_classes = ds->GetNumClasses();
  EXPECT_EQ(types.size(), 2);
  EXPECT_EQ(types[0].ToString(), "uint8");
  EXPECT_EQ(types[1].ToString(), "int32");
  EXPECT_EQ(shapes.size(), 2);
  EXPECT_EQ(shapes[0].ToString(), "<96,96,3>");
  EXPECT_EQ(shapes[1].ToString(), "<>");
  EXPECT_EQ(num_classes, -1);
  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);

  EXPECT_EQ(ds->GetDatasetSize(), 1);
  EXPECT_EQ(ToDETypes(ds->GetOutputTypes()), types);
  EXPECT_EQ(ToTensorShapeVec(ds->GetOutputShapes()), shapes);
  EXPECT_EQ(ds->GetNumClasses(), -1);

  EXPECT_EQ(ds->GetColumnNames(), column_names);
  EXPECT_EQ(ds->GetDatasetSize(), 1);
  EXPECT_EQ(ToDETypes(ds->GetOutputTypes()), types);
  EXPECT_EQ(ToTensorShapeVec(ds->GetOutputShapes()), shapes);
  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);
  EXPECT_EQ(ds->GetNumClasses(), -1);
  EXPECT_EQ(ds->GetDatasetSize(), 1);
}

/// Feature: STL10Dataset
/// Description: Test STL10Dataset with invalid folder path input
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, testSTL10DataFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-testSTL10DataFail.";

  // Create a STL10 Dataset
  std::shared_ptr<Dataset> ds = STL10("", "train", std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid STL10 input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: STL10Dataset
/// Description: Test STL10Dataset with invalid usage
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, testSTL10DataWithInvalidUsageFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-testSTL10DataWithNullSamplerFail.";

  // Create a STL10 Dataset
  std::string folder_path = datasets_root_path_ + "/testSTL10Data/";
  std::shared_ptr<Dataset> ds = STL10(folder_path, "validation");
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid STL10 input, validation is not a valid usage
  EXPECT_EQ(iter, nullptr);
}

/// Feature: STL10Dataset
/// Description: Test STL10Dataset with null sampler
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, testSTL10DataWithNullSamplerFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-testSTL10DataWithNullSamplerFail.";

  // Create a STL10 Dataset
  std::string folder_path = datasets_root_path_ + "/testSTL10Data/";
  std::shared_ptr<Dataset> ds = STL10(folder_path, "train", nullptr);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid STL10 input, sampler cannot be nullptr
  EXPECT_EQ(iter, nullptr);
}
