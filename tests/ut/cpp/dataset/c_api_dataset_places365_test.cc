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
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/include/dataset/datasets.h"

using namespace mindspore::dataset;
using mindspore::dataset::DataType;
using mindspore::dataset::Tensor;
using mindspore::dataset::TensorShape;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

/// Feature: Places365Dataset
/// Description: Test basic usage of Places365Dataset with train-standard dataset
/// Expectation: The dataset is processed successfully
TEST_F(MindDataTestPipeline, TestPlaces365TrainStandardDataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPlaces365TrainStandardDataset.";

  // Create a Places365 Train Dataset.
  std::string folder_path = datasets_root_path_ + "/testPlaces365Data";
  std::shared_ptr<Dataset> ds =
    Places365(folder_path, "train-standard", true, true, std::make_shared<RandomSampler>(false, 4));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
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

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: Places365Dataset
/// Description: Test basic usage of Places365Dataset with train-challenge dataset
/// Expectation: The dataset is processed successfully
TEST_F(MindDataTestPipeline, TestPlaces365TrainChallengeDataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPlaces365TrainChallengeDataset.";

  // Create a Places365 Train Dataset.
  std::string folder_path = datasets_root_path_ + "/testPlaces365Data";
  std::shared_ptr<Dataset> ds =
    Places365(folder_path, "train-challenge", false, true, std::make_shared<RandomSampler>(false, 4));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
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

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: Places365Dataset
/// Description: Test basic usage of Places365Dataset with val dataset
/// Expectation: The dataset is processed successfully
TEST_F(MindDataTestPipeline, TestPlaces365ValDataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPlaces365ValDataset.";

  // Create a Places365 Test Dataset.
  std::string folder_path = datasets_root_path_ + "/testPlaces365Data";
  std::shared_ptr<Dataset> ds = Places365(folder_path, "val", true, true, std::make_shared<RandomSampler>(false, 4));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
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

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: Places365Dataset
/// Description: Test usage of Places365Dataset with pipeline mode with train-standard dataset
/// Expectation: The dataset is processed successfully
TEST_F(MindDataTestPipeline, TestPlaces365TrainDatasetWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPlaces365TrainDatasetWithPipeline.";

  // Create two Places365 Train Dataset.
  std::string folder_path = datasets_root_path_ + "/testPlaces365Data";
  std::shared_ptr<Dataset> ds1 =
    Places365(folder_path, "train-standard", true, true, std::make_shared<RandomSampler>(false, 4));
  std::shared_ptr<Dataset> ds2 =
    Places365(folder_path, "train-standard", true, true, std::make_shared<RandomSampler>(false, 4));
  EXPECT_NE(ds1, nullptr);
  EXPECT_NE(ds2, nullptr);

  // Create two Repeat operation on ds.
  int32_t repeat_num = 2;
  ds1 = ds1->Repeat(repeat_num);
  EXPECT_NE(ds1, nullptr);
  repeat_num = 2;
  ds2 = ds2->Repeat(repeat_num);
  EXPECT_NE(ds2, nullptr);

  // Create two Project operation on ds.
  std::vector<std::string> column_project = {"image", "label"};
  ds1 = ds1->Project(column_project);
  EXPECT_NE(ds1, nullptr);
  ds2 = ds2->Project(column_project);
  EXPECT_NE(ds2, nullptr);

  // Create a Concat operation on the ds.
  ds1 = ds1->Concat({ds2});
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds1->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
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

  EXPECT_EQ(i, 16);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: Places365Dataset
/// Description: Test iterator of Places365Dataset with only the image column
/// Expectation: The dataset is processed successfully
TEST_F(MindDataTestPipeline, TestPlaces365IteratorOneColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPlaces365IteratorOneColumn.";
  // Create a Places365 Dataset
  std::string folder_path = datasets_root_path_ + "/testPlaces365Data";
  std::shared_ptr<Dataset> ds =
          Places365(folder_path, "train-standard", true, true, std::make_shared<RandomSampler>(false, 4));
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
  std::vector<int64_t> expect_image = {2, 256, 256, 3};

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

/// Feature: Places365Dataset
/// Description: Test iterator of Places365Dataset with wrong column
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestPlaces365IteratorWrongColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPlaces365IteratorWrongColumn.";
  // Create a Places365 Dataset
  std::string folder_path = datasets_root_path_ + "/testPlaces365Data";
  std::shared_ptr<Dataset> ds =
          Places365(folder_path, "train-standard", true, true, std::make_shared<RandomSampler>(false, 4));
  EXPECT_NE(ds, nullptr);
  
  // Pass wrong column name
  std::vector<std::string> columns = {"digital"};
  std::shared_ptr<ProjectDataset> project_ds = ds->Project(columns);
  std::shared_ptr<Iterator> iter = project_ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Places365Dataset
/// Description: Test usage of GetDatasetSize of Places365TrainDataset
/// Expectation: Get the correct size
TEST_F(MindDataTestPipeline, TestGetPlaces365TrainDatasetSize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetPlaces365TrainDatasetSize.";

  // Create a Places365 Train Dataset.
  std::string folder_path = datasets_root_path_ + "/testPlaces365Data";
  std::shared_ptr<Dataset> ds = Places365(folder_path, "train-standard", true, true);
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 4);
}

/// Feature: Places365Dataset
/// Description: Test Places365Dataset Getters method
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestPlaces365TrainDatasetGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPlaces365TrainDatasetGetters.";

  // Create a Places365 Train Dataset.
  std::string folder_path = datasets_root_path_ + "/testPlaces365Data";
  std::shared_ptr<Dataset> ds = Places365(folder_path, "train-standard", true, true);
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 4);
  std::vector<DataType> types = ToDETypes(ds->GetOutputTypes());
  std::vector<TensorShape> shapes = ToTensorShapeVec(ds->GetOutputShapes());
  std::vector<std::string> column_names = {"image", "label"};
  int64_t num_classes = ds->GetNumClasses();
  EXPECT_EQ(types.size(), 2);
  EXPECT_EQ(types[0].ToString(), "uint8");
  EXPECT_EQ(types[1].ToString(), "uint32");
  EXPECT_EQ(shapes.size(), 2);
  EXPECT_EQ(shapes[0].ToString(), "<256,256,3>");
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

/// Feature: Places365Dataset
/// Description: Test Places365Dataset with invalid folder path input
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestPlaces365DatasetFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPlaces365DatasetFail.";

  // Create a Places365 Dataset.
  std::shared_ptr<Dataset> ds = Places365("", "val", true, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Places365 input.
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Places365Dataset
/// Description: Test Places365Dataset with invalid usage
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestPlaces365DatasetWithInvalidUsageFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPlaces365DatasetWithInvalidUsageFail.";

  // Create a Places365 Dataset.
  std::string folder_path = datasets_root_path_ + "/testPlaces365Data";
  std::shared_ptr<Dataset> ds = Places365(folder_path, "validation", true, true);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Places365 input, validation is not a valid usage.
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Places365Dataset
/// Description: Test Places365Dataset with null sampler
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestPlaces365DatasetWithNullSamplerFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPlaces365DatasetWithNullSamplerFail.";

  // Create a Places365 Dataset.
  std::string folder_path = datasets_root_path_ + "/testPlaces365Data";
  std::shared_ptr<Dataset> ds = Places365(folder_path, "train-standard", true, true, nullptr);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Places365 input, sampler cannot be nullptr.
  EXPECT_EQ(iter, nullptr);
}
