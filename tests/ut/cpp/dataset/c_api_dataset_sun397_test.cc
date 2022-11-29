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
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/include/dataset/datasets.h"

using namespace mindspore::dataset;
using mindspore::dataset::DataType;
using mindspore::dataset::Tensor;
using mindspore::dataset::TensorShape;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

/// Feature: SUN397Dataset
/// Description: Test basic usage of SUN397Dataset
/// Expectation: The dataset is processed successfully
TEST_F(MindDataTestPipeline, TestSUN397TrainStandardDataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSUN397TrainStandardDataset.";

  // Create a SUN397 Train Dataset.
  std::string folder_path = datasets_root_path_ + "/testSUN397Data";
  std::shared_ptr<Dataset> ds = SUN397(folder_path, true, std::make_shared<RandomSampler>(false, 4));
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

/// Feature: SUN397Dataset
/// Description: Test usage of SUN397Dataset with pipeline mode
/// Expectation: The dataset is processed successfully
TEST_F(MindDataTestPipeline, TestSUN397TrainDatasetWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSUN397TrainDatasetWithPipeline.";

  // Create two SUN397 Train Dataset.
  std::string folder_path = datasets_root_path_ + "/testSUN397Data";
  std::shared_ptr<Dataset> ds1 = SUN397(folder_path, true, std::make_shared<RandomSampler>(false, 4));
  std::shared_ptr<Dataset> ds2 = SUN397(folder_path, true, std::make_shared<RandomSampler>(false, 4));
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

/// Feature: SUN397Dataset
/// Description: Test iterator of SUN397Dataset with only the image column
/// Expectation: The dataset is processed successfully
TEST_F(MindDataTestPipeline, TestSUN397IteratorOneColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSUN397IteratorOneColumn.";
  // Create a SUN397 Dataset
  std::string folder_path = datasets_root_path_ + "/testSUN397Data";
  std::shared_ptr<Dataset> ds = SUN397(folder_path, true, std::make_shared<RandomSampler>(false, 4));
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

/// Feature: SUN397Dataset
/// Description: Test iterator of SUN397Dataset with wrong column
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestSUN397IteratorWrongColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSUN397IteratorWrongColumn.";
  // Create a SUN397 Dataset
  std::string folder_path = datasets_root_path_ + "/testSUN397Data";
  std::shared_ptr<Dataset> ds = SUN397(folder_path, true, std::make_shared<RandomSampler>(false, 4));
  EXPECT_NE(ds, nullptr);

  // Pass wrong column name
  std::vector<std::string> columns = {"digital"};
  std::shared_ptr<ProjectDataset> project_ds = ds->Project(columns);
  std::shared_ptr<Iterator> iter = project_ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: SUN397Dataset
/// Description: Test usage of GetDatasetSize of SUN397TrainDataset
/// Expectation: Get the correct size
TEST_F(MindDataTestPipeline, TestGetSUN397TrainDatasetSize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetSUN397TrainDatasetSize.";

  // Create a SUN397 Train Dataset.
  std::string folder_path = datasets_root_path_ + "/testSUN397Data";
  std::shared_ptr<Dataset> ds = SUN397(folder_path, true);
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 4);
}

/// Feature: SUN397Dataset
/// Description: Test SUN397Dataset Getters method
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestSUN397TrainDatasetGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSUN397TrainDatasetGetters.";

  // Create a SUN397 Train Dataset.
  std::string folder_path = datasets_root_path_ + "/testSUN397Data";
  std::shared_ptr<Dataset> ds = SUN397(folder_path, true);
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

/// Feature: SUN397Dataset
/// Description: Test SUN397Dataset with invalid folder path input
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestSUN397DatasetFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSUN397DatasetFail.";

  // Create a SUN397 Dataset.
  std::shared_ptr<Dataset> ds = SUN397("", true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid SUN397 input.
  EXPECT_EQ(iter, nullptr);
}

/// Feature: SUN397Dataset
/// Description: Test SUN397Dataset with null sampler
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestSUN397DatasetWithNullSamplerFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSUN397DatasetWithNullSamplerFail.";

  // Create a SUN397 Dataset.
  std::string folder_path = datasets_root_path_ + "/testSUN397Data";
  std::shared_ptr<Dataset> ds = SUN397(folder_path, true, nullptr);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid SUN397 input, sampler cannot be nullptr.
  EXPECT_EQ(iter, nullptr);
}
