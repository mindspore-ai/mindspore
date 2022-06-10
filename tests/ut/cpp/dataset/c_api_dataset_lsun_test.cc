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

/// Feature: LSUNDataset
/// Description: Test LSUNDataset with train dataset
/// Expectation: Get correct LSUNDataset
TEST_F(MindDataTestPipeline, TestLSUNTrainDataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLSUNTrainDataset.";

  // Create a LSUN Train Dataset.
  std::string folder_path = datasets_root_path_ + "/testLSUN";
  std::shared_ptr<Dataset> ds = LSUN(folder_path, "train", {}, false, std::make_shared<RandomSampler>(false, 5));
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

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: LSUNDataset
/// Description: Test LSUNDataset with valid dataset
/// Expectation: Get correct LSUNDataset
TEST_F(MindDataTestPipeline, TestLSUNValidDataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLSUNValidDataset.";

  // Create a LSUN Validation Dataset.
  std::string folder_path = datasets_root_path_ + "/testLSUN";
  std::shared_ptr<Dataset> ds = LSUN(folder_path, "valid", {}, false, std::make_shared<RandomSampler>(false, 5));
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

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: LSUNDataset
/// Description: Test LSUNDataset with test dataset
/// Expectation: Get correct LSUNDataset
TEST_F(MindDataTestPipeline, TestLSUNTestDataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLSUNTestDataset.";

  // Create a LSUN Test Dataset.
  std::string folder_path = datasets_root_path_ + "/testLSUN";
  std::shared_ptr<Dataset> ds = LSUN(folder_path, "test", {}, false, std::make_shared<RandomSampler>(false, 2));
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

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: LSUNDataset
/// Description: Test LSUNDataset with all dataset
/// Expectation: Get correct LSUNDataset
TEST_F(MindDataTestPipeline, TestLSUNAllDataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLSUNAllDataset.";

  // Create a LSUN Test Dataset.
  std::string folder_path = datasets_root_path_ + "/testLSUN";
  std::shared_ptr<Dataset> ds = LSUN(folder_path, "all", {}, false, std::make_shared<RandomSampler>(false, 2));
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

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: LSUNDataset
/// Description: Test LSUNDataset with classes
/// Expectation: Get correct LSUNDataset
TEST_F(MindDataTestPipeline, TestLSUNClassesDataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLSUNClassesDataset.";

  // Create a LSUN Train Dataset.
  std::string folder_path = datasets_root_path_ + "/testLSUN";
  std::shared_ptr<Dataset> ds =
    LSUN(folder_path, "train", {"bedroom", "classroom"}, false, std::make_shared<RandomSampler>(false, 5));
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

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: LSUNDataset
/// Description: Test LSUNDataset in pipeline mode
/// Expectation: Get correct LSUNDataset
TEST_F(MindDataTestPipeline, TestLSUNDatasetWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLSUNDatasetWithPipeline.";

  // Create two LSUN Train Dataset.
  std::string folder_path = datasets_root_path_ + "/testLSUN";
  std::shared_ptr<Dataset> ds1 =
    LSUN(folder_path, "train", {"bedroom", "classroom"}, false, std::make_shared<RandomSampler>(false, 5));
  std::shared_ptr<Dataset> ds2 =
    LSUN(folder_path, "train", {"bedroom", "classroom"}, false, std::make_shared<RandomSampler>(false, 5));
  EXPECT_NE(ds1, nullptr);
  EXPECT_NE(ds2, nullptr);

  // Create two Repeat operation on ds.
  int32_t repeat_num = 1;
  ds1 = ds1->Repeat(repeat_num);
  EXPECT_NE(ds1, nullptr);
  repeat_num = 1;
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

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: LSUNDataset
/// Description: Test LSUNDataset GetDatasetSize
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestLSUNGetDatasetSize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLSUNGetDatasetSize.";

  // Create a LSUN Train Dataset.
  std::string folder_path = datasets_root_path_ + "/testLSUN";
  std::shared_ptr<Dataset> ds = LSUN(folder_path, "train", {}, false);
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 2);
}

/// Feature: LSUNDataset.
/// Description: Test LSUNDataset.
/// Expectation: Get correct lsun dataset.
TEST_F(MindDataTestPipeline, TestLSUNClassesGetDatasetSize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetLSUNClassesDatasetSize.";

  // Create a LSUN Train Dataset.
  std::string folder_path = datasets_root_path_ + "/testLSUN";
  std::shared_ptr<Dataset> ds = LSUN(folder_path, "train", {"bedroom", "classroom"}, false);
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 2);
}

/// Feature: LSUNDataset
/// Description: Test LSUNDataset Getters method
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestLSUNDatasetGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLSUNDatasetGetters.";

  // Create a LSUN Train Dataset.
  std::string folder_path = datasets_root_path_ + "/testLSUN";
  std::shared_ptr<Dataset> ds = LSUN(folder_path, "train", {}, true);
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 2);
  std::vector<DataType> types = ToDETypes(ds->GetOutputTypes());
  std::vector<TensorShape> shapes = ToTensorShapeVec(ds->GetOutputShapes());
  std::vector<std::string> column_names = {"image", "label"};
  int64_t num_classes = ds->GetNumClasses();
  EXPECT_EQ(types.size(), 2);
  EXPECT_EQ(types[0].ToString(), "uint8");
  EXPECT_EQ(types[1].ToString(), "uint32");
  EXPECT_EQ(shapes.size(), 2);
  EXPECT_EQ(shapes[1].ToString(), "<>");
  EXPECT_EQ(num_classes, 2);
  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);

  EXPECT_EQ(ds->GetDatasetSize(), 2);
  EXPECT_EQ(ToDETypes(ds->GetOutputTypes()), types);
  EXPECT_EQ(ToTensorShapeVec(ds->GetOutputShapes()), shapes);
  EXPECT_EQ(ds->GetNumClasses(), 2);
  EXPECT_EQ(ds->GetColumnNames(), column_names);
}

/// Feature: LSUNDataset
/// Description: Test LSUNDataset with wrong folder path
/// Expectation: Throw exception correctly
TEST_F(MindDataTestPipeline, TestLSUNDatasetFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLSUNDatasetFail.";

  // Create a LSUN Dataset in which th folder path is invalid.
  std::shared_ptr<Dataset> ds = LSUN("", "train", {}, false, std::make_shared<RandomSampler>(false, 5));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid LSUN input, state folder path is invalid.
  EXPECT_EQ(iter, nullptr);
}

/// Feature: LSUNDataset
/// Description: Test LSUNDataset with null sampler
/// Expectation: Throw exception correctly
TEST_F(MindDataTestPipeline, TestLSUNDatasetWithNullSamplerFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLSUNDatasetWithNullSamplerFail.";

  // Create a LSUN Dataset in which th Sampler is not provided.
  std::string folder_path = datasets_root_path_ + "/testLSUN";
  std::shared_ptr<Dataset> ds = LSUN(folder_path, "train", {}, false, nullptr);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid LSUN input, sampler cannot be nullptr.
  EXPECT_EQ(iter, nullptr);
}
