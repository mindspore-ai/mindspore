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

/// Feature: OmniglotDataset
/// Description: Test OmniglotDataset using background dataset
/// Expectation: Get correct Omniglot dataset
TEST_F(MindDataTestPipeline, TestOmniglotBackgroundDataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestOmniglotBackgroundDataset.";

  // Create a Omniglot Dataset.
  std::string folder_path = datasets_root_path_ + "/testOmniglot";
  std::shared_ptr<Dataset> ds = Omniglot(folder_path, true, false, std::make_shared<RandomSampler>(false, 5));
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

/// Feature: OmniglotDataset
/// Description: Test OmniglotDataset using evaluation dataset
/// Expectation: Get correct Omniglot dataset
TEST_F(MindDataTestPipeline, TestOmniglotEvaluationDataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestOmniglotEvaluationDataset.";

  // Create a Omniglot Dataset.
  std::string folder_path = datasets_root_path_ + "/testOmniglot";
  std::shared_ptr<Dataset> ds = Omniglot(folder_path, false, false, std::make_shared<RandomSampler>(false, 5));
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

/// Feature: OmniglotDataset
/// Description: Test OmniglotDataset using background dataset with pipeline mode
/// Expectation: Get correct Omniglot dataset
TEST_F(MindDataTestPipeline, TestOmniglotBackgroundDatasetWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestOmniglotBackgroundDatasetWithPipeline.";

  // Create two Omniglot Dataset.
  std::string folder_path = datasets_root_path_ + "/testOmniglot";
  std::shared_ptr<Dataset> ds1 = Omniglot(folder_path, true, false, std::make_shared<RandomSampler>(false, 5));
  std::shared_ptr<Dataset> ds2 = Omniglot(folder_path, true, false, std::make_shared<RandomSampler>(false, 5));
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

  EXPECT_EQ(i, 8);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: OmniglotDataset
/// Description: Test OmniglotDataset GetDatasetSize with background dataset
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestOmniglotBackgroundGetDatasetSize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetOmniglotBackgroundSize.";

  // Create a Omniglot Dataset.
  std::string folder_path = datasets_root_path_ + "/testOmniglot";
  std::shared_ptr<Dataset> ds = Omniglot(folder_path, true, false);
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 4);
}

/// Feature: OmniglotDataset
/// Description: Test OmniglotDataset GetDatasetSize with evaluation dataset
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestOmniglotEvaluationGetDatasetSize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetOmniglotEvaluationDatasetSize.";

  // Create a Omniglot Dataset.
  std::string folder_path = datasets_root_path_ + "/testOmniglot";
  std::shared_ptr<Dataset> ds = Omniglot(folder_path, false, false);
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 4);
}

/// Feature: OmniglotDataset
/// Description: Test OmniglotDataset Getters method with background dataset
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestOmniglotBackgroundDatasetGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestOmniglotBackgroundDatasetGetters.";

  // Create a Omniglot Dataset.
  std::string folder_path = datasets_root_path_ + "/testOmniglot";
  std::shared_ptr<Dataset> ds = Omniglot(folder_path, true, true);
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 4);
  std::vector<DataType> types = ToDETypes(ds->GetOutputTypes());
  std::vector<TensorShape> shapes = ToTensorShapeVec(ds->GetOutputShapes());
  std::vector<std::string> column_names = {"image", "label"};
  int64_t num_classes = ds->GetNumClasses();
  EXPECT_EQ(types.size(), 2);
  EXPECT_EQ(types[0].ToString(), "uint8");
  EXPECT_EQ(types[1].ToString(), "int32");
  EXPECT_EQ(shapes.size(), 2);
  EXPECT_EQ(shapes[1].ToString(), "<>");
  EXPECT_EQ(num_classes, 2);
  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);

  EXPECT_EQ(ds->GetDatasetSize(), 4);
  EXPECT_EQ(ToDETypes(ds->GetOutputTypes()), types);
  EXPECT_EQ(ToTensorShapeVec(ds->GetOutputShapes()), shapes);
  EXPECT_EQ(ds->GetNumClasses(), 2);

  EXPECT_EQ(ds->GetColumnNames(), column_names);
  EXPECT_EQ(ds->GetDatasetSize(), 4);
  EXPECT_EQ(ToDETypes(ds->GetOutputTypes()), types);
  EXPECT_EQ(ToTensorShapeVec(ds->GetOutputShapes()), shapes);
  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);
  EXPECT_EQ(ds->GetNumClasses(), 2);
  EXPECT_EQ(ds->GetDatasetSize(), 4);
}

/// Feature: OmniglotDataset
/// Description: Test OmniglotDataset Getters method with evaluation dataset
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestOmniglotEvaluationDatasetGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestOmniglotTestDatasetGetters.";

  // Create a Omniglot Test Dataset.
  std::string folder_path = datasets_root_path_ + "/testOmniglot";
  std::shared_ptr<Dataset> ds = Omniglot(folder_path, false, true);
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 4);
  std::vector<DataType> types = ToDETypes(ds->GetOutputTypes());
  std::vector<TensorShape> shapes = ToTensorShapeVec(ds->GetOutputShapes());
  std::vector<std::string> column_names = {"image", "label"};
  int64_t num_classes = ds->GetNumClasses();
  EXPECT_EQ(types.size(), 2);
  EXPECT_EQ(types[0].ToString(), "uint8");
  EXPECT_EQ(types[1].ToString(), "int32");
  EXPECT_EQ(shapes.size(), 2);
  EXPECT_EQ(shapes[1].ToString(), "<>");
  EXPECT_EQ(num_classes, 2);
  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);

  EXPECT_EQ(ToDETypes(ds->GetOutputTypes()), types);
  EXPECT_EQ(ToTensorShapeVec(ds->GetOutputShapes()), shapes);
  EXPECT_EQ(ds->GetNumClasses(), 2);

  EXPECT_EQ(ds->GetColumnNames(), column_names);
  EXPECT_EQ(ToDETypes(ds->GetOutputTypes()), types);
  EXPECT_EQ(ToTensorShapeVec(ds->GetOutputShapes()), shapes);
  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);
  EXPECT_EQ(ds->GetNumClasses(), 2);
}

/// Feature: OmniglotDataset
/// Description: Test OmniglotDataset with invalid num_images
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestOmniglotDatasetFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestOmniglotDatasetFail.";

  // Create a Omniglot Dataset.
  std::shared_ptr<Dataset> ds = Omniglot("", true, false, std::make_shared<RandomSampler>(false, 5));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Omniglot input.
  EXPECT_EQ(iter, nullptr);
}

/// Feature: OmniglotDataset
/// Description: Test OmniglotDataset with null sampler
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestOmniglotDatasetWithNullSamplerFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestOmniglotDatasetWithNullSamplerFail.";

  // Create a Omniglot Dataset.
  std::string folder_path = datasets_root_path_ + "/testOmniglot";
  std::shared_ptr<Dataset> ds = Omniglot(folder_path, true, false, nullptr);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Omniglot input, sampler cannot be nullptr
  EXPECT_EQ(iter, nullptr);
}
