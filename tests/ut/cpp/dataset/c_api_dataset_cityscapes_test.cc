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
#include <fstream>
#include <iostream>

#include "common/common.h"
#include "minddata/dataset/include/dataset/datasets.h"

using namespace mindspore::dataset;
using mindspore::dataset::Tensor;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

/// Feature: CityscapesDataset
/// Description: Basic test of CityscapesDataset
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestCityscapesBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCityscapesBasic.";

  std::string dataset_path = datasets_root_path_ + "/testCityscapesData/cityscapes";
  std::string usage = "train";        // quality_mode=fine 'train', 'test', 'val'  else 'train', 'train_extra', 'val'
  std::string quality_mode = "fine";  // fine coarse
  std::string task = "color";         // instance semantic polygon color

  // Create a Cityscapes Dataset
  std::shared_ptr<Dataset> ds = Cityscapes(dataset_path, usage, quality_mode, task);
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
    auto task = row["task"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: CityscapesDataset
/// Description: Test CityscapesDataset in pipeline mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestCityscapesBasicWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCityscapesBasicWithPipeline.";

  std::string dataset_path = datasets_root_path_ + "/testCityscapesData/cityscapes";
  std::string usage = "train";        // quality_mode=fine 'train', 'test', 'val'  else 'train', 'train_extra', 'val'
  std::string quality_mode = "fine";  // fine coarse

  // Create two Cityscapes Dataset
  std::shared_ptr<Dataset> ds1 =
    Cityscapes(dataset_path, usage, quality_mode, "color", false, std::make_shared<RandomSampler>(false, 2));
  std::shared_ptr<Dataset> ds2 =
    Cityscapes(dataset_path, usage, quality_mode, "color", false, std::make_shared<RandomSampler>(false, 3));
  EXPECT_NE(ds1, nullptr);
  EXPECT_NE(ds2, nullptr);

  // Create two Repeat operation on ds
  int32_t repeat_num = 3;
  ds1 = ds1->Repeat(repeat_num);
  EXPECT_NE(ds1, nullptr);
  repeat_num = 2;
  ds2 = ds2->Repeat(repeat_num);
  EXPECT_NE(ds2, nullptr);

  // Create two Project operation on ds
  std::vector<std::string> column_project = {"image"};
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

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 12);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: CityscapesDataset
/// Description: Test CityscapesDataset Getters method
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestCityscapesGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCityscapesGetters.";

  std::string dataset_path = datasets_root_path_ + "/testCityscapesData/cityscapes";
  std::string usage = "train";        // quality_mode=fine 'train', 'test', 'val'  else 'train', 'train_extra', 'val'
  std::string quality_mode = "fine";  // fine coarse
  std::string task = "color";         // instance semantic polygon color

  // Create a Cityscapes Dataset
  std::shared_ptr<Dataset> ds1 =
    Cityscapes(dataset_path, usage, quality_mode, task, false, std::make_shared<RandomSampler>(false, 4));
  std::shared_ptr<Dataset> ds2 = Cityscapes(dataset_path, usage, quality_mode, task);
  std::vector<std::string> column_names = {"image", "task"};

  EXPECT_NE(ds1, nullptr);
  EXPECT_EQ(ds1->GetDatasetSize(), 4);
  EXPECT_EQ(ds1->GetColumnNames(), column_names);
  EXPECT_EQ(ds1->GetBatchSize(), 1);

  EXPECT_NE(ds2, nullptr);
  EXPECT_EQ(ds2->GetDatasetSize(), 5);
  EXPECT_EQ(ds2->GetColumnNames(), column_names);
  EXPECT_EQ(ds2->GetBatchSize(), 1);
}

/// Feature: CityscapesDataset
/// Description: Test CityscapesDataset in where the dataset comes from .json file
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestCityscapesTaskJson) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCityscapesTaskJson.";

  std::string dataset_path = datasets_root_path_ + "/testCityscapesData/cityscapes/testTaskJson";
  std::string usage = "train";        // quality_mode=fine 'train', 'test', 'val'  else 'train', 'train_extra', 'val'
  std::string quality_mode = "fine";  // fine coarse
  std::string task = "polygon";       // instance semantic polygon color

  std::shared_ptr<Dataset> ds = Cityscapes(dataset_path, usage, quality_mode, task);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::string json_file_path = dataset_path + "/gtFine/train/aa/aa_000000_gtFine_polygons.json";
  std::ifstream file_handle(json_file_path);
  std::string contents((std::istreambuf_iterator<char>(file_handle)), std::istreambuf_iterator<char>());
  nlohmann::json contents_js = nlohmann::json::parse(contents);
  std::shared_ptr<Tensor> t_expect_item;
  Tensor::CreateScalar(contents_js.dump(), &t_expect_item);
  file_handle.close();

  mindspore::MSTensor expect_item = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(t_expect_item));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    auto task = row["task"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Tensor task shape: " << task.Shape();

    EXPECT_MSTENSOR_EQ(task, expect_item);
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: CityscapesDataset
/// Description: Test CityscapesDataset with Decode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestCityscapesDecode) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCityscapesDecode.";

  std::string dataset_path = datasets_root_path_ + "/testCityscapesData/cityscapes";
  std::string usage = "train";        // quality_mode=fine 'train', 'test', 'val' else 'train', 'train_extra', 'val'
  std::string quality_mode = "fine";  // fine coarse
  std::string task = "color";         // instance semantic polygon color

  // Create a Cityscapes Dataset
  std::shared_ptr<Dataset> ds =
    Cityscapes(dataset_path, usage, quality_mode, task, true, std::make_shared<RandomSampler>());
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
    auto task = row["task"];

    EXPECT_EQ(image.Shape().size(), 3);
    EXPECT_EQ(task.Shape().size(), 3);

    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: CityscapesDataset
/// Description: Test CityscapesDataset using sampler
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestCityscapesNumSamplers) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCityscapesNumSamplers.";

  std::string dataset_path = datasets_root_path_ + "/testCityscapesData/cityscapes";
  std::string usage = "train";        // quality_mode=fine 'train', 'test', 'val'  else 'train', 'train_extra', 'val'
  std::string quality_mode = "fine";  // fine coarse
  std::string task = "color";         // instance semantic polygon color

  // Create a Cityscapes Dataset
  std::shared_ptr<Dataset> ds =
    Cityscapes(dataset_path, usage, quality_mode, task, true, std::make_shared<RandomSampler>(false, 5));
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
    auto task = row["task"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Tensor task shape: " << task.Shape();

    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: CityscapesDataset
/// Description: Test CityscapesDataset with invalid inputs
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestCityscapesError) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCityscapesError.";

  std::string dataset_path = datasets_root_path_ + "/testCityscapesData/cityscapes";
  std::string usage = "train";        // quality_mode=fine 'train', 'test', 'val'  else 'train', 'train_extra', 'val'
  std::string quality_mode = "fine";  // fine coarse
  std::string task = "color";      // instance semantic polygon color

  // Create a Cityscapes Dataset with non-existing dataset dir
  std::shared_ptr<Dataset> ds0 = Cityscapes("NotExistDir", usage, quality_mode, task);
  EXPECT_NE(ds0, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter0 = ds0->CreateIterator();
  // Expect failure: invalid Cityscapes input
  EXPECT_EQ(iter0, nullptr);

  // Create a Cityscapes Dataset with err task
  std::shared_ptr<Dataset> ds1 = Cityscapes(dataset_path, usage, quality_mode, "task");
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid Cityscapes input
  EXPECT_EQ(iter1, nullptr);

  // Create a Cityscapes Dataset with err quality_mode
  std::shared_ptr<Dataset> ds2 = Cityscapes(dataset_path, usage, "quality_mode", task);
  EXPECT_NE(ds2, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid Cityscapes input
  EXPECT_EQ(iter2, nullptr);

  // Create a Cityscapes Dataset with err usage
  std::shared_ptr<Dataset> ds3 = Cityscapes(dataset_path, "usage", quality_mode, task);
  EXPECT_NE(ds3, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid Cityscapes input
  EXPECT_EQ(iter3, nullptr);
}

/// Feature: CityscapesDataset
/// Description: Test CityscapesDataset using nullptr for sampler
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestCityscapesWithNullSamplerError) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCityscapesWithNullSamplerError.";

  std::string dataset_path = datasets_root_path_ + "/testCityscapesData/cityscapes";
  std::string usage = "train";        // quality_mode=fine 'train', 'test', 'val'  else 'train', 'train_extra', 'val'
  std::string quality_mode = "fine";  // fine coarse
  std::string task = "color";      // instance semantic polygon color

  // Create a Cityscapes Dataset
  std::shared_ptr<Dataset> ds = Cityscapes(dataset_path, usage, quality_mode, task, false, nullptr);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Cityscapes input, sampler cannot be nullptr
  EXPECT_EQ(iter, nullptr);
}