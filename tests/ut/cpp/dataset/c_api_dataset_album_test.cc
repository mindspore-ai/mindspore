/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/include/datasets.h"

using namespace mindspore::dataset;
using mindspore::dataset::Tensor;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestPipeline, TestAlbumBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAlbumBasic.";

  std::string folder_path = datasets_root_path_ + "/testAlbum/images";
  std::string schema_file = datasets_root_path_ + "/testAlbum/datasetSchema.json";
  std::vector<std::string> column_names = {"image", "label", "id"};
  // Create a Album Dataset
  std::shared_ptr<Dataset> ds = Album(folder_path, schema_file, column_names);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 7);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestAlbumBasicWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAlbumBasicWithPipeline.";

  std::string folder_path = datasets_root_path_ + "/testAlbum/images";
  std::string schema_file = datasets_root_path_ + "/testAlbum/datasetSchema.json";
  std::vector<std::string> column_names = {"image", "label", "id"};

  // Create two Album Dataset
  std::shared_ptr<Dataset> ds1 = Album(folder_path, schema_file, column_names);
  std::shared_ptr<Dataset> ds2 = Album(folder_path, schema_file, column_names);
  EXPECT_NE(ds1, nullptr);
  EXPECT_NE(ds2, nullptr);

  // Create two Repeat operation on ds
  int32_t repeat_num = 2;
  ds1 = ds1->Repeat(repeat_num);
  EXPECT_NE(ds1, nullptr);
  repeat_num = 3;
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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 35);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestAlbumGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAlbumGetters.";

  std::string folder_path = datasets_root_path_ + "/testAlbum/images";
  std::string schema_file = datasets_root_path_ + "/testAlbum/datasetSchema.json";
  std::vector<std::string> column_names = {"image", "label", "id"};
  // Create a Album Dataset
  std::shared_ptr<Dataset> ds = Album(folder_path, schema_file, column_names);
  EXPECT_NE(ds, nullptr);

  int64_t num_classes = ds->GetNumClasses();
  EXPECT_EQ(num_classes, -1);
  int64_t num_samples = ds->GetDatasetSize();
  EXPECT_EQ(num_samples, 7);

  int64_t batch_size = ds->GetBatchSize();
  EXPECT_EQ(batch_size, 1);
  int64_t repeat_count = ds->GetRepeatCount();
  EXPECT_EQ(repeat_count, 1);
  EXPECT_EQ(ds->GetColumnNames(), column_names);

  // Test get dataset size with num_samples > files in dataset 
  auto sampler = std::make_shared<SequentialSampler>(0, 12);
  std::shared_ptr<Dataset> ds2 = Album(folder_path, schema_file, column_names, false, sampler);
  num_samples = ds->GetDatasetSize();
  EXPECT_EQ(num_samples, 7);
}

TEST_F(MindDataTestPipeline, TestAlbumDecode) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAlbumDecode.";
  std::string folder_path = datasets_root_path_ + "/testAlbum/images";
  std::string schema_file = datasets_root_path_ + "/testAlbum/datasetSchema.json";
  std::vector<std::string> column_names = {"image", "label", "id"};
  // Create a Album Dataset
  std::shared_ptr<Dataset> ds = Album(folder_path, schema_file, column_names, true);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    auto shape = image.Shape();
    MS_LOG(INFO) << "Tensor image shape size: " << shape.size();
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    EXPECT_GT(shape.size(), 1);  // Verify decode=true took effect
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 7);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestAlbumNumSamplers) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAlbumNumSamplers.";

  std::string folder_path = datasets_root_path_ + "/testAlbum/images";
  std::string schema_file = datasets_root_path_ + "/testAlbum/datasetSchema.json";
  std::vector<std::string> column_names = {"image", "label", "id"};
  // Create a Album Dataset
  std::shared_ptr<Dataset> ds = Album(folder_path, schema_file, column_names, true, std::make_shared<SequentialSampler>(0, 1));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestAlbumError) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAlbumError.";
  std::string folder_path = datasets_root_path_ + "/testAlbum/ima";
  std::string schema_file = datasets_root_path_ + "/testAlbum/datasetSchema.json";
  std::vector<std::string> column_names = {"image", "label", "id"};
  // Create an Album Dataset
  std::shared_ptr<Dataset> ds = Album(folder_path, schema_file, column_names, true, std::make_shared<SequentialSampler>(0, 1));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Album input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestAlbumWithNullSamplerError) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAlbumWithNullSamplerError.";
  std::string folder_path = datasets_root_path_ + "/testAlbum/images";
  std::string schema_file = datasets_root_path_ + "/testAlbum/datasetSchema.json";
  std::vector<std::string> column_names = {"image", "label", "id"};
  // Create an Album Dataset
  std::shared_ptr<Dataset> ds = Album(folder_path, schema_file, column_names, true, nullptr);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Album input, sampler cannot be nullptr
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestAlbumDuplicateColumnNameError) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAlbumDuplicateColumnNameError.";
  std::string folder_path = datasets_root_path_ + "/testAlbum/images";
  std::string schema_file = datasets_root_path_ + "/testAlbum/datasetSchema.json";
  std::vector<std::string> column_names = {"image", "image", "id"};
  // Create an Album Dataset
  std::shared_ptr<Dataset> ds = Album(folder_path, schema_file, column_names, true);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Album input, duplicate column names
  EXPECT_EQ(iter, nullptr);
}
