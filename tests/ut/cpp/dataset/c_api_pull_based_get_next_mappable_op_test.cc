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
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/include/dataset/datasets.h"
#include "minddata/dataset/include/dataset/vision.h"

namespace common = mindspore::common;

using namespace mindspore::dataset;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on Album
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableAlbum) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableAlbum.";

  std::string folder_path = datasets_root_path_ + "/testAlbum/images";
  std::string schema_file = datasets_root_path_ + "/testAlbum/datasetSchema.json";
  std::vector<std::string> column_names = {"label", "image"};
  std::shared_ptr<Dataset> ds = Album(folder_path, schema_file, column_names);
  EXPECT_NE(ds, nullptr);

  std::vector<mindspore::MSTensor> row;
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  ASSERT_OK(iter->GetNextRow(&row));
  auto count = 0;
  while (row.size() > 0) {
    ++count;
    auto label = row[0];
    auto image = row[1];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Tensor level shape: " << label.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }
  EXPECT_EQ(count, 7);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on ImageFolder
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableImageFolder) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableImageFolder.";
  // Test get dataset size in distributed scenario when num_per_shard is more than num_samples

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds =
    ImageFolder(folder_path, true, std::make_shared<DistributedSampler>(4, 0, ShuffleMode::kFalse, 10));
  EXPECT_NE(ds, nullptr);

  // num_per_shard is equal to 44/4 = 11 which is more than num_samples = 10, so the output is 10
  EXPECT_EQ(ds->GetDatasetSize(), 10);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_EQ(row.size(), 2);
  MS_LOG(INFO) << "row.size() " << row.size();

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row[0];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }
  // The value of i should be equal to the result of get dataset size
  EXPECT_EQ(i, 10);
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on IMDB
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableIMDB) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableIMDB.";

  std::string dataset_path = datasets_root_path_ + "/testIMDBDataset";
  std::string usage = "all";  // 'train', 'test', 'all'

  // Create a IMDB Dataset
  std::shared_ptr<Dataset> ds = IMDB(dataset_path, usage);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  // order: text, label
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto text = row[0];
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    auto label = row[1];
    MS_LOG(INFO) << "Tensor label shape: " << label.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 8);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on Manifest
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableManifest) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableManifest.";

  std::string file_path = datasets_root_path_ + "/testManifestData/cpp.json";
  // Create a Manifest Dataset
  std::shared_ptr<Dataset> ds = Manifest(file_path);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  // order: image, label
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_EQ(row.size(), 2);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row[0];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    auto label = row[1];
    MS_LOG(INFO) << "Tensor label shape: " << label.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on MindRecord
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableMinddata) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableMinddata with string file pattern.";
  // Create a MindData Dataset
  // Pass one mindrecord shard file to parse dataset info, and search for other mindrecord files with same dataset info,
  // thus all records in imagenet.mindrecord0 ~ imagenet.mindrecord3 will be read
  std::string file_path = datasets_root_path_ + "/../mindrecord/testMindDataSet/testImageNetData/imagenet.mindrecord0";
  std::vector<std::string> column_names = {"file_name", "label", "data"};
  std::shared_ptr<Dataset> ds = MindData(file_path, column_names);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_EQ(row.size(), 3);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row[0];
    TEST_MS_LOG_MSTENSOR(INFO, "Tensor image file name: ", image);
    auto label = row[1];
    MS_LOG(INFO) << "Tensor label shape: " << label.Shape();
    auto data = row[2];
    MS_LOG(INFO) << "Tensor data shape: " << data.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  // Each *.mindrecord file has 5 rows, so there are 20 rows in total(imagenet.mindrecord0 ~ imagenet.mindrecord3)
  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on RandomData
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableRandomData) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableRandomData.";

  // Create a RandomDataset
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("image", mindspore::DataType::kNumberTypeUInt8, {2}));
  ASSERT_OK(schema->add_column("label", mindspore::DataType::kNumberTypeUInt8, {1}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  // order: image, label
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_EQ(row.size(), 2);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row[0];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    auto label = row[1];
    MS_LOG(INFO) << "Tensor label shape: " << label.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }
  EXPECT_EQ(i, 50);

  // Manually terminate the pipeline
  iter->Stop();
}
