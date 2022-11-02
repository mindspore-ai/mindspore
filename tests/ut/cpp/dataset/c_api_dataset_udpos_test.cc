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
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/include/dataset/datasets.h"

using namespace mindspore::dataset;

using mindspore::dataset::ShuffleMode;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

/// Feature: Test UDPOS Dataset.
/// Description: Read data from a single file.
/// Expectation: Three data in one file.
TEST_F(MindDataTestPipeline, TestUDPOSDatasetBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUDPOSDatasetBasic.";
  // Test UDPOS Dataset with single UDPOS file and many default inputs.
  // Set configuration.
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(987);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  // Create a UDPOS Dataset, with single UDPOS file.
  // Note: en-ud-tag.v2.valid.txt has 3 rows.
  // Use 2 samples.
  // Use defaults for other input parameters.
  std::string dataset_dir = datasets_root_path_ + "/testUDPOSDataset/";
  std::vector<std::string> column_names = {"word", "universal", "stanford"};
  std::shared_ptr<Dataset> ds = UDPOS(dataset_dir, "valid", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);
  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("word"), row.end());

  std::vector<std::vector<std::string>> expected_result = {
    {"From", "Abed", "Ido"}, {"Psg", "Psg", "Nine"}, {"Bus", "Psg", "Nine"}};
  uint64_t i = 0;
  while (row.size() != 0) {
    for (int j = 0; j < column_names.size(); j++) {
      auto word = row[column_names[j]];
      std::shared_ptr<Tensor> de_text;
      ASSERT_OK(Tensor::CreateFromMSTensor(word, &de_text));
      std::string_view sv;
      ASSERT_OK(de_text->GetItemAt(&sv, {{}}));
      std::string ss(sv);
      EXPECT_STREQ(ss.c_str(), expected_result[i][j].c_str());
    }
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 3);
  // Expect 3 samples.

  // Manually terminate the pipeline.
  iter->Stop();

  // Restore configuration.
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

/// Feature: Test UDPOS Dataset.
/// Description: Repeat read data.
/// Expectation: Five times the read-in data.
TEST_F(MindDataTestPipeline, TestUDPOSDatasetBasicWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUDPOSDatasetBasicWithPipeline.";
  // Test UDPOS Dataset with single UDPOS file and many default inputs.

  // Set configuration.
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(987);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  // Create two UDPOSDataset, with single UDPOS file.
  // Note: en-ud-tag.v2.test.txt has 3 rows.
  // Use 2 samples.
  // Use defaults for other input parameters.
  std::string dataset_dir = datasets_root_path_ + "/testUDPOSDataset/";
  std::shared_ptr<Dataset> ds1 = UDPOS(dataset_dir, "test", 0, ShuffleMode::kFalse);
  std::shared_ptr<Dataset> ds2 = UDPOS(dataset_dir, "test", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds1, nullptr);
  EXPECT_NE(ds2, nullptr);

  // Create two Repeat operation on ds.
  int32_t repeat_num = 2;
  ds1 = ds1->Repeat(repeat_num);
  EXPECT_NE(ds1, nullptr);
  repeat_num = 3;
  ds2 = ds2->Repeat(repeat_num);
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
  std::vector<std::string> column_names = {"word", "universal", "stanford"};
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("word"), row.end());
  std::vector<std::vector<std::string>> expected_result = {{"What", "Psg", "What"}};
  uint64_t i = 0;
  while (row.size() != 0) {
    auto word = row["word"];
    MS_LOG(INFO) << "Tensor word shape: " << word.Shape();
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  // Expect 5 samples.
  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline.
  iter->Stop();

  // Restore configuration.
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

/// Feature: Test UDPOS Dataset.
/// Description: Includes tests for shape, type, size.
/// Expectation: Correct shape, type, size.
TEST_F(MindDataTestPipeline, TestUDPOSGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUDPOSGetters.";
  // Test UDPOS Dataset with single UDPOS file and many default inputs.

  // Set configuration.
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(987);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  // Create a UDPOS Dataset, with single UDPOS file.
  // Note: en-ud-tag.v2.test.txt has 1 rows.
  // Use 2 samples.
  // Use defaults for other input parameters.
  std::string dataset_dir = datasets_root_path_ + "/testUDPOSDataset/";
  std::shared_ptr<Dataset> ds = UDPOS(dataset_dir, "train", 2, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::vector<std::string> column_names = {"word", "universal", "stanford"};

  std::vector<DataType> types = ToDETypes(ds->GetOutputTypes());
  std::vector<TensorShape> shapes = ToTensorShapeVec(ds->GetOutputShapes());

  EXPECT_EQ(types.size(), 3);
  EXPECT_EQ(types[0].ToString(), "string");
  EXPECT_EQ(types[1].ToString(), "string");
  EXPECT_EQ(types[2].ToString(), "string");
  EXPECT_EQ(shapes.size(), 3);
  EXPECT_EQ(shapes[0].ToString(), "<6>");
  EXPECT_EQ(shapes[1].ToString(), "<6>");
  EXPECT_EQ(shapes[2].ToString(), "<6>");
  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);
  EXPECT_EQ(ds->GetDatasetSize(), 2);
  EXPECT_EQ(ds->GetColumnNames(), column_names);
  // Restore configuration.
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

/// Feature: Test UDPOS Dataset.
/// Description: Test with samplers=-1.
/// Expectation: Unable to read in data.
TEST_F(MindDataTestPipeline, TestUDPOSDatasetInvalidSamplers) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUDPOSDatasetInvalidSamplers.";

  // Create a UDPOS Dataset.
  // With invalid samplers=-1.
  std::string dataset_dir = datasets_root_path_ + "/testUDPOSDataset/";
  std::shared_ptr<Dataset> ds = UDPOS(dataset_dir, "test", -1, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: UDPOS number of samples cannot be negative.
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Test UDPOS Dataset.
/// Description: Test with wrongful empty dataset_files.
/// Expectation: Unable to read in data.
TEST_F(MindDataTestPipeline, TestUDPOSDatasetInvalidFilePath) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUDPOSDatasetInvalidFilePath.";

  // Attempt to create a UDPOS Dataset.
  // With wrongful empty dataset_files input.
  std::shared_ptr<Dataset> ds = UDPOS("NotExistFile", "test", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: dataset_files is not specified.
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Test UDPOS Dataset.
/// Description: Test with non-existent dataset_files.
/// Expectation: Unable to read in data.
TEST_F(MindDataTestPipeline, TestUDPOSDatasetInvalidFileName) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUDPOSDatasetInvalidFileName.";

  // Create a UDPOS Dataset.
  // With non-existent dataset_files input.
  std::string dataset_dir = datasets_root_path_ + "/testUDPOSDataset/";
  std::shared_ptr<Dataset> ds = UDPOS(dataset_dir, "dev", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: specified dataset_files does not exist.
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Test UDPOS Dataset.
/// Description: Test with empty string dataset_files.
/// Expectation: Unable to read in data.
TEST_F(MindDataTestPipeline, TestUDPOSDatasetEmptyFilePath) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUDPOSDatasetEmptyFilePath.";

  // Create a UDPOS Dataset.
  // With empty string dataset_files input.
  std::shared_ptr<Dataset> ds = UDPOS("", "dev", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: specified dataset_files does not exist
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Test UDPOS Dataset.
/// Description: Test with invalid num_shards=0 value.
/// Expectation: Unable to read in data.
TEST_F(MindDataTestPipeline, TestUDPOSDatasetInvalidNumShards) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUDPOSDatasetInvalidNumShards.";

  // Create a UDPOS Dataset.
  // With invalid num_shards=0 value.
  std::string dataset_dir = datasets_root_path_ + "/testUDPOSDataset/";
  std::shared_ptr<Dataset> ds = UDPOS(dataset_dir, "test", 0, ShuffleMode::kFalse, 0);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Number of shards cannot be <=0.
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Test UDPOS Dataset.
/// Description: Test with invalid shard_id=-1 value.
/// Expectation: Unable to read in data.
TEST_F(MindDataTestPipeline, TestUDPOSDatasetInvalidShardId) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUDPOSDatasetInvalidShardId.";

  // Create a UDPOS Dataset.
  // With invalid shard_id=-1 value.
  std::string dataset_dir = datasets_root_path_ + "/testUDPOSDataset/";
  std::shared_ptr<Dataset> ds = UDPOS(dataset_dir, "dev", 0, ShuffleMode::kFalse, -1);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: shard_id cannot be negative.
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Test UDPOS Dataset.
/// Description: Test with invalid shard_id=2 and num_shards=2 combination.
/// Expectation: Unable to read in data.
TEST_F(MindDataTestPipeline, TestUDPOSDatasetInvalidIdAndShards) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUDPOSDatasetInvalidIdAndShards.";

  // Create a UDPOS Dataset.
  // With invalid shard_id=2 and num_shards=2 combination.
  std::string dataset_dir = datasets_root_path_ + "/testUDPOSDataset/";
  std::shared_ptr<Dataset> ds = UDPOS(dataset_dir, "dev", 0, ShuffleMode::kFalse, 2, 2);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Cannot have shard_id >= num_shards.
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Test UDPOS Dataset.
/// Description: Read all data with no shuffle, num_parallel_workers=1.
/// Expectation: Return correct data.
TEST_F(MindDataTestPipeline, TestUDPOSDatasetShuffleFalse) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUDPOSDatasetShuffleFalse.";
  // Test UDPOS Dataset with three UDPOS files and no shuffle, num_parallel_workers=1.

  // Set configuration.
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(654);
  GlobalContext::config_manager()->set_num_parallel_workers(1);

  // Create a UDPOS Dataset, with three UDPOS files, en-ud-tag.v2.valid.txt ,
  // en-ud-tag.v2.test.txt and en-ud-tag.v2.train.txt, in lexicographical order.
  // Note: en-ud-tag.v2.valid.txt has 3 rows.
  // Note: en-ud-tag.v2.test.txt has 1 rows.
  // Note: en-ud-tag.v2.train.txt has 2 rows.
  // Use default of all samples.
  std::string dataset_dir = datasets_root_path_ + "/testUDPOSDataset/";
  std::shared_ptr<Dataset> ds = UDPOS(dataset_dir, "all", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  std::vector<std::string> column_names = {"word", "universal", "stanford"};
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("word"), row.end());
  std::vector<std::vector<std::string>> expected_result = {{"From", "Abed", "Ido"},    {"Psg", "Psg", "Nine"},
                                                           {"Bus", "Psg", "Nine"}, {"What", "Psg", "What"},
                                                           {"Abed", "Psg", "Nine"},   {"...", "Psg", "---"}};

  uint64_t i = 0;
  while (row.size() != 0) {
    for (int j = 0; j < column_names.size(); j++) {
      auto word = row[column_names[j]];
      std::shared_ptr<Tensor> de_text;
      ASSERT_OK(Tensor::CreateFromMSTensor(word, &de_text));
      std::string_view sv;
      ASSERT_OK(de_text->GetItemAt(&sv, {{}}));
      std::string ss(sv);
      EXPECT_STREQ(ss.c_str(), expected_result[i][j].c_str());
    }
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  // Expect 3 + 1 + 2 = 6 samples.
  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline.
  iter->Stop();

  // Restore configuration.
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

/// Feature: Test UDPOS Dataset.
/// Description: Read all data with files shuffle, num_parallel_workers=1.
/// Expectation: Return correct data.
TEST_F(MindDataTestPipeline, TestUDPOSDatasetShuffleFilesA) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUDPOSDatasetShuffleFilesA.";
  // Test TUDPOS Dataset with files shuffle, num_parallel_workers=1.

  // Set configuration.
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(135);
  GlobalContext::config_manager()->set_num_parallel_workers(1);

  // Create a UDPOS Dataset, with three UDPOS files, en-ud-tag.v2.valid.txt ,
  // en-ud-tag.v2.test.txt and en-ud-tag.v2.train.txt, in lexicographical order.
  // Note: en-ud-tag.v2.valid.txt has 3 rows.
  // Note: en-ud-tag.v2.test.txt has 1 rows.
  // Note: en-ud-tag.v2.train.txt has 2 rows.
  // Set shuffle to files shuffle.
  std::string dataset_dir = datasets_root_path_ + "/testUDPOSDataset/";
  std::shared_ptr<Dataset> ds = UDPOS(dataset_dir, "all", 0, ShuffleMode::kFiles);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  std::vector<std::string> column_names = {"word", "universal", "stanford"};
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("word"), row.end());
  std::vector<std::vector<std::string>> expected_result = {{"Abed", "Psg", "Nine"},    {"...", "Psg", "---"},
                                                           {"What", "Psg", "What"}, {"From", "Abed", "Ido"},
                                                           {"Psg", "Psg", "Nine"},   {"Bus", "Psg", "Nine"}};

  uint64_t i = 0;
  while (row.size() != 0) {
    for (int j = 0; j < column_names.size(); j++) {
      auto word = row[column_names[j]];
      std::shared_ptr<Tensor> de_text;
      ASSERT_OK(Tensor::CreateFromMSTensor(word, &de_text));
      std::string_view sv;
      ASSERT_OK(de_text->GetItemAt(&sv, {{}}));
      std::string ss(sv);
      EXPECT_STREQ(ss.c_str(), expected_result[i][j].c_str());
    }
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  // Expect 3 + 1 + 2 = 6 samples.
  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline.
  iter->Stop();

  // Restore configuration.
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

/// Feature: Test UDPOS Dataset.
/// Description: Read all data with global shuffle, num_parallel_workers=1.
/// Expectation: Return correct data.
TEST_F(MindDataTestPipeline, TestUDPOSDatasetShuffleGlobal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUDPOSDatasetShuffleGlobal.";
  // Test UDPOS Dataset with one UDPOS file, global shuffle, num_parallel_workers=1.

  // Set configuration.
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(246);
  GlobalContext::config_manager()->set_num_parallel_workers(1);

  // Create a UDPOS Dataset, with one UDPOS files.
  // Note: en-ud-tag.v2.test.txt has 1 rows.
  // Set shuffle to global shuffle.
  std::string dataset_dir = datasets_root_path_ + "/testUDPOSDataset/";
  std::shared_ptr<Dataset> ds = UDPOS(dataset_dir, "test", 0, ShuffleMode::kGlobal);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  std::vector<std::string> column_names = {"word", "universal", "stanford"};
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("word"), row.end());
  std::vector<std::vector<std::string>> expected_result = {{"What", "Psg", "What"}};

  uint64_t i = 0;
  while (row.size() != 0) {
    for (int j = 0; j < column_names.size(); j++) {
      auto word = row[column_names[j]];
      std::shared_ptr<Tensor> de_text;
      ASSERT_OK(Tensor::CreateFromMSTensor(word, &de_text));
      std::string_view sv;
      ASSERT_OK(de_text->GetItemAt(&sv, {{}}));
      std::string ss(sv);
      EXPECT_STREQ(ss.c_str(), expected_result[i][j].c_str());
    }
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  // Expect 1 samples.
  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline.
  iter->Stop();

  // Restore configuration
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}
