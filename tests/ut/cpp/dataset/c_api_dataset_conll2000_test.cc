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

/// Feature: CoNLL2000Dataset
/// Description: Test CoNLL2000Dataset basic usage
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestCoNLL2000DatasetBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCoNLL2000DatasetBasic.";
  // Test CoNLL2000 Dataset with single text file and many default inputs.
  
  // Set configuration.
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(987);
  GlobalContext::config_manager()->set_num_parallel_workers(2);

  // Create a CoNLL2000Dataset, with single text file.
  // Note: valid.txt has 3 rows.
  // Use 2 samples.
  // Use defaults for other input parameters.
  std::string dataset_dir = datasets_root_path_ + "/testCoNLL2000Dataset";
  std::vector<std::string> column_names = {"word", "pos_tag", "chunk_tag"};
  std::shared_ptr<Dataset> ds = CoNLL2000(dataset_dir, "train", 0, ShuffleMode::kFalse);
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
    {"Challenge", "NNP", "O"}, {"Her", "PP$", "B-NP"}, {"To", "TO", "I-VP"}};
  uint64_t i = 0;
  while (row.size() != 0) {
    for (int j = 0; j < column_names.size(); j++) {
      auto word = row[column_names[j]];
      std::shared_ptr<Tensor> de_word;
      ASSERT_OK(Tensor::CreateFromMSTensor(word, &de_word));
      std::string_view sv;
      ASSERT_OK(de_word->GetItemAt(&sv, {{}}));
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

/// Feature: CoNLL2000Dataset
/// Description: Test CoNLL2000Dataset in pipeline mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestCoNLL2000DatasetBasicWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCoNLL2000DatasetBasicWithPipeline.";
  // Test CoNLL2000 Dataset with single text file and many default inputs.

  // Set configuration.
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(987);
  GlobalContext::config_manager()->set_num_parallel_workers(2);

  // Create two CoNLL2000Dataset, with single text file.
  // Note: test.txt has 3 rows.
  // Use 2 samples.
  // Use defaults for other input parameters.
  std::string dataset_dir = datasets_root_path_ + "/testCoNLL2000Dataset";
  std::shared_ptr<Dataset> ds1 = CoNLL2000(dataset_dir, "test", 0, ShuffleMode::kFalse);
  std::shared_ptr<Dataset> ds2 = CoNLL2000(dataset_dir, "test", 0, ShuffleMode::kFalse);
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
  std::vector<std::string> column_names = {"word", "pos_tag", "chunk_tag"};
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("word"), row.end());
  std::vector<std::vector<std::string>> expected_result = {{"He", "PBP", "B-NP"}, {"The", "DT", "B-NP"}};
  uint64_t i = 0;
  while (row.size() != 0) {
    auto word = row["word"];
    MS_LOG(INFO) << "Tensor word shape: " << word.Shape();
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  // Expect 10 samples.
  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline.
  iter->Stop();

  // Restore configuration.
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

/// Feature: CoNLL2000Dataset
/// Description: Test CoNLL2000Dataset Getters method
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestCoNLL2000Getters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCoNLL2000Getters.";
  // Test CoNLL2000 Dataset with single text file and many default inputs.

  // Set configuration.
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(987);
  GlobalContext::config_manager()->set_num_parallel_workers(2);

  // Create a CoNLL2000 Dataset, with single text file.
  // Note: test.txt has 1 rows.
  // Use 2 samples.
  // Use defaults for other input parameters.
  std::string dataset_dir = datasets_root_path_ + "/testCoNLL2000Dataset";
  std::shared_ptr<Dataset> ds = CoNLL2000(dataset_dir, "test", 2, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::vector<std::string> column_names = {"word", "pos_tag", "chunk_tag"};
  EXPECT_EQ(ds->GetDatasetSize(), 2);
  EXPECT_EQ(ds->GetColumnNames(), column_names);

  std::shared_ptr<Dataset> ds1 = CoNLL2000(dataset_dir, "", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds1, nullptr);

  EXPECT_EQ(ds1->GetDatasetSize(), 30);
  // Restore configuration.
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

/// Feature: CoNLL2000Dataset
/// Description: Test CoNLL2000Dataset with invalid samplers
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestCoNLL2000DatasetFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCoNLL2000DatasetFail1.";

  // Create a CoNLL2000Dataset.
  // with invalid samplers=-1.
  std::string dataset_dir = datasets_root_path_ + "/testCoNLL2000Dataset";
  std::shared_ptr<Dataset> ds = CoNLL2000(dataset_dir, "test", -1, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: CoNLL2000 number of samples cannot be negative.
  EXPECT_EQ(iter, nullptr);
}

/// Feature: CoNLL2000Dataset
/// Description: Test CoNLL2000Dataset with empty dataset_files
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestCoNLL2000DatasetFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCoNLL2000DatasetFail2.";

  // Attempt to create a CoNLL2000 Dataset.
  // with wrongful empty dataset_files input.
  std::shared_ptr<Dataset> ds = CoNLL2000("NotExistFile", "test", 2, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: dataset_files is not specified.
  EXPECT_EQ(iter, nullptr);
}

/// Feature: CoNLL2000Dataset
/// Description: Test CoNLL2000Dataset with non-existent dataset_files
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestCoNLL2000DatasetFail3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCoNLL2000DatasetFail3.";

  // Create a CoNLL2000 Dataset.
  // with non-existent dataset_files input.
  std::string dataset_dir = datasets_root_path_ + "/testCoNLL2000Dataset";
  std::shared_ptr<Dataset> ds = CoNLL2000(dataset_dir, "dev", 2, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: specified dataset_files does not exist.
  EXPECT_EQ(iter, nullptr);
}

/// Feature: CoNLL2000Dataset
/// Description: Test CoNLL2000Dataset with empty string as an input to dataset_files
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestCoNLL2000DatasetFail4) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCoNLL2000DatasetFail4.";

  // Create a CoNLL2000Dataset.
  // with empty string dataset_files input.
  std::shared_ptr<Dataset> ds = CoNLL2000("", "test", 2, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  std::cout << iter;
  // Expect failure: specified dataset_files does not exist.
  EXPECT_EQ(iter, nullptr);
}

/// Feature: CoNLL2000Dataset
/// Description: Test CoNLL2000Dataset with invalid num_shards value
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestCoNLL2000DatasetFail5) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCoNLL2000DatasetFail5.";

  // Create a CoNLL2000 Dataset.
  // with invalid num_shards=0 value.
  std::string dataset_dir = datasets_root_path_ + "/testCoNLL2000Dataset";
  std::shared_ptr<Dataset> ds = CoNLL2000(dataset_dir, "test", 2, ShuffleMode::kFalse, 0);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Number of shards cannot be <=0.
  EXPECT_EQ(iter, nullptr);
}

/// Feature: CoNLL2000Dataset
/// Description: Test CoNLL2000Dataset with invalid shard_id value
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestCoNLL2000DatasetFail6) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCoNLL2000DatasetFail6.";

  // Create a CoNLL2000Dataset.
  // with invalid shard_id=-1 value.
  std::string dataset_dir = datasets_root_path_ + "/testCoNLL2000Dataset";
  std::shared_ptr<Dataset> ds = CoNLL2000(dataset_dir, "test", 2, ShuffleMode::kFalse, -1);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: shard_id cannot be negative.
  EXPECT_EQ(iter, nullptr);
}

/// Feature: CoNLL2000Dataset
/// Description: Test CoNLL2000Dataset with invalid shard_id and num_shards values
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestCoNLL2000DatasetFail7) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCoNLL2000DatasetFail7.";

  // Create a CoNLL2000 Dataset.
  // with invalid shard_id=2 and num_shards=2 combination.
  std::string dataset_dir = datasets_root_path_ + "/testCoNLL2000Dataset";
  std::shared_ptr<Dataset> ds = CoNLL2000(dataset_dir, "test", 2, ShuffleMode::kFalse, 2, 2);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Cannot have shard_id >= num_shards.
  EXPECT_EQ(iter, nullptr);
}

/// Feature: CoNLL2000Dataset
/// Description: Test CoNLL2000Dataset with no shuffle
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestCoNLL2000DatasetShuffleFalse) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCoNLL2000DatasetShuffleFalse.";
  // Test CoNLL2000 Dataset with two text files and no shuffle, num_parallel_workers=4.

  // Set configuration.
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(654);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  // Create a CoNLL2000 Dataset, with two text files,  test.txt and train.txt, in lexicographical order.
  // Note: test.txt has 2 rows.
  // Note: train.txt has 3 rows.
  // Use default of all samples.
  std::string dataset_dir = datasets_root_path_ + "/testCoNLL2000Dataset";
  std::shared_ptr<Dataset> ds = CoNLL2000(dataset_dir, "all", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  std::vector<std::string> column_names = {"word", "pos_tag", "chunk_tag"};
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("word"), row.end());
  std::vector<std::vector<std::string>> expected_result = {{"He", "PBP", "B-NP"},
                                                           {"Challenge", "NNP", "O"},
                                                           {"The", "DT", "B-NP"},
                                                           {"Her", "PP$", "B-NP"},
                                                           {"To", "TO", "I-VP"}};

  uint64_t i = 0;
  while (row.size() != 0) {
    for (int j = 0; j < column_names.size(); j++) {
      auto word = row[column_names[j]];
      std::shared_ptr<Tensor> de_word;
      ASSERT_OK(Tensor::CreateFromMSTensor(word, &de_word));
      std::string_view sv;
      ASSERT_OK(de_word->GetItemAt(&sv, {{}}));
      std::string ss(sv);
      EXPECT_STREQ(ss.c_str(), expected_result[i][j].c_str());
    }
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline.
  iter->Stop();

  // Restore configuration.
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

/// Feature: CoNLL2000Dataset
/// Description: Test CoNLL2000Dataset with ShuffleMode::kFalse
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestCoNLL2000DatasetShuffleFilesA) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCoNLL2000DatasetShuffleFilesA.";
  // Test CoNLL2000 Dataset with files shuffle, num_parallel_workers=4.

  // Set configuration.
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(135);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  // Create a CoNLL2000 Dataset, with two text files,test.txt and train.txt, in lexicographical order.
  // Note: test.txt has 2 rows.
  // Note: train.txt has 3 rows.
  // Set shuffle to files shuffle.
  std::string dataset_dir = datasets_root_path_ + "/testCoNLL2000Dataset";
  std::shared_ptr<Dataset> ds = CoNLL2000(dataset_dir, "all", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  std::vector<std::string> column_names = {"word", "pos_tag", "chunk_tag"};
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("word"), row.end());
  std::vector<std::vector<std::string>> expected_result = {{"He", "PBP", "B-NP"},
                                                           {"Challenge", "NNP", "O"},
                                                           {"The", "DT", "B-NP"},
                                                           {"Her", "PP$", "B-NP"},
                                                           {"To", "TO", "I-VP"}};
  uint64_t i = 0;
  while (row.size() != 0) {
    for (int j = 0; j < column_names.size(); j++) {
      auto word = row[column_names[j]];
      std::shared_ptr<Tensor> de_word;
      ASSERT_OK(Tensor::CreateFromMSTensor(word, &de_word));
      std::string_view sv;
      ASSERT_OK(de_word->GetItemAt(&sv, {{}}));
      std::string ss(sv);
      EXPECT_STREQ(ss.c_str(), expected_result[i][j].c_str());
    }
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  // Expect 3 + 1 + 2 = 6 samples.
  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline.
  iter->Stop();

  // Restore configuration.
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

/// Feature: CoNLL2000Dataset
/// Description: Test CoNLL2000Dataset with ShuffleMode::kFalse
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestCoNLL2000DatasetShuffleFilesB) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCoNLL2000DatasetShuffleFilesB.";
  // Test CoNLL2000 Dataset with files shuffle, num_parallel_workers=4.

  // Set configuration.
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(135);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  // Create a CoNLL2000 Dataset, with two text files test.txt and train.txt, in lexicographical order.
  // Note: test.txt has 2 rows.
  // Note: train.txt has  3 rows.
  // Set shuffle to files shuffle.
  std::string dataset_dir = datasets_root_path_ + "/testCoNLL2000Dataset";
  std::shared_ptr<Dataset> ds = CoNLL2000(dataset_dir, "all", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  std::vector<std::string> column_names = {"word", "pos_tag", "chunk_tag"};
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("word"), row.end());
  std::vector<std::vector<std::string>> expected_result = {{"He", "PBP", "B-NP"},
                                                           {"Challenge", "NNP", "O"},
                                                           {"The", "DT", "B-NP"},
                                                           {"Her", "PP$", "B-NP"},
                                                           {"To", "TO", "I-VP"}};
  uint64_t i = 0;
  while (row.size() != 0) {
    for (int j = 0; j < column_names.size(); j++) {
      auto word = row[column_names[j]];
      std::shared_ptr<Tensor> de_word;
      ASSERT_OK(Tensor::CreateFromMSTensor(word, &de_word));
      std::string_view sv;
      ASSERT_OK(de_word->GetItemAt(&sv, {{}}));
      std::string ss(sv);
      EXPECT_STREQ(ss.c_str(), expected_result[i][j].c_str());
    }
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  // Expect 3 + 1 + 2 = 6 samples.
  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline.
  iter->Stop();

  // Restore configuration.
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

/// Feature: CoNLL2000Dataset
/// Description: Test CoNLL2000Dataset with ShuffleMode::kFalse and with one text file only
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestCoNLL2000DatasetShuffleGlobal1A) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCoNLL2000DatasetShuffleGlobalA.";
  // Test CoNLL2000 Dataset with 1 text file, global shuffle, num_parallel_workers=4.

  // Set configuration.
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(246);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  // Create a CoNLL2000 Dataset, with one text files.
  // Note: test.txt has 2 rows.
  // Set shuffle to global shuffle.
  std::string dataset_dir = datasets_root_path_ + "/testCoNLL2000Dataset";
  std::shared_ptr<Dataset> ds = CoNLL2000(dataset_dir, "test", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  std::vector<std::string> column_names = {"word", "pos_tag", "chunk_tag"};
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("word"), row.end());
  std::vector<std::vector<std::string>> expected_result = {{"He", "PBP", "B-NP"}, {"The", "DT", "B-NP"}};

  uint64_t i = 0;
  while (row.size() != 0) {
    for (int j = 0; j < column_names.size(); j++) {
      auto word = row[column_names[j]];
      std::shared_ptr<Tensor> de_word;
      ASSERT_OK(Tensor::CreateFromMSTensor(word, &de_word));
      std::string_view sv;
      ASSERT_OK(de_word->GetItemAt(&sv, {{}}));
      std::string ss(sv);
      EXPECT_STREQ(ss.c_str(), expected_result[i][j].c_str());
    }
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  // Expect 1 samples.
  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline.
  iter->Stop();

  // Restore configuration.
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

/// Feature: CoNLL2000Dataset
/// Description: Test CoNLL2000Dataset with ShuffleMode::kFalse with 2 text files
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestCoNLL2000DatasetShuffleGlobalB) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCoNLL2000DatasetShuffleGlobalB.";
  // Test CoNLL200 Dataset with 2 text files, global shuffle, num_parallel_workers=4.

  // Set configuration.
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(246);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  // Create a CoNLL2000 Dataset, with two text files.
  // Note: test.txt has 2 rows.
  // Note: train.txt has 3 rows.
  // Set shuffle to global shuffle.
  std::string dataset_dir = datasets_root_path_ + "/testCoNLL2000Dataset";
  std::shared_ptr<Dataset> ds = CoNLL2000(dataset_dir, "all", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  std::vector<std::string> column_names = {"word", "pos_tag", "chunk_tag"};
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("word"), row.end());
  std::vector<std::vector<std::string>> expected_result = {{"He", "PBP", "B-NP"},
                                                           {"Challenge", "NNP", "O"},
                                                           {"The", "DT", "B-NP"},
                                                           {"Her", "PP$", "B-NP"},
                                                           {"To", "TO", "I-VP"}};
  uint64_t i = 0;
  while (row.size() != 0) {
    for (int j = 0; j < column_names.size(); j++) {
      auto word = row[column_names[j]];
      std::shared_ptr<Tensor> de_word;
      ASSERT_OK(Tensor::CreateFromMSTensor(word, &de_word));
      std::string_view sv;
      ASSERT_OK(de_word->GetItemAt(&sv, {{}}));
      std::string ss(sv);
      EXPECT_STREQ(ss.c_str(), expected_result[i][j].c_str());
    }
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  // Expect 3 + 1 + 2 = 6 samples.
  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline.
  iter->Stop();

  // Restore configuration.
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}
