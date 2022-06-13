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
#include "minddata/dataset/include/dataset/datasets.h"

using namespace mindspore::dataset;
using mindspore::Status;
using mindspore::dataset::ShuffleMode;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

/// Feature: Test Multi30k Dataset(English).
/// Description: Read Multi30kDataset data and get data.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestMulti30kSuccessEn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMulti30kSuccessEn.";
  // Test Multi30k English files with default parameters

  // Create a Multi30k dataset
  std::string en_file = datasets_root_path_ + "/testMulti30kDataset";

  // test train
  std::string usage = "train";
  std::shared_ptr<Dataset> ds = Multi30k(en_file, usage, {"en", "de"}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::string> expected = {"This is the first English sentence in train.",
                                       "This is the second English sentence in train.",
                                       "This is the third English sentence in train."};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["text"];
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    ASSERT_OK(de_text->GetItemAt(&sv, {}));
    std::string ss(sv);
    EXPECT_STREQ(ss.c_str(), expected[i].c_str());
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();

  // test valid
  usage = "valid";
  expected = {"This is the first English sentence in valid.",
              "This is the second English sentence in valid."};

  ds = Multi30k(en_file, usage, {"en", "de"}, 0, ShuffleMode::kFalse);
  
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("text"), row.end());

  i = 0;
  while (row.size() != 0) {
    auto text = row["text"];
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    ASSERT_OK(de_text->GetItemAt(&sv, {}));
    std::string ss(sv);
    EXPECT_STREQ(ss.c_str(), expected[i].c_str());
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 2);

  iter->Stop();

  // test test
  usage = "test";
  expected = {"This is the first English sentence in test.",
              "This is the second English sentence in test.",
              "This is the third English sentence in test."};

  ds = Multi30k(en_file, usage, {"en", "de"}, 0, ShuffleMode::kFalse);
  
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("text"), row.end());

  i = 0;
  while (row.size() != 0) {
    auto text = row["text"];
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    ASSERT_OK(de_text->GetItemAt(&sv, {}));
    std::string ss(sv);
    EXPECT_STREQ(ss.c_str(), expected[i].c_str());
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 3);

  iter->Stop();
}

/// Feature: Test Multi30k Dataset(Germany).
/// Description: Read Multi30kDataset data and get data.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestMulti30kSuccessDe) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMulti30kSuccessDe.";
  // Test Multi30k Germany files with default parameters

  // Create a Multi30k dataset
  std::string en_file = datasets_root_path_ + "/testMulti30kDataset";

  // test train
  std::string usage = "train";
  std::shared_ptr<Dataset> ds = Multi30k(en_file, usage, {"en", "de"}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("translation"), row.end());
  std::vector<std::string> expected = {"This is the first Germany sentence in train.",
                                       "This is the second Germany sentence in train.",
                                       "This is the third Germany sentence in train."};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["translation"];
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    ASSERT_OK(de_text->GetItemAt(&sv, {}));
    std::string ss(sv);
    EXPECT_STREQ(ss.c_str(), expected[i].c_str());
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();

  // test valid
  usage = "valid";
  expected = {"This is the first Germany sentence in valid.",
              "This is the second Germany sentence in valid."};

  ds = Multi30k(en_file, usage, {"en", "de"}, 0, ShuffleMode::kFalse);
  
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("translation"), row.end());

  i = 0;
  while (row.size() != 0) {
    auto text = row["translation"];
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    ASSERT_OK(de_text->GetItemAt(&sv, {}));
    std::string ss(sv);
    EXPECT_STREQ(ss.c_str(), expected[i].c_str());
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 2);

  iter->Stop();

  // test test
  usage = "test";
  expected = {"This is the first Germany sentence in test.",
              "This is the second Germany sentence in test.",
              "This is the third Germany sentence in test."};

  ds = Multi30k(en_file, usage, {"en", "de"}, 0, ShuffleMode::kFalse);
  
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("translation"), row.end());

  i = 0;
  while (row.size() != 0) {
    auto text = row["translation"];
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    ASSERT_OK(de_text->GetItemAt(&sv, {}));
    std::string ss(sv);
    EXPECT_STREQ(ss.c_str(), expected[i].c_str());
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 3);

  iter->Stop();
}

/// Feature: Test Multi30k Dataset(Germany).
/// Description: Read Multi30kDataset data and get data.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestMulti30kDatasetBasicWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMulti30kDatasetBasicWithPipeline.";

  // Create two Multi30kFile Dataset, with single Multi30k file
  std::string train_en_file = datasets_root_path_ + "/testMulti30kDataset";
  std::string usage = "train";
  std::shared_ptr<Dataset> ds1 = Multi30k(train_en_file, usage, {"en", "de"}, 2, ShuffleMode::kFalse);
  std::shared_ptr<Dataset> ds2 = Multi30k(train_en_file, usage, {"en", "de"}, 2, ShuffleMode::kFalse);
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
  std::vector<std::string> column_project = {"text"};
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

  EXPECT_NE(row.find("text"), row.end());
  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["text"];
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  // Expect 10 samples
  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Test Getters.
/// Description: Includes tests for shape, type, size.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestMulti30kGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMulti30kGetters.";

  std::string train_en_file = datasets_root_path_ + "/testMulti30kDataset";
  std::string usage = "train";
  std::shared_ptr<Dataset> ds = Multi30k(train_en_file, usage, {"en", "de"}, 2, ShuffleMode::kFalse);
  std::vector<std::string> column_names = {"text","translation"};
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 2);
  EXPECT_EQ(ds->GetColumnNames(), column_names);
}

/// Feature: Test Multi30kDataset in distribution.
/// Description: Test interface in a distributed state.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestMulti30kDatasetDistribution) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMulti30kDatasetDistribution.";

  // Create a Multi30kFile Dataset, with single Multi30k file
  std::string train_en_file = datasets_root_path_ + "/testMulti30kDataset";
  std::string usage = "train";
  std::shared_ptr<Dataset> ds = Multi30k(train_en_file, usage, {"en", "de"}, 0, ShuffleMode::kGlobal, 3, 2);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("text"), row.end());
  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["text"];
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  // Expect 1 samples
  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Error Test.
/// Description: Test the wrong input.
/// Expectation: Unable to read in data.
TEST_F(MindDataTestPipeline, TestMulti30kDatasetFailInvalidFilePath) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMulti30kDatasetFailInvalidFilePath.";

  // Create a Multi30k Dataset
  // with invalid file path
  std::string train_en_file = datasets_root_path_ + "/invalid/file.path";
  std::string usage = "train";
  std::shared_ptr<Dataset> ds = Multi30k(train_en_file, usage, {"en", "de"});
  EXPECT_NE(ds, nullptr); 
}

/// Feature: Error Test.
/// Description: Test the wrong input.
/// Expectation: Unable to read in data.
TEST_F(MindDataTestPipeline, TestMulti30kDatasetFailInvalidUsage) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMulti30kDatasetFailInvaildUsage.";

  // Create a Multi30k Dataset
  // with invalid usage
  std::string train_en_file = datasets_root_path_ + "/testMulti30kDataset";
  std::string usage = "invalid_usage";
  std::shared_ptr<Dataset> ds = Multi30k(train_en_file, usage, {"en", "de"});
  EXPECT_NE(ds, nullptr); 
}

/// Feature: Error Test.
/// Description: Test the wrong input.
/// Expectation: Unable to read in data.
TEST_F(MindDataTestPipeline, TestMulti30kDatasetFailInvalidLanguagePair) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMulti30kDatasetFailLanguagePair.";

  // Create a Multi30k Dataset
  // with invalid usage
  std::string train_en_file = datasets_root_path_ + "/testMulti30kDataset";
  std::string usage = "train";
  std::vector<std::string> language_pair0 = {"ch", "ja"};
  std::shared_ptr<Dataset> ds0 = Multi30k(train_en_file, usage, language_pair0);
  EXPECT_NE(ds0, nullptr);
  std::vector<std::string> language_pair1 = {"en", "de", "aa"};
  std::shared_ptr<Dataset> ds1 = Multi30k(train_en_file, usage, language_pair1);
  EXPECT_NE(ds1, nullptr);
}

/// Feature: Error Test.
/// Description: Test the wrong input.
/// Expectation: Unable to read in data.
TEST_F(MindDataTestPipeline, TestMulti30kDatasetFailInvalidNumSamples) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMulti30kDatasetFailInvalidNumSamples.";

  // Create a Multi30k Dataset
  // with invalid samplers=-1
  std::string train_en_file = datasets_root_path_ + "/testMulti30kDataset";
  std::string usage = "train";
  std::shared_ptr<Dataset> ds = Multi30k(train_en_file, usage, {"en", "de"}, -1);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: TextFile number of samples cannot be negative
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Error Test.
/// Description: Test the wrong input.
/// Expectation: Unable to read in data.
TEST_F(MindDataTestPipeline, TestMulti30kDatasetFailInvalidShards) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMulti30kDatasetFailInvalidShards.";

  // Create a Multi30k Dataset
  // with invalid shards.
  std::string train_en_file = datasets_root_path_ + "/testMulti30kDataset";
  std::string usage = "train";
  std::shared_ptr<Dataset> ds = Multi30k(train_en_file, usage, {"en", "de"}, 0, ShuffleMode::kFalse, 2, 3);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: TextFile number of samples cannot be negative
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Error Test.
/// Description: Test the wrong input.
/// Expectation: Unable to read in data.
TEST_F(MindDataTestPipeline, TestMulti30kDatasetFailInvalidShardID) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMulti30kDatasetFailInvalidShardID.";

  // Create a Multi30k Dataset
  // with invalid shard ID.
  std::string train_en_file = datasets_root_path_ + "/testMulti30kDataset";
  std::string usage = "train";
  std::shared_ptr<Dataset> ds = Multi30k(train_en_file, usage, {"en", "de"}, 0, ShuffleMode::kFalse, 0, -1);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: TextFile number of samples cannot be negative
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Error Test.
/// Description: Test the wrong input.
/// Expectation: Unable to read in data.
TEST_F(MindDataTestPipeline, TestMulti30kDatasetLanguagePair) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMulti30kDatasetLanguagePair.";

  std::string train_en_file = datasets_root_path_ + "/testMulti30kDataset";
  std::string usage = "train";
  std::shared_ptr<Dataset> ds = Multi30k(train_en_file, usage, {"de", "en"}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("translation"), row.end());
  std::vector<std::string> expected = {"This is the first English sentence in train.",
                                       "This is the second English sentence in train.",
                                       "This is the third English sentence in train."};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["translation"];
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    ASSERT_OK(de_text->GetItemAt(&sv, {}));
    std::string ss(sv);
    EXPECT_STREQ(ss.c_str(), expected[i].c_str());
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 3);

  iter->Stop();
}

/// Feature: Test Multi30k Dataset(shufflemode=kFalse).
/// Description: Test Multi30k Dataset interface with different ShuffleMode.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestMulti30kDatasetShuffleFilesFalse) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMulti30kDatasetShuffleFilesFalse.";

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(135);
  GlobalContext::config_manager()->set_num_parallel_workers(1);

  std::string train_en_file = datasets_root_path_ + "/testMulti30kDataset";
  std::string usage = "train";
  std::shared_ptr<Dataset> ds = Multi30k(train_en_file, usage, {"en", "de"}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::string> expected = {"This is the first English sentence in train.",
                                       "This is the second English sentence in train.",
                                       "This is the third English sentence in train."};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["text"];
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    ASSERT_OK(de_text->GetItemAt(&sv, {}));
    std::string ss(sv);
    MS_LOG(INFO) << "Text length: " << ss.length() << ", Text: " << ss.substr(0, 50);
    // Compare against expected result
    EXPECT_STREQ(ss.c_str(), expected[i].c_str());
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();

  // Restore configuration
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

/// Feature: Test Multi30k Dataset(shufflemode=kFiles).
/// Description: Test Multi30k Dataset interface with different ShuffleMode.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestMulti30kDatasetShuffleFilesFiles) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMulti30kDatasetShuffleFilesFiles.";

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(135);
  GlobalContext::config_manager()->set_num_parallel_workers(1);

  std::string train_en_file = datasets_root_path_ + "/testMulti30kDataset";
  std::string usage = "train";
  std::shared_ptr<Dataset> ds = Multi30k(train_en_file, usage, {"en", "de"}, 0, ShuffleMode::kFiles);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::string> expected = {"This is the first English sentence in train.",
                                       "This is the second English sentence in train.",
                                       "This is the third English sentence in train."};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["text"];
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    ASSERT_OK(de_text->GetItemAt(&sv, {}));
    std::string ss(sv);
    MS_LOG(INFO) << "Text length: " << ss.length() << ", Text: " << ss.substr(0, 50);
    // Compare against expected result
    EXPECT_STREQ(ss.c_str(), expected[i].c_str());
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();

  // Restore configuration
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

/// Feature: Test Multi30k Dataset(shufflemode=kGlobal).
/// Description: Test Multi30k Dataset interface with different ShuffleMode.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestMulti30kDatasetShuffleFilesGlobal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMulti30kDatasetShuffleFilesGlobal.";

  std::string train_en_file = datasets_root_path_ + "/testMulti30kDataset";
  std::string usage = "train";
  std::shared_ptr<Dataset> ds = Multi30k(train_en_file, usage, {"en", "de"}, 0, ShuffleMode::kGlobal);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("text"), row.end());

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["text"];
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    ASSERT_OK(de_text->GetItemAt(&sv, {}));
    std::string ss(sv);
    MS_LOG(INFO) << "Text length: " << ss.length() << ", Text: " << ss.substr(0, 50);
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}