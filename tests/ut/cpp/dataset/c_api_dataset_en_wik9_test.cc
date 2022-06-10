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
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/include/dataset/datasets.h"

using namespace mindspore::dataset;

using mindspore::dataset::ShuffleMode;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

/// Feature: EnWik9Dataset
/// Description: Test EnWik9Dataset basic usage
/// Expectation: The number of samples is proper
TEST_F(MindDataTestPipeline, TestEnWik9DatasetBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestEnWik9DatasetBasic.";
  // Test EnWik9 Dataset with single text file and many default inputs.

  // Set configuration.
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(987);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  // Create a EnWik9 Dataset, with single enwik9 file.
  // Note: /testEnWik9Dataset/enwik9 has 13 rows.
  // Use 2 samples.
  // Use defaults for other input parameters.
  std::string tf_file = datasets_root_path_ + "/testEnWik9Dataset";
  std::shared_ptr<Dataset> ds = EnWik9(tf_file, 2);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::string> expected_result = {"    <title>MindSpore</title>", "  <page>"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["text"];
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    ASSERT_OK(de_text->GetItemAt(&sv, {}));
    std::string ss(sv);
    MS_LOG(INFO) << "Text length: " << ss.length() << ", Text: " << ss.substr(0, 50);
    // Compare against expected result.
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());

    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  // Expect 2 samples.
  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline.
  iter->Stop();

  // Restore configuration.
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

/// Feature: EnWik9Dataset
/// Description: Test EnWik9Dataset basic usage with repeat op
/// Expectation: The number of samples is proper
TEST_F(MindDataTestPipeline, TestEnWik9DatasetBasicAndRepeat) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestEnWik9DatasetBasicAndRepeat.";
  // Test EnWik9 Dataset with single enwik9 file and many default inputs.

  // Set configuration.
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(987);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  // Create two EnWik9 Dataset, with single enwik9 file.
  // Note: /testEnWik9Dataset/enwik9 has 13 rows.
  // Use 2 samples.
  // Use defaults for other input parameters.
  std::string tf_file = datasets_root_path_ + "/testEnWik9Dataset";
  std::shared_ptr<Dataset> ds1 = EnWik9(tf_file, 2);
  std::shared_ptr<Dataset> ds2 = EnWik9(tf_file, 2);
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
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::string> expected_result = {"  <page>", "    <title>MindSpore</title>"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["text"];
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
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

/// Feature: EnWik9Dataset
/// Description: Test EnWik9Dataset Getters method
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestEnWik9Getters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestEnWik9Getters.";
  // Test EnWik9 Dataset with single text file and many default inputs.

  // Set configuration.
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(987);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  // Create a EnWik9 Dataset, with single enwik9 file.
  // Note: /testEnWik9Dataset/enwik9 has 3 rows.
  // Use 2 samples.
  // Use defaults for other input parameters.
  std::string tf_file = datasets_root_path_ + "/testEnWik9Dataset";
  std::shared_ptr<Dataset> ds = EnWik9(tf_file, 2);
  EXPECT_NE(ds, nullptr);

  std::vector<std::string> column_names = {"text"};
  EXPECT_EQ(ds->GetDatasetSize(), 2);
  EXPECT_EQ(ds->GetColumnNames(), column_names);

  ds = EnWik9(tf_file, 0);
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 13);

  // Restore configuration.
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

/// Feature: EnWik9Dataset
/// Description: Test EnWik9Dataset with non-existent dataset_files
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestEnWik9DatasetFailNoExistentPath) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestEnWik9DatasetFailNoExistentPath.";

  // Create a EnWik9 Dataset.
  // with non-existent dataset_files input.
  std::string tf_file = datasets_root_path_ + "/testEnWik9Dataset";
  std::shared_ptr<Dataset> ds = EnWik9("/NotExist", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: specified dataset_files does not exist
  EXPECT_EQ(iter, nullptr);
}

/// Feature: EnWik9Dataset
/// Description: Test EnWik9Dataset with ShuffleMode::kFalse
/// Expectation: The data of samples is proper
TEST_F(MindDataTestPipeline, TestEnWik9DatasetShuffleFalse1A) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestEnWik9DatasetShuffleFalse1A.";
  // Test EnWik9 Dataset with two enwik9 files and no shuffle, num_parallel_workers=1.

  // Set configuration.
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(654);
  GlobalContext::config_manager()->set_num_parallel_workers(1);

  // Create a EnWik9 Dataset, with one enwik9 file, /testEnWik9Dataset/enwik9.
  // Note: /testEnWik9Dataset/enwik9 has 13 rows.
  // Use default of all samples
  std::string tf_file = datasets_root_path_ + "/testEnWik9Dataset";
  std::shared_ptr<Dataset> ds = EnWik9(tf_file, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::string> expected_result = {"  <page>",
                                              "    <title>MindSpore</title>",
                                              "    <id>1</id>",
                                              "    <revision>",
                                              "      <id>234</id>",
                                              "      <timestamp>2020-01-01T00:00:00Z</timestamp>",
                                              "      <contributor>",
                                              "        <username>MS</username>",
                                              "        <id>567</id>",
                                              "      </contributor>",
                                              "      <text xml:space=\"preserve\">666</text>",
                                              "    </revision>",
                                              "  </page>"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["text"];
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    ASSERT_OK(de_text->GetItemAt(&sv, {}));
    std::string ss(sv);
    MS_LOG(INFO) << "Text length: " << ss.length() << ", Text: " << ss.substr(0, 50);
    // Compare against expected result.
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());

    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  // Expect 13 samples.
  EXPECT_EQ(i, 13);

  // Manually terminate the pipeline.
  iter->Stop();

  // Restore configuration.
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

/// Feature: EnWik9Dataset
/// Description: Test EnWik9Dataset with ShuffleMode::kFalse and with num_shards and shard_id
/// Expectation: The data of samples is proper
TEST_F(MindDataTestPipeline, TestEnWik9DatasetShuffleFalse4Shard) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestEnWik9DatasetShuffleFalse4Shard.";
  // Test EnWik9 Dataset with one enwik9 files and no shuffle, num_parallel_workers=4, shard coverage.

  // Set configuration.
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(654);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  // Create a EnWik9 Dataset, with one enwik9 file.
  // Note: /testEnWik9Dataset/enwik9 has 13 rows.
  // Set shuffle to file shuffle, num_shards=2, shard_id=0
  std::string tf_file = datasets_root_path_ + "/testEnWik9Dataset";
  std::shared_ptr<Dataset> ds = EnWik9(tf_file, 0, ShuffleMode::kFalse, 2, 0);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::string> expected_result = {"  <page>",
                                              "    <title>MindSpore</title>",
                                              "    <id>1</id>",
                                              "    <revision>",
                                              "      <id>234</id>",
                                              "      <timestamp>2020-01-01T00:00:00Z</timestamp>",
                                              "      <contributor>",
                                              "        <username>MS</username>",
                                              "        <id>567</id>",
                                              "      </contributor>",
                                              "      <text xml:space=\"preserve\">666</text>",
                                              "    </revision>",
                                              "  </page>"};
  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["text"];
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    ASSERT_OK(de_text->GetItemAt(&sv, {}));
    std::string ss(sv);
    MS_LOG(INFO) << "Text length: " << ss.length() << ", Text: " << ss.substr(0, 50);
    // Compare against expected result.
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());

    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  // Expect 7 samples for this shard.
  EXPECT_EQ(i, 7);

  // Manually terminate the pipeline.
  iter->Stop();

  // Restore configuration.
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

/// Feature: EnWik9Dataset
/// Description: Test EnWik9Dataset with ShuffleMode::kGlobal
/// Expectation: The data of samples is proper
TEST_F(MindDataTestPipeline, TestEnWik9DatasetShuffleGlobal1A) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestEnWik9DatasetShuffleGlobal1A.";
  // Test EnWik9 Dataset with one enwik9 file, global shuffle, num_parallel_workers=1.

  // Set configuration.
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(246);
  GlobalContext::config_manager()->set_num_parallel_workers(1);

  // Create a EnWik9 Dataset, with one enwik9 file.
  // Note: /testEnWik9Dataset/enwik9 has 13 rows.
  // Set shuffle to global shuffle.
  std::string tf_file = datasets_root_path_ + "/testEnWik9Dataset";
  std::shared_ptr<Dataset> ds = EnWik9(tf_file, 0, ShuffleMode::kGlobal);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::string> expected_result = {"      </contributor>",
                                              "  <page>",
                                              "      <contributor>",
                                              "        <username>MS</username>",
                                              "    <title>MindSpore</title>",
                                              "      <timestamp>2020-01-01T00:00:00Z</timestamp>",
                                              "      <text xml:space=\"preserve\">666</text>",
                                              "    <revision>",
                                              "        <id>567</id>",
                                              "    </revision>",
                                              "  </page>",
                                              "      <id>234</id>",
                                              "    <id>1</id>"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["text"];
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    ASSERT_OK(de_text->GetItemAt(&sv, {}));
    std::string ss(sv);
    MS_LOG(INFO) << "Text length: " << ss.length() << ", Text: " << ss.substr(0, 50);
    // Compare against expected result.
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());

    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  // Expect 13 samples.
  EXPECT_EQ(i, 13);

  // Manually terminate the pipeline.
  iter->Stop();

  // Restore configuration.
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}