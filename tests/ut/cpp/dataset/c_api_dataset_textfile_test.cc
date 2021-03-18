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
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/include/datasets.h"

using namespace mindspore::dataset;

using mindspore::dataset::ShuffleMode;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestPipeline, TestTextFileDatasetBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTextFileDatasetBasic.";
  // Test TextFile Dataset with single text file and many default inputs

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(987);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  // Create a TextFile Dataset, with single text file
  // Note: 1.txt has 3 rows
  // Use 2 samples
  // Use defaults for other input parameters
  std::string tf_file1 = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({tf_file1}, 2);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::string> expected_result = {"Be happy every day.", "This is a text file."};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["text"];
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    de_text->GetItemAt(&sv, {0});
    std::string ss(sv);
    MS_LOG(INFO) << "Text length: " << ss.length() << ", Text: " << ss.substr(0, 50);
    // Compare against expected result
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());

    i++;
    iter->GetNextRow(&row);
  }

  // Expect 2 samples
  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();

  // Restore configuration
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

TEST_F(MindDataTestPipeline, TestTextFileDatasetBasicWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTextFileDatasetBasicWithPipeline.";
  // Test TextFile Dataset with single text file and many default inputs

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(987);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  // Create two TextFile Dataset, with single text file
  // Note: 1.txt has 3 rows
  // Use 2 samples
  // Use defaults for other input parameters
  std::string tf_file1 = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds1 = TextFile({tf_file1}, 2);
  std::shared_ptr<Dataset> ds2 = TextFile({tf_file1}, 2);
  EXPECT_NE(ds1, nullptr);
  EXPECT_NE(ds2, nullptr);

  // Create two Repeat operation on ds
  int32_t repeat_num = 2;
  ds1 = ds1->Repeat(repeat_num);
  EXPECT_NE(ds1, nullptr);
  repeat_num = 3;
  ds2 = ds2->Repeat(repeat_num);
  EXPECT_NE(ds2, nullptr);

  // Create a Concat operation on the ds
  ds1 = ds1->Concat({ds2});
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds1->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::string> expected_result = {"Be happy every day.", "This is a text file."};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["text"];
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    i++;
    iter->GetNextRow(&row);
  }

  // Expect 10 samples
  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();

  // Restore configuration
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

TEST_F(MindDataTestPipeline, TestTextFileGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTextFileGetters.";
  // Test TextFile Dataset with single text file and many default inputs

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(987);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  // Create a TextFile Dataset, with single text file
  // Note: 1.txt has 3 rows
  // Use 2 samples
  // Use defaults for other input parameters
  std::string tf_file1 = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({tf_file1}, 2);
  EXPECT_NE(ds, nullptr);

  std::vector<std::string> column_names = {"text"};
  EXPECT_EQ(ds->GetDatasetSize(), 2);
  EXPECT_EQ(ds->GetColumnNames(), column_names);

  ds = TextFile({tf_file1}, 0);
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 3);
  // Restore configuration
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

TEST_F(MindDataTestPipeline, TestTextFileDatasetFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTextFileDatasetFail1.";

  // Create a TextFile Dataset
  // with invalid samplers=-1
  std::string tf_file1 = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({tf_file1}, -1);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: TextFile number of samples cannot be negative
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestTextFileDatasetFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTextFileDatasetFail2.";

  // Attempt to create a TextFile Dataset
  // with wrongful empty dataset_files input
  std::shared_ptr<Dataset> ds = TextFile({});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: dataset_files is not specified
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestTextFileDatasetFail3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTextFileDatasetFail3.";

  // Create a TextFile Dataset
  // with non-existent dataset_files input
  std::string tf_file1 = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({tf_file1, "notexist.txt"}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: specified dataset_files does not exist
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestTextFileDatasetFail4) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTextFileDatasetFail4.";

  // Create a TextFile Dataset
  // with empty string dataset_files input
  std::shared_ptr<Dataset> ds = TextFile({""}, 0, ShuffleMode::kFiles);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: specified dataset_files does not exist
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestTextFileDatasetFail5) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTextFileDatasetFail5.";

  // Create a TextFile Dataset
  // with invalid num_shards=0 value
  std::string tf_file1 = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({tf_file1}, 1, ShuffleMode::kFalse, 0);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Number of shards cannot be <=0
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestTextFileDatasetFail6) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTextFileDatasetFail6.";

  // Create a TextFile Dataset
  // with invalid shard_id=-1 value
  std::string tf_file1 = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({tf_file1}, 0, ShuffleMode::kFiles, -1);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: shard_id cannot be negative
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestTextFileDatasetFail7) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTextFileDatasetFail7.";

  // Create a TextFile Dataset
  // with invalid shard_id=2 and num_shards=2 combination
  std::string tf_file1 = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({tf_file1}, 0, ShuffleMode::kGlobal, 2, 2);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Cannot have shard_id >= num_shards
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestTextFileDatasetShuffleFalse1A) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTextFileDatasetShuffleFalse1A.";
  // Test TextFile Dataset with two text files and no shuffle, num_parallel_workers=1

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(654);
  GlobalContext::config_manager()->set_num_parallel_workers(1);

  // Create a TextFile Dataset, with two text files, 1.txt then 2.txt, in lexicographical order.
  // Note: 1.txt has 3 rows
  // Note: 2.txt has 2 rows
  // Use default of all samples
  std::string tf_file1 = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::string tf_file2 = datasets_root_path_ + "/testTextFileDataset/2.txt";
  std::shared_ptr<Dataset> ds = TextFile({tf_file1, tf_file2}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::string> expected_result = {"This is a text file.", "Be happy every day.", "Good luck to everyone.",
                                              "Another file.", "End of file."};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["text"];
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    de_text->GetItemAt(&sv, {0});
    std::string ss(sv);
    MS_LOG(INFO) << "Text length: " << ss.length() << ", Text: " << ss.substr(0, 50);
    // Compare against expected result
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());

    i++;
    iter->GetNextRow(&row);
  }

  // Expect 2 + 3 = 5 samples
  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();

  // Restore configuration
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

TEST_F(MindDataTestPipeline, TestTextFileDatasetShuffleFalse1B) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTextFileDatasetShuffleFalse1B.";
  // Test TextFile Dataset with two text files and no shuffle, num_parallel_workers=1

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(654);
  GlobalContext::config_manager()->set_num_parallel_workers(1);

  // Create a TextFile Dataset, with two text files, 2.txt then 1.txt, in non-lexicographical order
  // Note: 1.txt has 3 rows
  // Note: 2.txt has 2 rows
  // Use default of all samples
  std::string tf_file1 = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::string tf_file2 = datasets_root_path_ + "/testTextFileDataset/2.txt";
  std::shared_ptr<Dataset> ds = TextFile({tf_file2, tf_file1}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::string> expected_result = {"This is a text file.", "Be happy every day.", "Good luck to everyone.",
                                              "Another file.", "End of file."};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["text"];
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    de_text->GetItemAt(&sv, {0});
    std::string ss(sv);
    MS_LOG(INFO) << "Text length: " << ss.length() << ", Text: " << ss.substr(0, 50);
    // Compare against expected result
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());

    i++;
    iter->GetNextRow(&row);
  }

  // Expect 2 + 3 = 5 samples
  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();

  // Restore configuration
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

TEST_F(MindDataTestPipeline, TestTextFileDatasetShuffleFalse4Shard) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTextFileDatasetShuffleFalse4Shard.";
  // Test TextFile Dataset with two text files and no shuffle, num_parallel_workers=4, shard coverage

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(654);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  // Create a TextFile Dataset, with two text files
  // Note: 1.txt has 3 rows
  // Note: 2.txt has 2 rows
  // Set shuffle to file shuffle, num_shards=2, shard_id=0
  std::string tf_file1 = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::string tf_file2 = datasets_root_path_ + "/testTextFileDataset/2.txt";
  std::shared_ptr<Dataset> ds = TextFile({tf_file1, tf_file2}, 0, ShuffleMode::kFalse, 2, 0);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::string> expected_result = {"This is a text file.", "Be happy every day.", "Good luck to everyone."};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["text"];
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    de_text->GetItemAt(&sv, {0});
    std::string ss(sv);
    MS_LOG(INFO) << "Text length: " << ss.length() << ", Text: " << ss.substr(0, 50);
    // Compare against expected result
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());

    i++;
    iter->GetNextRow(&row);
  }

  // Expect 3 samples for this shard
  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();

  // Restore configuration
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

TEST_F(MindDataTestPipeline, TestTextFileDatasetShuffleFiles1A) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTextFileDatasetShuffleFiles1A.";
  // Test TextFile Dataset with files shuffle, num_parallel_workers=1

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(135);
  GlobalContext::config_manager()->set_num_parallel_workers(1);

  // Create a TextFile Dataset, with two text files, 1.txt then 2.txt, in lexicographical order.
  // Note: 1.txt has 3 rows
  // Note: 2.txt has 2 rows
  // Use default of all samples
  // Set shuffle to files shuffle
  std::string tf_file1 = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::string tf_file2 = datasets_root_path_ + "/testTextFileDataset/2.txt";
  std::shared_ptr<Dataset> ds = TextFile({tf_file1, tf_file2}, 0, ShuffleMode::kFiles);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::string> expected_result = {
    "This is a text file.", "Be happy every day.", "Good luck to everyone.", "Another file.", "End of file.",
  };

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["text"];
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    de_text->GetItemAt(&sv, {0});
    std::string ss(sv);
    MS_LOG(INFO) << "Text length: " << ss.length() << ", Text: " << ss.substr(0, 50);
    // Compare against expected result
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());

    i++;
    iter->GetNextRow(&row);
  }

  // Expect 2 + 3 = 5 samples
  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();

  // Restore configuration
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

TEST_F(MindDataTestPipeline, TestTextFileDatasetShuffleFiles1B) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTextFileDatasetShuffleFiles1B.";
  // Test TextFile Dataset with files shuffle, num_parallel_workers=1

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(135);
  GlobalContext::config_manager()->set_num_parallel_workers(1);

  // Create a TextFile Dataset, with two text files, 2.txt then 1.txt, in non-lexicographical order.
  // Note: 1.txt has 3 rows
  // Note: 2.txt has 2 rows
  // Use default of all samples
  // Set shuffle to files shuffle
  std::string tf_file1 = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::string tf_file2 = datasets_root_path_ + "/testTextFileDataset/2.txt";
  std::shared_ptr<Dataset> ds = TextFile({tf_file2, tf_file1}, 0, ShuffleMode::kFiles);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::string> expected_result = {
    "This is a text file.", "Be happy every day.", "Good luck to everyone.", "Another file.", "End of file.",
  };

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["text"];
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    de_text->GetItemAt(&sv, {0});
    std::string ss(sv);
    MS_LOG(INFO) << "Text length: " << ss.length() << ", Text: " << ss.substr(0, 50);
    // Compare against expected result
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());

    i++;
    iter->GetNextRow(&row);
  }

  // Expect 2 + 3 = 5 samples
  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();

  // Restore configuration
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

TEST_F(MindDataTestPipeline, TestTextFileDatasetShuffleFiles4) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTextFileDatasetShuffleFiles4.";
  // Test TextFile Dataset with files shuffle, num_parallel_workers=4

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(135);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  // Create a TextFile Dataset, with two text files
  // Note: 1.txt has 3 rows
  // Note: 2.txt has 2 rows
  // Use default of all samples
  // Set shuffle to files shuffle
  std::string tf_file1 = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::string tf_file2 = datasets_root_path_ + "/testTextFileDataset/2.txt";
  std::shared_ptr<Dataset> ds = TextFile({tf_file1, tf_file2}, 0, ShuffleMode::kFiles);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::string> expected_result = {"This is a text file.", "Another file.", "Be happy every day.",
                                              "End of file.", "Good luck to everyone."};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["text"];
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    de_text->GetItemAt(&sv, {0});
    std::string ss(sv);
    MS_LOG(INFO) << "Text length: " << ss.length() << ", Text: " << ss.substr(0, 50);
    // Compare against expected result
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());

    i++;
    iter->GetNextRow(&row);
  }

  // Expect 2 + 3 = 5 samples
  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();

  // Restore configuration
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

TEST_F(MindDataTestPipeline, TestTextFileDatasetShuffleGlobal1A) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTextFileDatasetShuffleGlobal1A.";
  // Test TextFile Dataset with 1 text file, global shuffle, num_parallel_workers=1

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(246);
  GlobalContext::config_manager()->set_num_parallel_workers(1);

  // Create a TextFile Dataset, with two text files
  // Note: 1.txt has 3 rows
  // Set shuffle to global shuffle
  std::string tf_file1 = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({tf_file1}, 0, ShuffleMode::kGlobal);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::string> expected_result = {"Good luck to everyone.", "This is a text file.", "Be happy every day."};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["text"];
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    de_text->GetItemAt(&sv, {0});
    std::string ss(sv);
    MS_LOG(INFO) << "Text length: " << ss.length() << ", Text: " << ss.substr(0, 50);
    // Compare against expected result
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());

    i++;
    iter->GetNextRow(&row);
  }

  // Expect 3 samples
  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();

  // Restore configuration
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

TEST_F(MindDataTestPipeline, TestTextFileDatasetShuffleGlobal1B) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTextFileDatasetShuffleGlobal1B.";
  // Test TextFile Dataset with 2 text files, global shuffle, num_parallel_workers=1

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(246);
  GlobalContext::config_manager()->set_num_parallel_workers(1);

  // Create a TextFile Dataset, with two text files
  // Note: 1.txt has 3 rows
  // Note: 2.txt has 2 rows
  // Set shuffle to global shuffle
  std::string tf_file1 = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::string tf_file2 = datasets_root_path_ + "/testTextFileDataset/2.txt";
  std::shared_ptr<Dataset> ds = TextFile({tf_file1, tf_file2}, 0, ShuffleMode::kGlobal);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::string> expected_result = {"Another file.", "Good luck to everyone.", "This is a text file.",
                                              "End of file.", "Be happy every day."};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["text"];
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    de_text->GetItemAt(&sv, {0});
    std::string ss(sv);
    MS_LOG(INFO) << "Text length: " << ss.length() << ", Text: " << ss.substr(0, 50);
    // Compare against expected result
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());

    i++;
    iter->GetNextRow(&row);
  }

  // Expect 2 + 3 = 5 samples
  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();

  // Restore configuration
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

TEST_F(MindDataTestPipeline, TestTextFileDatasetShuffleGlobal4) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTextFileDatasetShuffleGlobal4.";
  // Test TextFile Dataset with 2 text files, global shuffle, num_parallel_workers=4

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(246);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  // Create a TextFile Dataset, with two text files
  // Note: 1.txt has 3 rows
  // Note: 2.txt has 2 rows
  // Set shuffle to global shuffle
  std::string tf_file1 = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::string tf_file2 = datasets_root_path_ + "/testTextFileDataset/2.txt";
  std::shared_ptr<Dataset> ds = TextFile({tf_file1, tf_file2}, 0, ShuffleMode::kGlobal);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::string> expected_result = {"Another file.", "Good luck to everyone.", "End of file.",
                                              "This is a text file.", "Be happy every day."};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["text"];
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    de_text->GetItemAt(&sv, {0});
    std::string ss(sv);
    MS_LOG(INFO) << "Text length: " << ss.length() << ", Text: " << ss.substr(0, 50);
    // Compare against expected result
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());

    i++;
    iter->GetNextRow(&row);
  }

  // Expect 2 + 3 = 5 samples
  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();

  // Restore configuration
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}
