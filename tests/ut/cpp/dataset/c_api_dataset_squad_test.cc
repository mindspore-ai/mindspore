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
using mindspore::dataset::GlobalContext;
using mindspore::dataset::ShuffleMode;
using mindspore::dataset::Tensor;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

/// Feature: SQuADDataset.
/// Description: Test SQuADDataset basic.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestSQuADDatasetBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSQuADDatasetBasic.";

  // Create a SQuAD Dataset, with single SQuAD file.
  std::string dataset_dir = datasets_root_path_ + "/testSQuAD/SQuAD1";
  std::string usage = "train";
  std::shared_ptr<Dataset> ds = SQuAD(dataset_dir, usage, 2);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("context"), row.end());
  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["context"];
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  // Expect 2 samples.
  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: SQuADDataset.
/// Description: Test SQuADDataset in pipeline mode.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestSQuADDatasetBasicWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSQuADDatasetBasicWithPipeline.";

  // Create two SQuAD Dataset, with single SQuAD file.
  std::string dataset_dir = datasets_root_path_ + "/testSQuAD/SQuAD1";
  std::string usage = "train";
  std::shared_ptr<Dataset> ds1 = SQuAD(dataset_dir, usage, 2);
  std::shared_ptr<Dataset> ds2 = SQuAD(dataset_dir, usage, 2);
  EXPECT_NE(ds1, nullptr);
  EXPECT_NE(ds2, nullptr);

  // Create two Repeat operation on ds.
  int32_t repeat_num = 2;
  ds1 = ds1->Repeat(repeat_num);
  EXPECT_NE(ds1, nullptr);
  repeat_num = 3;
  ds2 = ds2->Repeat(repeat_num);
  EXPECT_NE(ds2, nullptr);

  // Create two Project operation on ds.
  std::vector<std::string> column_project = {"context"};
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

  EXPECT_NE(row.find("context"), row.end());
  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["context"];
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  // Expect 10 samples.
  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: SQuADDataset.
/// Description: Test the getter functions of SQuADDataset.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestSQuADGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSQuADGetters.";

  // Create a SQuAD Dataset, with single SQuAD file.
  std::string dataset_dir = datasets_root_path_ + "/testSQuAD/SQuAD1";
  std::string usage = "train";
  std::shared_ptr<Dataset> ds = SQuAD(dataset_dir, usage, 2);
  std::vector<std::string> column_names = {"context", "question", "text", "answer_start"};
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 2);
  EXPECT_EQ(ds->GetColumnNames(), column_names);
}

/// Feature: SQuADDataset.
/// Description: Test SQuAD1.1 for train, dev and all.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestSQuADDatasetVersion1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSQuADDatasetVersion1.";

  // Create a SQuAD Dataset, with single SQuAD file.
  std::string dataset_dir = datasets_root_path_ + "/testSQuAD/SQuAD1";

  // train.
  std::string usage = "train";
  std::shared_ptr<Dataset> ds = SQuAD(dataset_dir, usage, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("context"), row.end());
  std::vector<std::string> expected_result = {
    "John von Neumann (December 28, 1903-February 8, 1957), American Hungarian mathematician, computer scientist, and "
    "physicist, was called \"The Father of Modern Computers\" and \"The Father of Game Theory.\"",
    "John von Neumann (December 28, 1903-February 8, 1957), American Hungarian mathematician, computer scientist, and "
    "physicist, was called \"The Father of Modern Computers\" and \"The Father of Game Theory.\"",
    "John von Neumann (December 28, 1903-February 8, 1957), American Hungarian mathematician, computer scientist, and "
    "physicist, was called \"The Father of Modern Computers\" and \"The Father of Game Theory.\""};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["context"];
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    ASSERT_OK(de_text->GetItemAt(&sv, {}));
    std::string ss(sv);
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  // Expect 3 samples.
  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline.
  iter->Stop();

  // dev.
  usage = "dev";
  expected_result = {"Who is the author of \"The Mathematical Principles of Natural Philosophy\"?",
                     "When was the publication year of \"The Mathematical Principles of Natural Philosophy\"?"};
  ds = SQuAD(dataset_dir, usage, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("question"), row.end());
  i = 0;
  while (row.size() != 0) {
    auto text = row["question"];
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    ASSERT_OK(de_text->GetItemAt(&sv, {}));
    std::string ss(sv);
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  // Expect 2 samples.
  EXPECT_EQ(i, 2);

  iter->Stop();

  // all.
  usage = "all";
  expected_result = {"John von Neumann", "Isaac Newton", "December 28, 1903", "1687", "American Hungarian"};
  ds = SQuAD(dataset_dir, usage, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("text"), row.end());
  i = 0;
  while (row.size() != 0) {
    auto text = row["text"];
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    ASSERT_OK(de_text->GetItemAt(&sv, {0}));
    std::string ss(sv);
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());
    ASSERT_OK(iter->GetNextRow(&row));
     i++;
  }

  // Expect 5 samples.
  EXPECT_EQ(i, 5);

  iter->Stop();
}

/// Feature: SQuADDataset.
/// Description: Test SQuAD2.0 for train, dev and all.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestSQuADDatasetVersion2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSQuADDatasetVersion2.";

  // Create a SQuAD Dataset, with single SQuAD file
  std::string dataset_dir = datasets_root_path_ + "/testSQuAD/SQuAD2";
  std::string usage = "train";
  std::shared_ptr<Dataset> ds = SQuAD(dataset_dir, usage, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("question"), row.end());
  std::vector<std::string> expected_result = {"Where is Stephen William Hawking's birthplace?",
                                              "When was Stephen William Hawking's birth date?"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["question"];
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    ASSERT_OK(de_text->GetItemAt(&sv, {}));
    std::string ss(sv);
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  // Expect 2 samples.
  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline.
  iter->Stop();

  // dev.
  usage = "dev";
  ds = SQuAD(dataset_dir, usage, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("context"), row.end());
  expected_result = {
    "Dolphins are a collective name for a class of aquatic mammals in the Dolphin family. They are small or "
    "medium-sized toothed whales. They live widely in all oceans of the world. They are also distributed in inland "
    "seas and brackish waters near the estuary of rivers. Some species are found in inland rivers.  Usually like to "
    "live in groups, prey on fish, squid, etc.",
    "Dolphins are a collective name for a class of aquatic mammals in the Dolphin family. They are small or "
    "medium-sized toothed whales. They live widely in all oceans of the world. They are also distributed in inland "
    "seas and brackish waters near the estuary of rivers. Some species are found in inland rivers.  Usually like to "
    "live in groups, prey on fish, squid, etc."};
  i = 0;
  while (row.size() != 0) {
    auto text = row["context"];
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    ASSERT_OK(de_text->GetItemAt(&sv, {}));
    std::string ss(sv);
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());
    ASSERT_OK(iter->GetNextRow(&row));
     i++;
  }

  // Expect 2 samples.
  EXPECT_EQ(i, 2);

  iter->Stop();

  // all
  usage = "all";
  ds = SQuAD(dataset_dir, usage, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("answer_start"), row.end());
  std::vector<std::int32_t> expected = {52, 324, 33, -1};
  i = 0;
  while (row.size() != 0) {
    auto text = row["answer_start"];
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::uint32_t num;
    ASSERT_OK(de_text->GetItemAt(&num, {0}));
    EXPECT_EQ(num, expected[i]);
    ASSERT_OK(iter->GetNextRow(&row));
     i++;
  }

  // Expect 4 samples.
  EXPECT_EQ(i, 4);

  iter->Stop();
}

/// Feature: SQuADDataset.
/// Description: Test the distribution of SQuADDataset.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestSQuADDatasetDistribution) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSQuADDatasetDistribution.";

  // Create a SQuAD Dataset, with single SQuAD file.
  std::string dataset_dir = datasets_root_path_ + "/testSQuAD/SQuAD1";
  std::string usage = "train";
  std::shared_ptr<Dataset> ds = SQuAD(dataset_dir, usage, 0, ShuffleMode::kGlobal, 3, 0);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("question"), row.end());
  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["question"];
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  // Expect 1 samples.
  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: SQuADDataset.
/// Description: Test some failed cases of SQuADDataset.
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr.
TEST_F(MindDataTestPipeline, TestSQuADDatasetFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSQuADDatasetFail.";
  // Create a SQuAD Dataset
  std::string dataset_dir = datasets_root_path_ + "/testSQuAD/SQuAD1";
  std::string invalid_dataset_dir = "/NotExistedDir";
  std::string usage = "train";

  std::shared_ptr<Dataset> ds0 = SQuAD(invalid_dataset_dir, usage);
  EXPECT_NE(ds0, nullptr);
  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter0 = ds0->CreateIterator();
  // Expect failure: invalid SQuAD input.
  EXPECT_EQ(iter0, nullptr);

  std::shared_ptr<Dataset> ds1 = SQuAD(dataset_dir, "invalid_usage");
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid SQuAD input.
  EXPECT_EQ(iter1, nullptr);

  std::shared_ptr<Dataset> ds3 = SQuAD(dataset_dir, usage, 0, ShuffleMode::kGlobal, 2, 2);
  EXPECT_NE(ds3, nullptr);
  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid SQuAD input.
  EXPECT_EQ(iter3, nullptr);

  std::shared_ptr<Dataset> ds4 = SQuAD(dataset_dir, usage, -1, ShuffleMode::kGlobal);
  EXPECT_NE(ds4, nullptr);
  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter4 = ds4->CreateIterator();
  // Expect failure: invalid SQuAD input.
  EXPECT_EQ(iter4, nullptr);

  std::shared_ptr<Dataset> ds5 = SQuAD(dataset_dir, usage, 0, ShuffleMode::kGlobal, -1);
  EXPECT_NE(ds5, nullptr);
  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter5 = ds5->CreateIterator();
  // Expect failure: invalid SQuAD input.
  EXPECT_EQ(iter5, nullptr);

  std::shared_ptr<Dataset> ds6 = SQuAD(dataset_dir, usage, 0, ShuffleMode::kGlobal, 0, -1);
  EXPECT_NE(ds6, nullptr);
  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter6 = ds6->CreateIterator();
  // Expect failure: invalid SQuAD input.
  EXPECT_EQ(iter6, nullptr);
}

/// Feature: SQuADDataset.
/// Description: Test the Shuffle of SQuADDataset.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestSQuADDatasetShuffleFilesA) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSQuADDatasetShuffleFilesA.";
  // Test SQuAD Dataset with files shuffle, num_parallel_workers=1.

  // Set configuration.
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(135);
  GlobalContext::config_manager()->set_num_parallel_workers(1);

  // Create a SQuAD Dataset, with two text files, dev.json and train.json, in lexicographical order.
  // Note: train.json has 3 rows.
  // Note: dev.json has 2 rows.
  // Use default of all samples.
  // They have the same keywords.
  // Set shuffle to files shuffle.
  std::string dataset_dir = datasets_root_path_ + "/testSQuAD/SQuAD1";
  std::string usage = "all";
  std::shared_ptr<Dataset> ds = SQuAD(dataset_dir, usage, 0, ShuffleMode::kFiles);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("question"), row.end());
  std::vector<std::string> expected_result = {
    "Who is \"The Father of Modern Computers\"?", "When was John von Neumann's birth date?",
    "Where is John von Neumann's birthplace?",
    "Who is the author of \"The Mathematical Principles of Natural Philosophy\"?",
    "When was the publication year of \"The Mathematical Principles of Natural Philosophy\"?"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["question"];
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    ASSERT_OK(de_text->GetItemAt(&sv, {}));
    std::string ss(sv);
    // Compare against expected result.
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  // Expect 3 + 2 = 5 samples.
  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline.
  iter->Stop();

  // Restore configuration.
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

/// Feature: SQuADDataset.
/// Description: Test the Shuffle of SQuADDataset.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestSQuADDatasetShuffleFilesB) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSQuADDatasetShuffleFilesB.";
  // Test SQuAD Dataset with files shuffle, num_parallel_workers=1.

  // Set configuration.
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(135);
  GlobalContext::config_manager()->set_num_parallel_workers(1);

  // Create a SQuAD Dataset, with two text files, train.json and dev.json, in non-lexicographical order.
  // Note: train.json has 3 rows.
  // Note: dev.json has 2 rows.
  // Use default of all samples.
  // They have the same keywords.
  // Set shuffle to files shuffle.
  std::string dataset_dir = datasets_root_path_ + "/testSQuAD/SQuAD1";
  std::string usage = "all";
  std::shared_ptr<Dataset> ds = SQuAD(dataset_dir, usage, 0, ShuffleMode::kFiles);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("question"), row.end());
  std::vector<std::string> expected_result = {
    "Who is \"The Father of Modern Computers\"?", "When was John von Neumann's birth date?",
    "Where is John von Neumann's birthplace?",
    "Who is the author of \"The Mathematical Principles of Natural Philosophy\"?",
    "When was the publication year of \"The Mathematical Principles of Natural Philosophy\"?"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["question"];
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    ASSERT_OK(de_text->GetItemAt(&sv, {}));
    std::string ss(sv);
    // Compare against expected result.
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  // Expect 3 + 2 = 5 samples.
  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline.
  iter->Stop();

  // Restore configuration.
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

/// Feature: SQuADDataset.
/// Description: Test the global Shuffle of SQuADDataset.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestSQuADDatasetShuffleGlobal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSQuADDatasetShuffleGlobal.";
  // Test SQuAD Dataset with GLOBLE shuffle

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(135);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  // Create a SQuAD Dataset, with single SQuAD file.
  std::string dataset_dir = datasets_root_path_ + "/testSQuAD/SQuAD1";
  std::string usage = "train";
  std::shared_ptr<Dataset> ds = SQuAD(dataset_dir, usage, 0, ShuffleMode::kGlobal);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("question"), row.end());
  std::vector<std::string> expected_result = {"When was John von Neumann's birth date?",
                                              "Where is John von Neumann's birthplace?",
                                              "Who is \"The Father of Modern Computers\"?"};
  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["question"];
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    ASSERT_OK(de_text->GetItemAt(&sv, {}));
    std::string ss(sv);
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  // Expect 3 samples.
  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline.
  iter->Stop();

  // Restore configuration.
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}