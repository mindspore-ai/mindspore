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

/// Feature: WikiTextDataset
/// Description: Test WikiTextDataset basic usage
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestWikiTextDatasetBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestWikiTextDatasetBasic.";
  // Test WikiText Dataset with single text file and many default inputs

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(987);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  std::string dataset_dir = datasets_root_path_ + "/testWikiText";
  std::shared_ptr<Dataset> ds = WikiText(dataset_dir, "test", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::string> expected_result = {
    {" no it was black friday "},
    {" I am happy "},
    {" finish math homework "},
  };

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
    // Compare against expected result
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());

    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  // Expect 3 samples
  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();

  // Restore configuration
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

/// Feature: WikiTextDataset
/// Description: Test WikiTextDataset in pipeline mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestWikiTextDatasetBasicWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestWikiTextDatasetBasicWithPipeline.";
  // Test WikiText Dataset with single text file and many default inputs

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(987);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  std::string dataset_dir = datasets_root_path_ + "/testWikiText";
  std::shared_ptr<Dataset> ds1 = WikiText(dataset_dir, "test", 0, ShuffleMode::kFalse);
  std::shared_ptr<Dataset> ds2 = WikiText(dataset_dir, "test", 0, ShuffleMode::kFalse);
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
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("text"), row.end());

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["text"];
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  // Expect 15 samples
  EXPECT_EQ(i, 15);

  // Manually terminate the pipeline
  iter->Stop();

  // Restore configuration
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

/// Feature: WikiTextDataset
/// Description: Test iterator of WikiTextDataset with only the "text" column
/// Expectation: Get correct data
TEST_F(MindDataTestPipeline, TestWikiTextIteratorOneColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestWikiTextIteratorOneColumn.";
  // Test WikiText Dataset with single text file and many default inputs

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(987);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  std::string dataset_dir = datasets_root_path_ + "/testWikiText";
  std::shared_ptr<Dataset> ds = WikiText(dataset_dir, "test", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // Only select "text" column and drop others
  std::vector<std::string> columns = {"text"};
  std::shared_ptr<ProjectDataset> project_ds = ds->Project(columns);
  std::shared_ptr<Iterator> iter = project_ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  std::vector<int64_t> expect_image = {1};

  uint64_t i = 0;
  while (row.size() != 0) {
    for (auto &v : row) {
      MS_LOG(INFO) << "image shape:" << v.Shape();
      EXPECT_EQ(expect_image, v.Shape());
    }
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: WikiTextDataset
/// Description: Test iterator of WikiTextDataset with wrong column
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestWikiTextIteratorWrongColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestWikiTextIteratorWrongColumn.";
  // Test WikiText Dataset with single text file and many default inputs

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(987);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  std::string dataset_dir = datasets_root_path_ + "/testWikiText";
  std::shared_ptr<Dataset> ds = WikiText(dataset_dir, "test", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Pass wrong column name
  std::vector<std::string> columns = {"digital"};
  std::shared_ptr<ProjectDataset> project_ds = ds->Project(columns);
  std::shared_ptr<Iterator> iter = project_ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: WikiTextDataset
/// Description: Test WikiTextDataset Getters method
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestWikiTextGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestWikiTextGetters.";
  // Test WikiText Dataset with single text file and many default inputs

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(987);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  std::string dataset_dir = datasets_root_path_ + "/testWikiText";
  std::shared_ptr<Dataset> ds = WikiText(dataset_dir, "test", 2, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::vector<std::string> column_names = {"text"};
  EXPECT_EQ(ds->GetDatasetSize(), 2);
  EXPECT_EQ(ds->GetColumnNames(), column_names);

  ds = WikiText(dataset_dir, "test", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 3);

  std::vector<DataType> types = ToDETypes(ds->GetOutputTypes());
  std::vector<TensorShape> shapes = ToTensorShapeVec(ds->GetOutputShapes());
  EXPECT_EQ(types.size(), 1);
  EXPECT_EQ(types[0].ToString(), "string");
  EXPECT_EQ(shapes.size(), 1);
  EXPECT_EQ(shapes[0].ToString(), "<>");
  // Restore configuration
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

/// Feature: WikiTextDataset
/// Description: Test WikiTextDataset with invalid samplers=-1
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestWikiTextDatasetFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestWikiTextDatasetFail1.";

  // Create a WikiText Dataset
  // with invalid samplers=-1
  std::string dataset_dir = datasets_root_path_ + "/testWikiText";
  std::shared_ptr<Dataset> ds = WikiText(dataset_dir, "test", -1, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: WikiText number of samples cannot be negative
  EXPECT_EQ(iter, nullptr);
}

/// Feature: WikiTextDataset
/// Description: Test WikiTextDataset with empty dataset_files input
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestWikiTextDatasetFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestWikiTextDatasetFail2.";

  // Attempt to create a WikiText Dataset
  // with wrongful empty dataset_files input
  std::string dataset_dir = datasets_root_path_ + "/testWikiText";
  std::shared_ptr<Dataset> ds = WikiText("123", "test", 2, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: dataset_dir is not specified
  EXPECT_EQ(iter, nullptr);
}

/// Feature: WikiTextDataset
/// Description: Test WikiTextDataset with non-existent dataset_files input
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestWikiTextDatasetFail3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestWikiTextDatasetFail3.";

  // Create a WikiText Dataset
  // with non-existent dataset_files input
  std::string dataset_dir = datasets_root_path_ + "/testWikiText";
  std::shared_ptr<Dataset> ds = WikiText(dataset_dir, "asd", 2, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid usage
  EXPECT_EQ(iter, nullptr);
}

/// Feature: WikiTextDataset
/// Description: Test WikiTextDataset with empty string dataset_files input
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestWikiTextDatasetFail4) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestWikiTextDatasetFail4.";

  // Create a WikiText Dataset
  // with empty string dataset_files input
  std::string dataset_dir = datasets_root_path_ + "/testWikiText";
  std::shared_ptr<Dataset> ds = WikiText("", "test", 2, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: specified dataset_files does not exist
  EXPECT_EQ(iter, nullptr);
}

/// Feature: WikiTextDataset
/// Description: Test WikiTextDataset with invalid num_shards=0
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestWikiTextDatasetFail5) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestWikiTextDatasetFail5.";

  // Create a WikiText Dataset
  // with invalid num_shards=0 value
  std::string dataset_dir = datasets_root_path_ + "/testWikiText";
  std::shared_ptr<Dataset> ds = WikiText(dataset_dir, "test", 2, ShuffleMode::kFalse, 0);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Number of shards cannot be <=0
  EXPECT_EQ(iter, nullptr);
}

/// Feature: WikiTextDataset
/// Description: Test WikiTextDataset with invalid shard_id=-1
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestWikiTextDatasetFail6) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestWikiTextDatasetFail6.";

  // Create a WikiText Dataset
  // with invalid shard_id=-1 value
  std::string dataset_dir = datasets_root_path_ + "/testWikiText";
  std::shared_ptr<Dataset> ds = WikiText(dataset_dir, "test", 2, ShuffleMode::kFalse, 1, -1);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: shard_id cannot be negative
  EXPECT_EQ(iter, nullptr);
}

/// Feature: WikiTextDataset
/// Description: Test WikiTextDataset with invalid shard_id=2 and num_shards=2 combination
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestWikiTextDatasetFail7) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestWikiTextDatasetFail7.";

  // Create a WikiText Dataset
  // with invalid shard_id=2 and num_shards=2 combination
  std::string dataset_dir = datasets_root_path_ + "/testWikiText";
  std::shared_ptr<Dataset> ds = WikiText(dataset_dir, "test", 2, ShuffleMode::kFalse, 2, 2);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Cannot have shard_id >= num_shards
  EXPECT_EQ(iter, nullptr);
}

/// Feature: WikiTextDataset
/// Description: Test WikiTextDataset with ShuffleMode::kFalse
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestWikiTextDatasetShuffleFalse) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestWikiTextDatasetShuffleFalse.";

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(246);
  GlobalContext::config_manager()->set_num_parallel_workers(2);

  std::string dataset_dir = datasets_root_path_ + "/testWikiText";
  std::shared_ptr<Dataset> ds = WikiText(dataset_dir, "all", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::string> expected_result = {
    {" no it was black friday "},
    {" go to china "},
    {" I am happy "},
    {" I lova MindSpore "},
    {" finish math homework "},
    {" black white grapes "},
    {" just ahead of them there was a huge fissure "},
    {" zhejiang, china "},
    {" MindSpore Ascend "},
  };

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
    // Compare against expected result
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());

    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  // Expect 9 samples
  EXPECT_EQ(i, 9);

  // Manually terminate the pipeline
  iter->Stop();

  // Restore configuration
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

/// Feature: WikiTextDataset
/// Description: Test WikiTextDataset with ShuffleMode::kFiles
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestWikiTextDatasetShuffleFilesA) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestWikiTextDatasetShuffleFilesA.";

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(654);
  GlobalContext::config_manager()->set_num_parallel_workers(1);

  std::string dataset_dir = datasets_root_path_ + "/testWikiText";
  std::shared_ptr<Dataset> ds = WikiText(dataset_dir, "all", 0, ShuffleMode::kFiles);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::string> expected_result = {
    {" go to china "},
    {" I lova MindSpore "},
    {" black white grapes "},
    {" no it was black friday "},
    {" I am happy "},
    {" finish math homework "},
    {" just ahead of them there was a huge fissure "},
    {" zhejiang, china "},
    {" MindSpore Ascend "},
  };

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
    // Compare against expected result
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());

    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  // Expect 9 samples
  EXPECT_EQ(i, 9);

  // Manually terminate the pipeline
  iter->Stop();

  // Restore configuration
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

/// Feature: WikiTextDataset
/// Description: Test WikiTextDataset with ShuffleMode::kGlobal
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestWikiTextDatasetShuffleGlobal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestWikiTextDatasetShuffleGlobal.";

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
  std::string dataset_dir = datasets_root_path_ + "/testWikiText";
  std::shared_ptr<Dataset> ds = WikiText(dataset_dir, "all", 0, ShuffleMode::kGlobal);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::string> expected_result = {
    {" MindSpore Ascend "},
    {" go to china "},
    {" I am happy "},
    {" no it was black friday "},
    {" just ahead of them there was a huge fissure "},
    {" zhejiang, china "},
    {" finish math homework "},
    {" I lova MindSpore "},
    {" black white grapes "},
  };

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
    // Compare against expected result
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());

    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  // Expect 9 samples
  EXPECT_EQ(i, 9);

  // Manually terminate the pipeline
  iter->Stop();

  // Restore configuration
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}
