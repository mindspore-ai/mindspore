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
#include "minddata/dataset/include/dataset/datasets.h"
#include "minddata/dataset/engine/ir/datasetops/source/yahoo_answers_node.h"

using namespace mindspore::dataset;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

/// Feature: YahooAnswersDataset.
/// Description: Read test data.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestYahooAnswersDatasetBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestYahooAnswersDatasetBasic.";

  // Create a YahooAnswers Dataset.
  std::string folder_path = datasets_root_path_ + "/testYahooAnswers/";
  std::vector<std::string> column_names = {"class", "title", "content", "answer"};
  std::shared_ptr<Dataset> ds = YahooAnswers(folder_path, "test", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterator the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("class"), row.end());

  std::vector<std::vector<std::string>> expected_result = {
    {"4","My pet","My pet is a toy bear.","He is white."},
    {"1","My favourite seasion","My favorite season is summer.", "In summer it is often sunny and hot."}};
  uint64_t i = 0;
  while (row.size() != 0) {
    for (int j = 0; j < column_names.size(); j++) {
      auto text = row[column_names[j]];
      std::shared_ptr<Tensor> de_text;
      ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
      std::string_view sv;
      ASSERT_OK(de_text->GetItemAt(&sv, {}));
      std::string ss(sv);
      EXPECT_STREQ(ss.c_str(), expected_result[i][j].c_str());
    }
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  // Expect 2 samples.
  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: YahooAnswersDataset.
/// Description: Read train data and test data.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestYahooAnswersDatasetUsageAll) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestYahooAnswersDatasetUsageAll.";

  std::string folder_path = datasets_root_path_ + "/testYahooAnswers/";
  std::vector<std::string> column_names = {"class", "title", "content", "answer"};
  std::shared_ptr<Dataset> ds = YahooAnswers(folder_path, "all", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterator the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("class"), row.end());

  std::vector<std::vector<std::string>> expected_result = {
    {"4","My pet","My pet is a toy bear.","He is white."},
    {"3","My Chinese teacher","I have a Chinese Teacher.","She is from LanCha."},
    {"1","My favourite seasion","My favorite season is summer.","In summer it is often sunny and hot."},
    {"5","Last weekend","We played games, we were very happy.","I visited my friends."},
    {"1","A Happy Day","Last Sunday, I visited my grandmother.","I counted the flowers."},
    {"8","My Good Friend","She lives in China.","He likes listening to music."}};
  uint64_t i = 0;
  while (row.size() != 0) {
    for (int j = 0; j < column_names.size(); j++) {
      auto text = row[column_names[j]];
      std::shared_ptr<Tensor> de_text;
      ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
      std::string_view sv;
      ASSERT_OK(de_text->GetItemAt(&sv, {}));
      std::string ss(sv);
      EXPECT_STREQ(ss.c_str(), expected_result[i][j].c_str());
    }
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  // Expect 6 samples.
  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: YahooAnswersDataset.
/// Description: Includes tests for shape, type, size.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestYahooAnswersDatasetGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestYahooAnswersDatasetGetters.";

  std::string folder_path = datasets_root_path_ + "/testYahooAnswers/";
  std::shared_ptr<Dataset> ds = YahooAnswers(folder_path, "test", 0, ShuffleMode::kFalse);
  std::vector<std::string> column_names = {"class", "title", "content", "answer"};
  EXPECT_NE(ds, nullptr);

  std::vector<DataType> types = ToDETypes(ds->GetOutputTypes());
  std::vector<TensorShape> shapes = ToTensorShapeVec(ds->GetOutputShapes());
  EXPECT_EQ(types.size(), 4);
  EXPECT_EQ(types[0].ToString(), "string");
  EXPECT_EQ(types[1].ToString(), "string");
  EXPECT_EQ(types[2].ToString(), "string");
  EXPECT_EQ(types[3].ToString(), "string");
  EXPECT_EQ(shapes.size(), 4);
  EXPECT_EQ(shapes[0].ToString(), "<>");
  EXPECT_EQ(shapes[1].ToString(), "<>");
  EXPECT_EQ(shapes[2].ToString(), "<>");
  EXPECT_EQ(shapes[3].ToString(), "<>");
  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);
  EXPECT_EQ(ds->GetDatasetSize(), 2);
  EXPECT_EQ(ds->GetColumnNames(), column_names);
}

/// Feature: YahooAnswersDataset.
/// Description: Read 2 samples from train file.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestYahooAnswersDatasetNumSamples) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestYahooAnswersDatasetNumSamples.";

  // Create a YahooAnswersDataset.
  std::string folder_path = datasets_root_path_ + "/testYahooAnswers/";
  std::vector<std::string> column_names = {"class", "title", "content", "answer"};
  std::shared_ptr<Dataset> ds = YahooAnswers(folder_path, "train", 2, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("class"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
    {"3","My Chinese teacher","I have a Chinese Teacher.","She is from LanCha."},
    {"5","Last weekend","We played games, we were very happy.","I visited my friends."}};

  uint64_t i = 0;
  while (row.size() != 0) {
    for (int j = 0; j < column_names.size(); j++) {
      auto text = row[column_names[j]];
      std::shared_ptr<Tensor> de_text;
      ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
      std::string_view sv;
      ASSERT_OK(de_text->GetItemAt(&sv, {}));
      std::string ss(sv);
      EXPECT_STREQ(ss.c_str(), expected_result[i][j].c_str());
    }
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  // Expect 2 samples.
  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: YahooAnswersDataset.
/// Description: Test in a distributed state.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestYahooAnswersDatasetDistribution) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestYahooAnswersDatasetDistribution.";

  // Create a YahooAnswersDataset.
  std::string folder_path = datasets_root_path_ + "/testYahooAnswers/";
  std::vector<std::string> column_names = {"class", "title", "content", "answer"};
  std::shared_ptr<Dataset> ds = YahooAnswers(folder_path, "train", 0, ShuffleMode::kFalse, 2, 0);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("class"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
    {"3","My Chinese teacher","I have a Chinese Teacher.","She is from LanCha."},
    {"5","Last weekend","We played games, we were very happy.","I visited my friends."}};

  uint64_t i = 0;
  while (row.size() != 0) {
    for (int j = 0; j < column_names.size(); j++) {
      auto text = row[column_names[j]];
      std::shared_ptr<Tensor> de_text;
      ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
      std::string_view sv;
      ASSERT_OK(de_text->GetItemAt(&sv, {}));
      std::string ss(sv);
      EXPECT_STREQ(ss.c_str(), expected_result[i][j].c_str());
    }
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  // Expect 2 samples.
  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: YahooAnswersDataset.
/// Description: Test with invalid input.
/// Expectation: Throw error messages when certain errors occur.
TEST_F(MindDataTestPipeline, TestYahooAnswersDatasetFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestYahooAnswersDatasetFail.";
  // Create a YahooAnswers Dataset.
  std::string folder_path = datasets_root_path_ + "/testYahooAnswers/";
  std::string invalid_folder_path = "./NotExistPath";
  std::vector<std::string> column_names = {"class", "title", "content", "answer"};

  // Test invalid folder_path.
  std::shared_ptr<Dataset> ds0 = YahooAnswers(invalid_folder_path, "all", -1, ShuffleMode::kFalse);
  EXPECT_NE(ds0, nullptr);
  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter0 = ds0->CreateIterator();
  // Expect failure: invalid YahooAnswers input.
  EXPECT_EQ(iter0, nullptr);

  // Test invalid usage.
  std::shared_ptr<Dataset> ds1 = YahooAnswers(folder_path, "invalid_usage", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid YahooAnswers input.
  EXPECT_EQ(iter1, nullptr);

  // Test invalid num_samples < -1.
  std::shared_ptr<Dataset> ds2 = YahooAnswers(folder_path, "all", -1, ShuffleMode::kFalse);
  EXPECT_NE(ds2, nullptr);
  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid YahooAnswers input.
  EXPECT_EQ(iter2, nullptr);

  // Test invalid num_shards < 1.
  std::shared_ptr<Dataset> ds3 = YahooAnswers(folder_path, "all", 0, ShuffleMode::kFalse, 0);
  EXPECT_NE(ds3, nullptr);
  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid YahooAnswers input.
  EXPECT_EQ(iter3, nullptr);

  // Test invalid shard_id >= num_shards.
  std::shared_ptr<Dataset> ds4 = YahooAnswers(folder_path, "all", 0, ShuffleMode::kFalse, 2, 2);
  EXPECT_NE(ds4, nullptr);
  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter4 = ds4->CreateIterator();
  // Expect failure: invalid YahooAnswers input.
  EXPECT_EQ(iter4, nullptr);
}

/// Feature: YahooAnswersDataset.
/// Description: Read data with pipeline from test file.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestYahooAnswersDatasetWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestYahooAnswersDatasetWithPipeline.";

  // Create two YahooAnswers Dataset, with single YahooAnswers file.
  std::string dataset_dir = datasets_root_path_ + "/testYahooAnswers/";

  std::shared_ptr<Dataset> ds1 = YahooAnswers(dataset_dir, "test", 0, ShuffleMode::kFalse);
  std::shared_ptr<Dataset> ds2 = YahooAnswers(dataset_dir, "test", 0, ShuffleMode::kFalse);
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
  std::vector<std::string> column_project = {"class"};
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

  EXPECT_NE(row.find("class"), row.end());
  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["class"];
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  // Expect 10 samples.
  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: YahooAnswersDataset.
/// Description: Test with shuffle files.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestYahooAnswersDatasetShuffleFilesA) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestYahooAnswersDatasetShuffleFilesA.";

  // Set configuration.
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(130);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  std::string folder_path = datasets_root_path_ + "/testYahooAnswers/";
  std::vector<std::string> column_names = {"class", "title", "content", "answer"};
  std::shared_ptr<Dataset> ds = YahooAnswers(folder_path, "all", 0, ShuffleMode::kFiles);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("class"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
    {"3","My Chinese teacher","I have a Chinese Teacher.","She is from LanCha."},
    {"4","My pet","My pet is a toy bear.","He is white."},
    {"5","Last weekend","We played games, we were very happy.","I visited my friends."},
    {"1","My favourite seasion","My favorite season is summer.","In summer it is often sunny and hot."},
    {"1","A Happy Day","Last Sunday, I visited my grandmother.","I counted the flowers."},
    {"8","My Good Friend","She lives in China.","He likes listening to music."}};

  uint64_t i = 0;
  while (row.size() != 0) {
    for (int j = 0; j < column_names.size(); j++) {
      auto text = row[column_names[j]];
      std::shared_ptr<Tensor> de_text;
      ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
      std::string_view sv;
      ASSERT_OK(de_text->GetItemAt(&sv, {}));
      std::string ss(sv);
      EXPECT_STREQ(ss.c_str(), expected_result[i][j].c_str());
    }
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  // Expect 6 samples.
  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline.
  iter->Stop();

  // Restore configuration.
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

/// Feature: YahooAnswersDataset.
/// Description: Test with global shuffle.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestYahooAnswersDatasetShuffleGlobal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestYahooAnswersDatasetShuffleFilesGlobal.";

  // Set configuration.
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(130);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  std::string folder_path = datasets_root_path_ + "/testYahooAnswers/";
  std::vector<std::string> column_names = {"class", "title", "content", "answer"};
  std::shared_ptr<Dataset> ds = YahooAnswers(folder_path, "test", 0, ShuffleMode::kGlobal);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("class"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
    {"4","My pet","My pet is a toy bear.","He is white."},
    {"1","My favourite seasion","My favorite season is summer.", "In summer it is often sunny and hot."}};

  uint64_t i = 0;
  while (row.size() != 0) {
    for (int j = 0; j < column_names.size(); j++) {
      auto text = row[column_names[j]];
      std::shared_ptr<Tensor> de_text;
      ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
      std::string_view sv;
      ASSERT_OK(de_text->GetItemAt(&sv, {}));
      std::string ss(sv);
      EXPECT_STREQ(ss.c_str(), expected_result[i][j].c_str());
    }
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  // Expect 2 samples.
  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline.
  iter->Stop();

  // Restore configuration.
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}
