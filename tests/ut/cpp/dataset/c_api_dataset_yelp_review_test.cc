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

class MindDataTestPipeline : public UT::DatasetOpTesting {
protected:
};

/// Feature: YelpReviewDataset
/// Description: Test YelpReviewDataset basic usage with polarity dataset
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestYelpReviewPolarityDatasetBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestYelpReviewPolarityDatasetBasic.";

  std::string dataset_dir = datasets_root_path_ + "/testYelpReview/polarity";
  std::vector<std::string> column_names = {"label", "text"};

  std::shared_ptr<Dataset> ds = YelpReview(dataset_dir, "test", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("label"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
      {"2", "\\\"Yelp\\\" service was very good.\\n"},
      {"1", "\\\"Yelp\\\" service was very bad.\\n"}
  };

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

/// Feature: YelpReviewDataset
/// Description: Test YelpReviewDataset basic usage with full dataset
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestYelpReviewFullDatasetBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestYelpReviewFullDatasetBasic.";

  std::string dataset_dir = datasets_root_path_ + "/testYelpReview/full";
  std::vector<std::string> column_names = {"label", "text"};

  std::shared_ptr<Dataset> ds = YelpReview(dataset_dir, "test", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("label"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
      {"1", "\\\"YelpFull\\\" service was very good.\\n"},
      {"1", "\\\"YelpFull\\\" service was very bad.\\n"}
  };

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

/// Feature: YelpReviewDataset
/// Description: Test YelpReviewDataset with pipeline mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestYelpReviewDatasetWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestYelpReviewDatasetWithPipeline.";

  // Create two STL10 Dataset
  std::string dataset_dir = datasets_root_path_ + "/testYelpReview/polarity";
  std::shared_ptr<Dataset> ds1 = YelpReview(dataset_dir, "test", 0, ShuffleMode::kFalse);
  std::shared_ptr<Dataset> ds2 = YelpReview(dataset_dir, "test", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds1, nullptr);
  EXPECT_NE(ds2, nullptr);

  // Create two Repeat operation on ds
  int32_t repeat_num = 1;
  ds1 = ds1->Repeat(repeat_num);
  EXPECT_NE(ds1, nullptr);
  repeat_num = 1;
  ds2 = ds2->Repeat(repeat_num);
  EXPECT_NE(ds2, nullptr);

  // Create two Project operation on ds
  std::vector<std::string> column_project = {"label", "text"};
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

  EXPECT_NE(row.find("label"), row.end());
  EXPECT_NE(row.find("text"), row.end());

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: YelpReviewDataset
/// Description: Test iterator of YelpReviewDataset with only the text column
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestYelpReviewDatasetIteratorOneColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestYelpReviewIteratorOneColumn.";
  // Create a YelpReview dataset
  std::string dataset_dir = datasets_root_path_ + "/testYelpReview/polarity";
  std::shared_ptr<Dataset> ds = YelpReview(dataset_dir, "test", 0, ShuffleMode::kFalse);
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
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  std::vector<int64_t> expect_shape = {1, 1, 16000};

  uint64_t i = 0;
  while (row.size() != 0) {
      auto audio = row["text"];
      MS_LOG(INFO) << "Tensor text shape: " << audio.Shape();
      ASSERT_OK(iter->GetNextRow(&row));
      i++;
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: YelpReviewDataset
/// Description: Test iterator of YelpReviewDataset with wrong column
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestYelpReviewDatasetIteratorWrongColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestYelpReviewDatasetIteratorWrongColumn.";
  // Create a YelpReview dataset
  std::string dataset_dir = datasets_root_path_ + "/testYelpReview/polarity";
  std::shared_ptr<Dataset> ds = YelpReview(dataset_dir, "test", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Pass wrong column name
  std::vector<std::string> columns = {"digital"};
  std::shared_ptr<ProjectDataset> project_ds = ds->Project(columns);
  std::shared_ptr<Iterator> iter = project_ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: YelpReviewDataset
/// Description: Test YelpReviewDataset with all usage and polarity dataset
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestYelpReviewDatasetUsageAll) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestYelpReviewDatasetUsageAll.";

  std::string dataset_dir = datasets_root_path_ + "/testYelpReview/polarity";
  std::vector<std::string> column_names = {"label", "text"};

  std::shared_ptr<Dataset> ds = YelpReview(dataset_dir, "all" , 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("label"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
      {"1", "The food today is terrible.\\n"},
      {"2", "\\\"Yelp\\\" service was very good.\\n"},
      {"2", "The food is delicious today.\\n"},
      {"1", "\\\"Yelp\\\" service was very bad.\\n"},
      {"1", "Today's drink tastes bad.\\n"}
  };

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

  // Expect 5 samples.
  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: YelpReviewDataset
/// Description: Test YelpReviewDataset Getters method
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestYelpReviewDatasetGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestYelpReviewDatasetGetters.";

  std::string dataset_dir = datasets_root_path_ + "/testYelpReview/polarity";
  std::shared_ptr<Dataset> ds = YelpReview(dataset_dir, "test", 0, ShuffleMode::kFalse);
  std::vector<std::string> column_names = {"label", "text"};
  EXPECT_NE(ds, nullptr);

  std::vector<DataType> types = ToDETypes(ds->GetOutputTypes());
  std::vector<TensorShape> shapes = ToTensorShapeVec(ds->GetOutputShapes());
  EXPECT_EQ(types.size(), 2);
  EXPECT_EQ(types[0].ToString(), "string");
  EXPECT_EQ(types[1].ToString(), "string");
  EXPECT_EQ(shapes.size(), 2);
  EXPECT_EQ(shapes[0].ToString(), "<>");
  EXPECT_EQ(shapes[1].ToString(), "<>");
  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);
  EXPECT_EQ(ds->GetDatasetSize(), 2);
  EXPECT_EQ(ds->GetColumnNames(), column_names);
}

/// Feature: YelpReviewDataset
/// Description: Test YelpReviewDataset with num_samples=2
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestYelpReviewDatasetNumSamples) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestYelpReviewDatasetNumSamples.";

  std::string dataset_dir = datasets_root_path_ + "/testYelpReview/polarity";
  std::vector<std::string> column_names = {"label", "text"};

  std::shared_ptr<Dataset> ds = YelpReview(dataset_dir, "test", 2, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("label"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
      {"2", "\\\"Yelp\\\" service was very good.\\n"},
      {"1", "\\\"Yelp\\\" service was very bad.\\n"}
  };

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

/// Feature: YelpReviewDataset
/// Description: Test YelpReviewDataset in distributed state (with num_shards and shard_id)
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestYelpReviewDatasetDistribution) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestYelpReviewDatasetDistribution.";

  // Create a YelpReviewDataset.
  std::string dataset_dir = datasets_root_path_ + "/testYelpReview/polarity";
  std::vector<std::string> column_names = {"label", "text"};

  std::shared_ptr<Dataset> ds = YelpReview(dataset_dir, "test", 0, ShuffleMode::kFalse, 2, 0);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("label"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
      {"2", "\\\"Yelp\\\" service was very good.\\n"},
      {"1", "\\\"Yelp\\\" service was very bad.\\n"}
  };

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

  // Expect 1 samples.
  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: YelpReviewDataset
/// Description: Test YelpReviewDataset with invalid inputs
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestYelpReviewDatasetFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestYelpReviewDatasetFail.";

  std::string dataset_dir = datasets_root_path_ + "/testYelpReview/polarity";

  std::shared_ptr<Dataset> ds0 = YelpReview("NotExistFile", "test", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds0, nullptr);
  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter0 = ds0->CreateIterator();
  // Expect failure: invalid YelpReview input.
  EXPECT_EQ(iter0, nullptr);

  // Create a YelpReview Dataset with invalid usage.
  std::shared_ptr<Dataset> ds1 = YelpReview(dataset_dir, "invalid_usage", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid YelpReview input.
  EXPECT_EQ(iter1, nullptr);

  // Test invalid num_samples < -1.
  std::shared_ptr<Dataset> ds2 = YelpReview(dataset_dir, "test", -1, ShuffleMode::kFalse);
  EXPECT_NE(ds2, nullptr);
  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid YelpReview input.
  EXPECT_EQ(iter2, nullptr);

  // Test invalid num_shards < 1.
  std::shared_ptr<Dataset> ds3 = YelpReview(dataset_dir, "test", 0, ShuffleMode::kFalse, 0);
  EXPECT_NE(ds3, nullptr);
  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid YelpReview input.
  EXPECT_EQ(iter3, nullptr);

  // Test invalid shard_id >= num_shards.
  std::shared_ptr<Dataset> ds4 = YelpReview(dataset_dir,"test", 0, ShuffleMode::kFalse, 2, 2);
  EXPECT_NE(ds4, nullptr);
  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter4 = ds4->CreateIterator();
  // Expect failure: invalid YelpReview input.
  EXPECT_EQ(iter4, nullptr);
}

/// Feature: YelpReviewDataset
/// Description: Test YelpReviewDataset basic usage with polarity dataset with pipeline mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestYelpReviewDatasetBasicWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestYelpReviewDatasetBasicWithPipeline.";

  // Create two YelpReview Dataset, with single YelpReview file.
  std::string dataset_dir = datasets_root_path_ + "/testYelpReview/polarity";

  std::shared_ptr<Dataset> ds1 = YelpReview(dataset_dir, "test", 0, ShuffleMode::kFalse);
  std::shared_ptr<Dataset> ds2 = YelpReview(dataset_dir, "test", 0, ShuffleMode::kFalse);
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
  std::vector<std::string> column_project = {"label"};
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

  EXPECT_NE(row.find("label"), row.end());
  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["label"];
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  // Expect 10 samples.
  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: YelpReviewDataset
/// Description: Test YelpReviewDataset with ShuffleMode::kFiles
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TesYelpReviewDatasetShuffleFilesA) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TesYelpReviewDatasetShuffleFilesA.";

  // Set configuration.
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(130);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  std::string dataset_dir = datasets_root_path_ + "/testYelpReview/polarity";
  std::vector<std::string> column_names = {"label", "text"};

  std::shared_ptr<Dataset> ds = YelpReview(dataset_dir, "all" , 0, ShuffleMode::kFiles);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("label"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
      {"2", "\\\"Yelp\\\" service was very good.\\n"},
      {"1", "The food today is terrible.\\n"},
      {"1", "\\\"Yelp\\\" service was very bad.\\n"},
      {"2", "The food is delicious today.\\n"},
      {"1", "Today's drink tastes bad.\\n"}
  };

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
  // Expect 5 samples.
  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline.
  iter->Stop();

  // Restore configuration.
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

/// Feature: YelpReviewDataset
/// Description: Test YelpReviewDataset with ShuffleMode::kGlobal
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TesYelpReviewDatasetShuffleFilesGlobal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TesYelpReviewDatasetShuffleFilesGlobal.";

  // Set configuration.
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(130);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  std::string dataset_dir = datasets_root_path_ + "/testYelpReview/polarity";
  std::vector<std::string> column_names = {"label", "text"};

  std::shared_ptr<Dataset> ds = YelpReview(dataset_dir, "train" , 0, ShuffleMode::kGlobal);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("label"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
      {"1", "The food today is terrible.\\n"},
      {"1", "Today's drink tastes bad.\\n"},
      {"2", "The food is delicious today.\\n"}
  };

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

  // Expect 3 samples.
  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline.
  iter->Stop();

  // Restore configuration.
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}
