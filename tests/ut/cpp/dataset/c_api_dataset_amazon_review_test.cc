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
#include "minddata/dataset/engine/ir/datasetops/source/amazon_review_node.h"

using namespace mindspore::dataset;

class MindDataTestPipeline : public UT::DatasetOpTesting {
protected:
};

/// Feature: AmazonReview
/// Description: Read AmazonReviewPolarityDataset data and get data.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestAmazonReviewPolarityDatasetBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAmazonReviewPolarityDatasetBasic.";

  std::string dataset_dir = datasets_root_path_ + "/testAmazonReview/polarity";
  std::vector<std::string> column_names = {"label", "title", "content"};

  std::shared_ptr<Dataset> ds = AmazonReview(dataset_dir, "test", 0, ShuffleMode::kFalse);
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
      {"1", "DVD", "It is very good!"},
      {"2", "Book", "I would read it again lol."}
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

/// Feature: AmazonReview
/// Description: Read AmazonReviewFullDataset data and get data.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestAmazonReviewFullDatasetBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAmazonReviewFullDatasetBasic.";

  std::string dataset_dir = datasets_root_path_ + "/testAmazonReview/full";
  std::vector<std::string> column_names = {"label", "title", "content"};
  
  std::shared_ptr<Dataset> ds = AmazonReview(dataset_dir, "test", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("label"), row.end());
  std::vector<std::vector<std::string>> expected_result = {                
    {"1", "amazing", "unlimited buyback!"},
    {"4", "delightful", "a funny book!"},
    {"3", "Small", "It is a small ball!"}
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

  // Expect 3 samples
  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: AmazonReview(usage=all).
/// Description: Read train data and test data.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestAmazonReviewDatasetUsageAll) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAmazonReviewDatasetUsageAll.";

  std::string dataset_dir = datasets_root_path_ + "/testAmazonReview/full";
  std::vector<std::string> column_names = {"label", "title", "content"};

  std::shared_ptr<Dataset> ds = AmazonReview(dataset_dir, "all" , 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("label"), row.end());
    std::vector<std::vector<std::string>> expected_result = {
    {"1", "amazing", "unlimited buyback!"},
    {"3", "Satisfied", "good quality."},
    {"4", "delightful", "a funny book!"},
    {"5", "good", "This is an very good product."},
    {"3", "Small", "It is a small ball!"},
    {"1", "bad", "work badly."}
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

  // Expect 6 samples
  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: AmazonReview
/// Description: Test Getter methods
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestAmazonReviewGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAmazonReviewGetters.";

  std::string dataset_dir = datasets_root_path_ + "/testAmazonReview/full";
  std::shared_ptr<Dataset> ds = AmazonReview(dataset_dir, "test", 0, ShuffleMode::kFalse);
  std::vector<std::string> column_names = {"label", "title", "content"};
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds-> GetDatasetSize(),3);
  EXPECT_EQ(ds->GetColumnNames(),column_names);
}

/// Feature: AmazonReview(num_samples = 3).
/// Description: Test whether the interface meets expectations when NumSamples is equal to 2.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestAmazonReviewNumSamples) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAmazonReviewNumSamples.";

  std::string dataset_dir = datasets_root_path_ + "/testAmazonReview/full";
  std::vector<std::string> column_names = {"label", "title", "content"};

  std::shared_ptr<Dataset> ds = AmazonReview(dataset_dir, "test", 3, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("label"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
    {"1", "amazing", "unlimited buyback!"},
    {"4", "delightful", "a funny book!"},
    {"3", "Small", "It is a small ball!"}
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
}

/// Feature: AmazonReview
/// Description: Test interface in a distributed state.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestAmazonReviewDatasetDistribution) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAmazonReviewDatasetDistribution.";

  std::string dataset_dir = datasets_root_path_ + "/testAmazonReview/full";
  std::vector<std::string> column_names = {"label", "title", "content"};

  std::shared_ptr<Dataset> ds = AmazonReview(dataset_dir, "test", 0, ShuffleMode::kFalse, 2, 0);
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
    {"1", "amazing", "unlimited buyback!"},
    {"4", "delightful", "a funny book!"},
    {"3", "Small", "It is a small ball!"}
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

/// Feature: AmazonReview
/// Description: Test the wrong input.
/// Expectation: Unable to read in data.
TEST_F(MindDataTestPipeline, TestAmazonReviewDatasetFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAmazonReviewDatasetFail.";

  std::string dataset_dir = datasets_root_path_ + "/testAmazonReview/full";
  std::string invalid_csv_file = "./NotExistFile";
  std::vector<std::string> column_names = {"label", "title", "content"};

  std::shared_ptr<Dataset> ds0 = AmazonReview("", "test", 0);
  EXPECT_NE(ds0, nullptr);
  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter0 = ds0->CreateIterator();
  // Expect failure: invalid AmazonReview input.
  EXPECT_EQ(iter0, nullptr);

  // Create a AmazonReview Dataset with invalid usage.
  std::shared_ptr<Dataset> ds1 = AmazonReview(invalid_csv_file);
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid SogouNews input.
  EXPECT_EQ(iter1, nullptr);

  // Test invalid num_samples < -1.
  std::shared_ptr<Dataset> ds2 = AmazonReview(dataset_dir, "test", -1, ShuffleMode::kFalse);
  EXPECT_NE(ds2, nullptr);
  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid AmazonReviewNews input.
  EXPECT_EQ(iter2, nullptr);

  // Test invalid num_shards < 1.
  std::shared_ptr<Dataset> ds3 = AmazonReview(dataset_dir, "test", 0, ShuffleMode::kFalse, 0);
  EXPECT_NE(ds3, nullptr);
  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid AmazonReview input.
  EXPECT_EQ(iter3, nullptr);

  // Test invalid shard_id >= num_shards.
  std::shared_ptr<Dataset> ds4 = AmazonReview(dataset_dir, "test", 0, ShuffleMode::kFalse, 2, 2);
  EXPECT_NE(ds4, nullptr);
  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter4 = ds4->CreateIterator();
  // Expect failure: invalid AmazonReview input.
  EXPECT_EQ(iter4, nullptr);
}

/// Feature: AmazonReview
/// Description: Test AmazonReview Dataset interface in pipeline.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestAmazonReviewDatasetBasicWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAmazonReviewDatasetBasicWithPipeline.";

  // Create two AmazonReview Dataset, with single AmazonReview file.
  std::string dataset_dir = datasets_root_path_ + "/testAmazonReview/full";

  std::shared_ptr<Dataset> ds1 = AmazonReview(dataset_dir, "test", 0, ShuffleMode::kFalse);
  std::shared_ptr<Dataset> ds2 = AmazonReview(dataset_dir, "test", 0, ShuffleMode::kFalse);
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
  EXPECT_EQ(i, 15);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: AmazonReview(ShuffleMode=kFiles).
/// Description: Test AmazonReview Dataset interface with different ShuffleMode.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestAmazonReviewDatasetShuffleFilesA) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-AmazonReviewDatasetShuffleFilesA.";

  // Set configuration.
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(130);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  std::string dataset_dir = datasets_root_path_ + "/testAmazonReview/full";
  std::vector<std::string> column_names = {"label", "title", "content"};

  std::shared_ptr<Dataset> ds = AmazonReview(dataset_dir, "all" , 0, ShuffleMode::kFiles);
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
    {"3", "Satisfied", "good quality."}, 
    {"1", "amazing", "unlimited buyback!"},
    {"5", "good", "This is an very good product."},
    {"4", "delightful", "a funny book!"},
    {"1", "bad", "work badly."},
    {"3", "Small", "It is a small ball!"}    
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
  // Expect 6 samples.
  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline.
  iter->Stop();

  // Restore configuration
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

/// Feature: AmazonReview(ShuffleMode=kInfile).
/// Description: Test AmazonReview Dataset interface with different ShuffleMode.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestAmazonReviewDatasetShuffleFilesB) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAmazonReviewDatasetShuffleFilesB.";

  // Set configuration.
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(130);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  std::string dataset_dir = datasets_root_path_ + "/testAmazonReview/full";
  std::vector<std::string> column_names = {"label", "title", "content"};

  std::shared_ptr<Dataset> ds = AmazonReview(dataset_dir, "all" , 0, ShuffleMode::kFiles);
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
    {"3", "Satisfied", "good quality."}, 
    {"1", "amazing", "unlimited buyback!"},
    {"5", "good", "This is an very good product."},
    {"4", "delightful", "a funny book!"},
    {"1", "bad", "work badly."},
    {"3", "Small", "It is a small ball!"}  
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
  // Expect 6 samples
  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();

  // Restore configuration
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

/// Feature: AmazonReview(ShuffleMode=kGlobal).
/// Description: Test AmazonReview Dataset interface with different ShuffleMode.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestAmazonReviewDatasetShuffleFilesGlobal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAmazonReviewDatasetShuffleFilesGlobal.";

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(130);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  std::string dataset_dir = datasets_root_path_ + "/testAmazonReview/full";
  std::vector<std::string> column_names = {"label", "title", "content"};

  std::shared_ptr<Dataset> ds = AmazonReview(dataset_dir, "all" , 0, ShuffleMode::kGlobal);
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
    {"3", "Satisfied", "good quality."},
    {"1", "amazing", "unlimited buyback!"},
    {"5", "good", "This is an very good product."},
    {"3", "Small", "It is a small ball!"},
    {"4", "delightful", "a funny book!"},
    {"1", "bad", "work badly."}
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
  // Expect 6 samples.
  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline.
  iter->Stop();

  // Restore configuration
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers); 
}