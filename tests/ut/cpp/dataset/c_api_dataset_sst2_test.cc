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
#include "minddata/dataset/engine/ir/datasetops/source/sst2_node.h"
#include "minddata/dataset/include/dataset/datasets.h"

using namespace mindspore::dataset;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

/// Feature: SST2Dataset
/// Description: Read test data
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestSST2DatasetBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSST2DatasetBasic.";

  // Create a SST2 Dataset
  std::string folder_path = datasets_root_path_ + "/testSST2/";
  std::vector<std::string> column_names = {"sentence"};
  std::shared_ptr<Dataset> ds = SST2(folder_path, "test", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterator the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_EQ(row.find("index"), row.end());
  EXPECT_NE(row.find("sentence"), row.end());

  std::vector<std::vector<std::string>> expected_result = {
    {"test read SST2dataset 1 ."},
    {"test read SST2dataset 2 ."},
    {"test read SST2dataset 3 ."}};
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

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: SST2Dataset
/// Description: Read train data and test data
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestSST2DatasetUsageTrain) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSST2DatasetUsageTrain.";

  std::string folder_path = datasets_root_path_ + "/testSST2/";
  std::vector<std::string> column_names = {"sentence", "label"};
  std::shared_ptr<Dataset> ds = SST2(folder_path, "train", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterator the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("sentence"), row.end());

  std::vector<std::vector<std::string>> expected_result = {
    {"train read SST2Dataset 1 . ","0"},
    {"train read SST2Dataset 2 . ","1"},
    {"train read SST2Dataset 3 . ","1"},
    {"train read SST2Dataset 4 . ","1"},
    {"train read SST2Dataset 5 . ","0"}};
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

  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: SST2Dataset
/// Description: Includes tests for shape, type, size
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestSST2DatasetGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSST2DatasetGetters.";

  std::string folder_path = datasets_root_path_ + "/testSST2/";
  std::shared_ptr<Dataset> ds = SST2(folder_path, "test", 0, ShuffleMode::kFalse);
  std::vector<std::string> column_names = {"sentence"};
  EXPECT_NE(ds, nullptr);

  std::vector<DataType> types = ToDETypes(ds->GetOutputTypes());
  std::vector<TensorShape> shapes = ToTensorShapeVec(ds->GetOutputShapes());
  EXPECT_EQ(types.size(), 1);
  EXPECT_EQ(types[0].ToString(), "string");

  EXPECT_EQ(shapes.size(), 1);
  EXPECT_EQ(shapes[0].ToString(), "<>");

  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);
  EXPECT_EQ(ds->GetDatasetSize(), 3);
  EXPECT_EQ(ds->GetColumnNames(), column_names);
}

/// Feature: SST2Dataset
/// Description: Read 2 samples from train file
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestSST2DatasetNumSamples) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSST2DatasetNumSamples.";

  // Create a SST2Dataset.
  std::string folder_path = datasets_root_path_ + "/testSST2/";
  std::vector<std::string> column_names = {"sentence", "label"};
  std::shared_ptr<Dataset> ds = SST2(folder_path, "train", 2, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("sentence"), row.end());
  EXPECT_NE(row.find("label"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
    {"train read SST2Dataset 1 . ","0"},
    {"train read SST2Dataset 2 . ","1"}};

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

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: SST2Dataset
/// Description: Test in a distributed state
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestSST2DatasetDistribution) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSST2DatasetDistribution.";

  // Create a SST2Dataset
  std::string folder_path = datasets_root_path_ + "/testSST2/";
  std::vector<std::string> column_names = {"sentence", "label"};
  std::shared_ptr<Dataset> ds = SST2(folder_path, "train", 0, ShuffleMode::kFalse, 2, 0);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("sentence"), row.end());
  EXPECT_NE(row.find("label"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
    {"train read SST2Dataset 1 . ","0"},
    {"train read SST2Dataset 2 . ","1"},
    {"train read SST2Dataset 3 . ","1"}};

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

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: SST2Dataset
/// Description: Test with invalid input
/// Expectation: Throw error messages when certain errors occur
TEST_F(MindDataTestPipeline, TestSST2DatasetFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSST2DatasetFail.";
  // Create a SST2 Dataset
  std::string folder_path = datasets_root_path_ + "/testSST2/";
  std::string invalid_folder_path = "./NotExistPath";
  std::vector<std::string> column_names = {"sentence", "label"};

  // Test invalid folder_path
  std::shared_ptr<Dataset> ds0 = SST2(invalid_folder_path, "train", -1, ShuffleMode::kFalse);
  EXPECT_NE(ds0, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter0 = ds0->CreateIterator();
  // Expect failure: invalid SST2 input
  EXPECT_EQ(iter0, nullptr);

  // Test invalid usage
  std::shared_ptr<Dataset> ds1 = SST2(folder_path, "all", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid SST2 input
  EXPECT_EQ(iter1, nullptr);

  // Test invalid num_samples < -1
  std::shared_ptr<Dataset> ds2 = SST2(folder_path, "train", -1, ShuffleMode::kFalse);
  EXPECT_NE(ds2, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid SST2 input
  EXPECT_EQ(iter2, nullptr);

  // Test invalid num_shards < 1
  std::shared_ptr<Dataset> ds3 = SST2(folder_path, "train", 0, ShuffleMode::kFalse, 0);
  EXPECT_NE(ds3, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid SST2 input
  EXPECT_EQ(iter3, nullptr);

  // Test invalid shard_id >= num_shards
  std::shared_ptr<Dataset> ds4 = SST2(folder_path, "train", 0, ShuffleMode::kFalse, 2, 2);
  EXPECT_NE(ds4, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter4 = ds4->CreateIterator();
  // Expect failure: invalid SST2 input
  EXPECT_EQ(iter4, nullptr);
}

/// Feature: SST2Dataset
/// Description: Read data with pipeline from test file
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestSST2DatasetWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSST2DatasetWithPipeline.";

  // Create two SST2 Dataset, with single SST2 file
  std::string dataset_dir = datasets_root_path_ + "/testSST2/";

  std::shared_ptr<Dataset> ds1 = SST2(dataset_dir, "test", 0, ShuffleMode::kFalse);
  std::shared_ptr<Dataset> ds2 = SST2(dataset_dir, "test", 0, ShuffleMode::kFalse);
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
  std::vector<std::string> column_project = {"sentence"};
  ds1 = ds1->Project(column_project);
  EXPECT_NE(ds1, nullptr);
  ds2 = ds2->Project(column_project);
  EXPECT_NE(ds2, nullptr);

  // Create a Concat operation on the ds
  ds1 = ds1->Concat({ds2});
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it
  std::shared_ptr<Iterator> iter = ds1->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("sentence"), row.end());
  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["sentence"];
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  // Expect 2 × 3 + 3 × 3 = 15 samples
  EXPECT_EQ(i, 15);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: SST2Dataset
/// Description: Test with shuffle files
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestSST2DatasetShuffleFilesA) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSST2DatasetShuffleFilesA.";

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(130);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  std::string folder_path = datasets_root_path_ + "/testSST2/";
  std::vector<std::string> column_names = {"sentence", "label"};
  std::shared_ptr<Dataset> ds = SST2(folder_path, "train", 0, ShuffleMode::kFiles);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("sentence"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
    {"train read SST2Dataset 1 . ","0"},
    {"train read SST2Dataset 2 . ","1"},
    {"train read SST2Dataset 3 . ","1"},
    {"train read SST2Dataset 4 . ","1"},
    {"train read SST2Dataset 5 . ","0"}};

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

  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();

  // Restore configuration
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

/// Feature: SST2Dataset
/// Description: Test with shuffle in file
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestSST2DatasetShuffleFilesB) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSST2DatasetShuffleFilesB.";

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(130);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  std::string folder_path = datasets_root_path_ + "/testSST2/";
  std::vector<std::string> column_names = {"sentence"};
  std::shared_ptr<Dataset> ds = SST2(folder_path, "test", 0, ShuffleMode::kInfile);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("sentence"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
    {"test read SST2dataset 1 ."},
    {"test read SST2dataset 2 ."},
    {"test read SST2dataset 3 ."}};

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

  // Restore configuration
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

/// Feature: SST2Dataset
/// Description: Test with global shuffle
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestSST2DatasetShuffleGlobal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSST2DatasetShuffleFilesGlobal.";

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(130);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  std::string folder_path = datasets_root_path_ + "/testSST2/";
  std::vector<std::string> column_names = {"sentence"};
  std::shared_ptr<Dataset> ds = SST2(folder_path, "test", 0, ShuffleMode::kGlobal);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("sentence"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
    {"test read SST2dataset 1 ."},
    {"test read SST2dataset 3 ."},
    {"test read SST2dataset 2 ."}};

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

  // Restore configuration
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}
