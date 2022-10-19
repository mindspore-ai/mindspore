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
#include "minddata/dataset/engine/ir/datasetops/source/dbpedia_node.h"
#include "minddata/dataset/include/dataset/datasets.h"

using namespace mindspore::dataset;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

/// Feature: DBpediaDataset
/// Description: Test DBpediaDataset basic usage
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestDBpediaDatasetBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDBpediaDatasetBasic.";

  // Create a DBpedia Dataset
  std::string folder_path = datasets_root_path_ + "/testDBpedia/";
  std::vector<std::string> column_names = {"class", "title", "content"};
  std::shared_ptr<Dataset> ds = DBpedia(folder_path, "test", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterator the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("class"), row.end());

  std::vector<std::vector<std::string>> expected_result = {
    {"5", "My Bedroom", "Look at this room. It's my bedroom."},
    {"8", "My English teacher", "She has two big eyes and a small mouth."},
    {"6", "My Holiday", "I have a lot of fun every day."}};
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

/// Feature: DBpediaDataset
/// Description: Test DBpediaDataset with all as usage
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestDBpediaDatasetUsageAll) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDBpediaDatasetUsageAll.";

  std::string folder_path = datasets_root_path_ + "/testDBpedia/";
  std::vector<std::string> column_names = {"class", "title", "content"};
  std::shared_ptr<Dataset> ds = DBpedia(folder_path, "all", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterator the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("class"), row.end());

  std::vector<std::vector<std::string>> expected_result = {
    {"5", "My Bedroom", "Look at this room. It's my bedroom."},
    {"7", "My Last Weekend", "I was busy last week, but I have fun every day."},
    {"8", "My English teacher", "She has two big eyes and a small mouth."},
    {"5", "My Friend", "She likes singing, dancing and swimming very much."},
    {"6", "My Holiday", "I have a lot of fun every day."},
    {"8", "I Can Do Housework", "My mother is busy, so I often help my mother with the housework."}};
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

/// Feature: DBpediaDataset
/// Description: Test iterator of DBpediaDataset with only the class column
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestDBpediaDatasetIteratorOneColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDBpediaDatasetIteratorOneColumn.";
  // Create a DBpedia Dataset
  std::string folder_path = datasets_root_path_ + "/testDBpedia/";
  std::vector<std::string> column_names = {"class", "title", "content"};
  std::shared_ptr<Dataset> ds = DBpedia(folder_path, "test", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // Only select "class" column and drop others
  std::vector<std::string> columns = {"class"};
  std::shared_ptr<ProjectDataset> project_ds = ds->Project(columns);
  std::shared_ptr<Iterator> iter = project_ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  std::vector<int64_t> expect_class = {1};

  uint64_t i = 0;
  while (row.size() != 0) {
    for (auto &v : row) {
      MS_LOG(INFO) << "class shape:" << v.Shape();
      EXPECT_EQ(expect_class, v.Shape());
    }
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: DBpediaDataset
/// Description: Test iterator of DBpediaDataset with wrong column
/// Expectation: Get none piece of data
TEST_F(MindDataTestPipeline, TestDBpediaDatasetIteratorWrongColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDBpediaDatasetIteratorWrongColumn.";
  // Create a DBpedia Dataset
  std::string folder_path = datasets_root_path_ + "/testDBpedia/";
  std::vector<std::string> column_names = {"class", "title", "content"};
  std::shared_ptr<Dataset> ds = DBpedia(folder_path, "test", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Pass wrong column name
  std::vector<std::string> columns = {"digital"};
  std::shared_ptr<ProjectDataset> project_ds = ds->Project(columns);
  std::shared_ptr<Iterator> iter = project_ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: DBpediaDataset
/// Description: Test DBpediaDataset Getters method
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestDBpediaDatasetGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDBpediaDatasetGetters.";

  std::string folder_path = datasets_root_path_ + "/testDBpedia/";
  std::shared_ptr<Dataset> ds = DBpedia(folder_path, "test", 0, ShuffleMode::kFalse);
  std::vector<std::string> column_names = {"class", "title", "content"};
  EXPECT_NE(ds, nullptr);

  std::vector<DataType> types = ToDETypes(ds->GetOutputTypes());
  std::vector<TensorShape> shapes = ToTensorShapeVec(ds->GetOutputShapes());
  EXPECT_EQ(types.size(), 3);
  EXPECT_EQ(types[0].ToString(), "string");
  EXPECT_EQ(types[1].ToString(), "string");
  EXPECT_EQ(types[2].ToString(), "string");
  EXPECT_EQ(shapes.size(), 3);
  EXPECT_EQ(shapes[0].ToString(), "<>");
  EXPECT_EQ(shapes[1].ToString(), "<>");
  EXPECT_EQ(shapes[2].ToString(), "<>");
  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);
  EXPECT_EQ(ds->GetDatasetSize(), 3);
  EXPECT_EQ(ds->GetColumnNames(), column_names);
}

/// Feature: DBpediaDataset
/// Description: Test DBpediaDataset with num_samples
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestDBpediaDatasetNumSamples) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDBpediaDatasetNumSamples.";

  // Create a DBpediaDataset
  std::string folder_path = datasets_root_path_ + "/testDBpedia/";
  std::vector<std::string> column_names = {"class", "title", "content"};
  std::shared_ptr<Dataset> ds = DBpedia(folder_path, "train", 2, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("class"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
    {"7", "My Last Weekend", "I was busy last week, but I have fun every day."},
    {"5", "My Friend", "She likes singing, dancing and swimming very much."}};

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

  // Expect 2 samples
  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: DBpediaDataset
/// Description: Test distributed DBpediaDataset (with num_shards and shard_id)
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestDBpediaDatasetDistribution) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDBpediaDatasetDistribution.";

  // Create a DBpediaDataset
  std::string folder_path = datasets_root_path_ + "/testDBpedia/";
  std::vector<std::string> column_names = {"class", "title", "content"};
  std::shared_ptr<Dataset> ds = DBpedia(folder_path, "train", 0, ShuffleMode::kFalse, 2, 0);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("class"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
    {"7", "My Last Weekend", "I was busy last week, but I have fun every day."},
    {"5", "My Friend", "She likes singing, dancing and swimming very much."}};

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

  // Expect 2 samples
  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: DBpediaDataset
/// Description: Test DBpediaDataset with invalid inputs
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestDBpediaDatasetFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDBpediaDatasetFail.";
  // Create a DBpedia Dataset
  std::string folder_path = datasets_root_path_ + "/testDBpedia/";
  std::string invalid_folder_path = "./NotExistPath";
  std::vector<std::string> column_names = {"class", "title", "content"};

  // Test invalid folder_path
  std::shared_ptr<Dataset> ds0 = DBpedia(invalid_folder_path, "all", -1, ShuffleMode::kFalse);
  EXPECT_NE(ds0, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter0 = ds0->CreateIterator();
  // Expect failure: invalid DBpedia input
  EXPECT_EQ(iter0, nullptr);

  // Test invalid usage
  std::shared_ptr<Dataset> ds1 = DBpedia(folder_path, "invalid_usage", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid DBpedia input
  EXPECT_EQ(iter1, nullptr);

  // Test invalid num_samples < -1
  std::shared_ptr<Dataset> ds2 = DBpedia(folder_path, "all", -1, ShuffleMode::kFalse);
  EXPECT_NE(ds2, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid DBpedia input
  EXPECT_EQ(iter2, nullptr);

  // Test invalid num_shards < 1
  std::shared_ptr<Dataset> ds3 = DBpedia(folder_path, "all", 0, ShuffleMode::kFalse, 0);
  EXPECT_NE(ds3, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid DBpedia input
  EXPECT_EQ(iter3, nullptr);

  // Test invalid shard_id >= num_shards
  std::shared_ptr<Dataset> ds4 = DBpedia(folder_path, "all", 0, ShuffleMode::kFalse, 2, 2);
  EXPECT_NE(ds4, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter4 = ds4->CreateIterator();
  // Expect failure: invalid DBpedia input
  EXPECT_EQ(iter4, nullptr);
}

/// Feature: DBpediaDataset
/// Description: Test DBpediaDataset in pipeline mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestDBpediaDatasetWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDBpediaDatasetWithPipeline.";

  // Create two DBpedia Dataset, with single DBpedia file.
  std::string dataset_dir = datasets_root_path_ + "/testDBpedia/";

  std::shared_ptr<Dataset> ds1 = DBpedia(dataset_dir, "test", 0, ShuffleMode::kFalse);
  std::shared_ptr<Dataset> ds2 = DBpedia(dataset_dir, "test", 0, ShuffleMode::kFalse);
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

  // Expect 15 samples.
  EXPECT_EQ(i, 15);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: DBpediaDataset
/// Description: Test DBpediaDataset with ShuffleMode::kFiles
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestDBpediaDatasetShuffleFilesA) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDBpediaDatasetShuffleFilesA.";

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(130);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  std::string folder_path = datasets_root_path_ + "/testDBpedia/";
  std::vector<std::string> column_names = {"class", "title", "content"};
  std::shared_ptr<Dataset> ds = DBpedia(folder_path, "all", 0, ShuffleMode::kFiles);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("class"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
    {"7", "My Last Weekend", "I was busy last week, but I have fun every day."},
    {"5", "My Bedroom", "Look at this room. It's my bedroom."},
    {"5", "My Friend", "She likes singing, dancing and swimming very much."},
    {"8", "My English teacher", "She has two big eyes and a small mouth."},
    {"8", "I Can Do Housework", "My mother is busy, so I often help my mother with the housework."},
    {"6", "My Holiday", "I have a lot of fun every day."}};

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

/// Feature: DBpediaDataset
/// Description: Test DBpediaDataset with ShuffleMode::kInfile
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestDBpediaDatasetShuffleFilesB) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDBpediaDatasetShuffleFilesB.";

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(130);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  std::string folder_path = datasets_root_path_ + "/testDBpedia/";
  std::vector<std::string> column_names = {"class", "title", "content"};
  std::shared_ptr<Dataset> ds = DBpedia(folder_path, "test", 0, ShuffleMode::kInfile);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("class"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
    {"5", "My Bedroom", "Look at this room. It's my bedroom."},
    {"8", "My English teacher", "She has two big eyes and a small mouth."},
    {"6", "My Holiday", "I have a lot of fun every day."}};

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

/// Feature: DBpediaDataset
/// Description: Test DBpediaDataset with ShuffleMode::kGlobal
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestDBpediaDatasetShuffleGlobal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDBpediaDatasetShuffleFilesGlobal.";

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(130);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  std::string folder_path = datasets_root_path_ + "/testDBpedia/";
  std::vector<std::string> column_names = {"class", "title", "content"};
  std::shared_ptr<Dataset> ds = DBpedia(folder_path, "test", 0, ShuffleMode::kGlobal);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("class"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
    {"5", "My Bedroom", "Look at this room. It's my bedroom."},
    {"6", "My Holiday", "I have a lot of fun every day."},
    {"8", "My English teacher", "She has two big eyes and a small mouth."}};

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
