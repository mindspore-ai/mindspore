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

#include "minddata/dataset/engine/ir/datasetops/source/ag_news_node.h"

using namespace mindspore::dataset;

class MindDataTestPipeline : public UT::DatasetOpTesting {
protected:
};

/// Feature: AGNewsDataset
/// Description: Basic test for AGNewsDataset
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestAGNewsDatasetBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAGNewsDatasetBasic.";

  std::string dataset_dir = datasets_root_path_ + "/testAGNews";
  std::vector<std::string> column_names = {"index", "title", "description"};
  std::shared_ptr<Dataset> ds =
      AGNews(dataset_dir, "test", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);
  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("index"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
      {"3", "Background of the selection",
       "In this day and age, the internet is growing rapidly, "
       "the total number of connected devices is increasing and "
       "we are entering the era of big data."},
      {"4", "Related technologies",
       "\"Leaflet is the leading open source JavaScript library "
       "for mobile-friendly interactive maps.\""},
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

/// Feature: AGNewsDataset
/// Description: Test AGNewsDataset in pipeline mode
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestAGNewsGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAGNewsGetters.";

  std::string dataset_dir = datasets_root_path_ + "/testAGNews";
  std::shared_ptr<Dataset> ds =
      AGNews(dataset_dir, "test", 0, ShuffleMode::kFalse);
  std::vector<std::string> column_names = {"index", "title", "description"};
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
  EXPECT_EQ(ds->GetColumnNames(), column_names);
  EXPECT_EQ(ds->GetDatasetSize(), 2);
  EXPECT_EQ(ds->GetColumnNames(), column_names);
}

/// Feature: AGNewsDataset
/// Description: Test AGNewsDataset with invalid inputs
/// Expectation: Correct error and message are thrown
TEST_F(MindDataTestPipeline, TestAGNewsDatasetFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAGNewsDatasetFail.";

  std::string dataset_dir = datasets_root_path_ + "/testAGNews";
  std::string invalid_csv_file = "./NotExistFile";
  std::vector<std::string> column_names = {"index", "title", "description"};
  std::shared_ptr<Dataset> ds0 = AGNews("", "test", 0);
  EXPECT_NE(ds0, nullptr);
  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter0 = ds0->CreateIterator();
  // Expect failure: invalid AGNews input.
  EXPECT_EQ(iter0, nullptr);
  // Create a AGNews Dataset with invalid usage.
  std::shared_ptr<Dataset> ds1 = AGNews(invalid_csv_file);
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid AGNews input.
  EXPECT_EQ(iter1, nullptr);
  // Test invalid num_samples < -1.
  std::shared_ptr<Dataset> ds2 =
      AGNews(dataset_dir, "test", -1, ShuffleMode::kFalse);
  EXPECT_NE(ds2, nullptr);
  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid AGNews input.
  EXPECT_EQ(iter2, nullptr);
  // Test invalid num_shards < 1.
  std::shared_ptr<Dataset> ds3 =
      AGNews(dataset_dir, "test", 0, ShuffleMode::kFalse, 0);
  EXPECT_NE(ds3, nullptr);
  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid AGNews input.
  EXPECT_EQ(iter3, nullptr);
  // Test invalid shard_id >= num_shards.
  std::shared_ptr<Dataset> ds4 =
      AGNews(dataset_dir, "test", 0, ShuffleMode::kFalse, 2, 2);
  EXPECT_NE(ds4, nullptr);
  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter4 = ds4->CreateIterator();
  // Expect failure: invalid AGNews input.
  EXPECT_EQ(iter4, nullptr);
}

/// Feature: AGNewsDataset
/// Description: Test AGNewsDataset with valid num_samples
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestAGNewsDatasetNumSamples) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAGNewsDatasetNumSamples.";

  // Create a AGNewsDataset, with single CSV file.
  std::string dataset_dir = datasets_root_path_ + "/testAGNews";
  std::shared_ptr<Dataset> ds =
      AGNews(dataset_dir, "test", 2, ShuffleMode::kFalse);
  std::vector<std::string> column_names = {"index", "title", "description"};
  EXPECT_NE(ds, nullptr);
  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it..
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("index"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
      {"3", "Background of the selection",
       "In this day and age, the internet is growing rapidly, "
       "the total number of connected devices is increasing and "
       "we are entering the era of big data."},
      {"4", "Related technologies",
       "\"Leaflet is the leading open source JavaScript library "
       "for mobile-friendly interactive maps.\""},
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

/// Feature: AGNewsDataset
/// Description: Test distributed AGNewsDataset (with num_shards and shard_id)
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestAGNewsDatasetDistribution) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAGNewsDatasetDistribution.";

  // Create a AGNewsDataset, with single CSV file.
  std::string dataset_dir = datasets_root_path_ + "/testAGNews";
  std::shared_ptr<Dataset> ds =
      AGNews(dataset_dir, "test", 0, ShuffleMode::kFalse, 2, 0);
  std::vector<std::string> column_names = {"index", "title", "description"};
  EXPECT_NE(ds, nullptr);
  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("index"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
      {"3", "Background of the selection",
       "In this day and age, the internet is growing rapidly, "
       "the total number of connected devices is increasing and "
       "we are entering the era of big data."},
      {"4", "Related technologies",
       "\"Leaflet is the leading open source JavaScript library "
       "for mobile-friendly interactive maps.\""},
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

/// Feature: AGNewsDataset
/// Description: Test AGNewsDataset with all as usage
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestAGNewsDatasetMultiFiles) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAGNewsDatasetMultiFiles.";

  // Create a AGNewsDataset, with single CSV file.
  std::string dataset_dir = datasets_root_path_ + "/testAGNews";
  std::shared_ptr<Dataset> ds =
      AGNews(dataset_dir, "all", 0, ShuffleMode::kFalse);
  std::vector<std::string> column_names = {"index", "title", "description"};
  EXPECT_NE(ds, nullptr);
  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("index"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
      {"3", "Background of the selection",
       "In this day and age, the internet is growing rapidly, "
       "the total number of connected devices is increasing and "
       "we are entering the era of big data."},
      {"3", "Demand analysis",
       "\"Users simply click on the module they want to view to "
       "browse information about that module.\""},
      {"4", "Related technologies",
       "\"Leaflet is the leading open source JavaScript library "
       "for mobile-friendly interactive maps.\""},
      {"3", "UML Timing Diagram",
       "Information is mainly displayed using locally stored data and mapping, "
       "which is not timely and does not have the ability to update itself."},
      {"3", "In summary",
       "This paper implements a map visualization system for Hangzhou city "
       "information, using extensive knowledge of visualization techniques."},
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

/// Feature: AGNewsDataset
/// Description: Test AGNewsDataset header
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestAGNewsDatasetHeader) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAGNewsDatasetHeader.";

  // Create a AGNewsDataset, with single CSV file.
  std::string dataset_dir = datasets_root_path_ + "/testAGNews";
  std::shared_ptr<Dataset> ds =
      AGNews(dataset_dir, "test", 0, ShuffleMode::kFalse);
  std::vector<std::string> column_names = {"index", "title", "description"};
  EXPECT_NE(ds, nullptr);
  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("index"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
      {"3", "Background of the selection",
       "In this day and age, the internet is growing rapidly, "
       "the total number of connected devices is increasing and "
       "we are entering the era of big data."},
      {"4", "Related technologies",
       "\"Leaflet is the leading open source JavaScript library "
       "for mobile-friendly interactive maps.\""},
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

/// Feature: AGNewsDataset
/// Description: Test AGNewsDataset using ShuffleMode::kFiles
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestAGNewsDatasetShuffleFilesA) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAGNewsDatasetShuffleFilesA.";

  // Set configuration.
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers =
      GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed
                << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(130);
  GlobalContext::config_manager()->set_num_parallel_workers(4);
  std::string dataset_dir = datasets_root_path_ + "/testAGNews";
  std::shared_ptr<Dataset> ds =
      AGNews(dataset_dir, "all", 0, ShuffleMode::kFiles);
  std::vector<std::string> column_names = {"index", "title", "description"};
  EXPECT_NE(ds, nullptr);
  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("index"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
      {"3", "Demand analysis",
       "\"Users simply click on the module they want to view to "
       "browse information about that module.\""},
      {"3", "Background of the selection",
       "In this day and age, the internet is growing rapidly, "
       "the total number of connected devices is increasing and "
       "we are entering the era of big data."},
      {"3", "UML Timing Diagram",
       "Information is mainly displayed using locally stored data and mapping, "
       "which is not timely and does not have the ability to update itself."},
      {"4", "Related technologies",
       "\"Leaflet is the leading open source JavaScript library "
       "for mobile-friendly interactive maps.\""},
      {"3", "In summary",
       "This paper implements a map visualization system for Hangzhou city "
       "information, using extensive knowledge of visualization techniques."},
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
  GlobalContext::config_manager()->set_num_parallel_workers(
      original_num_parallel_workers);
}

/// Feature: AGNewsDataset
/// Description: Test AGNewsDataset using ShuffleMode::kGlobal
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestAGNewsDatasetShuffleGlobal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAGNewsDatasetShuffleGlobal.";
  // Test AGNews Dataset with GLOBLE shuffle.
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers =
      GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed
                << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(135);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  std::string dataset_dir = datasets_root_path_ + "/testAGNews";
  std::shared_ptr<Dataset> ds =
      AGNews(dataset_dir, "train", 0, ShuffleMode::kGlobal);
  std::vector<std::string> column_names = {"index", "title", "description"};
  EXPECT_NE(ds, nullptr);
  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("index"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
      {"3", "UML Timing Diagram",
       "Information is mainly displayed using locally stored data and mapping, "
       "which is not timely and does not have the ability to update itself."},
      {"3", "In summary",
       "This paper implements a map visualization system for Hangzhou city "
       "information, using extensive knowledge of visualization techniques."},
      {"3", "Demand analysis",
       "\"Users simply click on the module they want to view to "
       "browse information about that module.\""},
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
  GlobalContext::config_manager()->set_num_parallel_workers(
      original_num_parallel_workers);
}
