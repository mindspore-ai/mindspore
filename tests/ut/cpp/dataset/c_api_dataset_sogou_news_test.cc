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

/// Feature: Test SogouNews Dataset.
/// Description: Read SogouNewsDataset data and get data.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestSogouNewsDatasetBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSogouNewsDatasetBasic.";

  std::string dataset_dir = datasets_root_path_ + "/testSogouNews/";
  std::vector<std::string> column_names = {"index", "title", "content"};

  std::shared_ptr<Dataset> ds = SogouNews(dataset_dir, "test", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("index"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
      {"1","Make history","Su Bingtian's 100m breakthrough\\n 9.83"},
      {"4","Tesla price","Tesla reduced its price by 70000 yuan"},
      {"1","Opening ceremony of the 14th National Games","On the evening of September 15, Beijing time, "
      "the 14th games of the people's Republic of China opened in Xi'an Olympic Sports Center Stadium, "
      "Shaanxi Province. Yang Qian, the first gold medalist of the Chinese delegation in the Tokyo Olympic"
      " Games and a Post-00 shooter, lit the main torch platform. From then on, to September 27, the 14th"
      " National Games flame will burn here for 12 days."}
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

/// Feature: Test SogouNews Dataset(usage=all).
/// Description: Read train data and test data.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestSogouNewsDatasetUsageAll) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSogouNewsDatasetUsageAll.";

  std::string dataset_dir = datasets_root_path_ + "/testSogouNews/";
  std::vector<std::string> column_names = {"index", "title", "content"};

  std::shared_ptr<Dataset> ds = SogouNews(dataset_dir, "all" , 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("index"), row.end());
    std::vector<std::vector<std::string>> expected_result = {
      {"1","Jefferson commented on thick eyebrow: he has the top five talents in the league, but he is not the"
      " top five","They say he has the talent of the top five in the league. The talent of the top five in the"
      " league is one of the most disrespectful statements. I say he has the talent of the top five in the league,"
      " but he is not the top five players because the top five players play every night."},
      {"1","Make history","Su Bingtian's 100m breakthrough\\n 9.83"},
      {"3","Group pictures: Liu Shishi's temperament in early autumn released a large piece of micro curly long"
      " hair, elegant, lazy, gentle and capable","Liu Shishi's latest group of cover magazine blockbusters are"
      " released. In the photos, Liu Shishi's long hair is slightly curly, or camel colored belted woolen coat,"
      " or plaid suit, which is gentle and elegant and beautiful to a new height."},
      {"4","Tesla price","Tesla reduced its price by 70000 yuan"},
      {"3","Ni Ni deduces elegant retro style in different styles","Ni Ni's latest group of magazine cover"
      " blockbusters released that wearing gift hats is cool, retro, unique and full of fashion expression."},
      {"1","Opening ceremony of the 14th National Games","On the evening of September 15, Beijing time, "
      "the 14th games of the people's Republic of China opened in Xi'an Olympic Sports Center Stadium, "
      "Shaanxi Province. Yang Qian, the first gold medalist of the Chinese delegation in the Tokyo Olympic"
      " Games and a Post-00 shooter, lit the main torch platform. From then on, to September 27, the 14th"
      " National Games flame will burn here for 12 days."}
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

/// Feature: Test Getters.
/// Description: Includes tests for shape, type, size.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestSogouNewsGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSogouNewsGetters.";

  std::string dataset_dir = datasets_root_path_ + "/testSogouNews/";
  std::shared_ptr<Dataset> ds = SogouNews(dataset_dir, "test", 0, ShuffleMode::kFalse);
  std::vector<std::string> column_names = {"index", "title", "content"};
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds-> GetDatasetSize(),3);
  EXPECT_EQ(ds->GetColumnNames(),column_names);
}

/// Feature: Test SogouNews Dataset(num_samples = 3).
/// Description: Test whether the interface meets expectations when NumSamples is equal to 3.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestSogouNewsNumSamples) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSogouNewsNumSamples.";

  std::string dataset_dir = datasets_root_path_ + "/testSogouNews/";
  std::vector<std::string> column_names = {"index", "title", "content"};

  std::shared_ptr<Dataset> ds = SogouNews(dataset_dir, "test", 3, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("index"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
      {"1","Make history","Su Bingtian's 100m breakthrough\\n 9.83"},
      {"4","Tesla price","Tesla reduced its price by 70000 yuan"},
      {"1","Opening ceremony of the 14th National Games","On the evening of September 15, Beijing time, "
      "the 14th games of the people's Republic of China opened in Xi'an Olympic Sports Center Stadium, "
      "Shaanxi Province. Yang Qian, the first gold medalist of the Chinese delegation in the Tokyo Olympic"
      " Games and a Post-00 shooter, lit the main torch platform. From then on, to September 27, the 14th"
      " National Games flame will burn here for 12 days."}
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

/// Feature: Test SogouNewsDataset in distribution.
/// Description: Test interface in a distributed state.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestSogouNewsDatasetDistribution) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSogouNewsDatasetDistribution.";

  std::string dataset_dir = datasets_root_path_ + "/testSogouNews/";
  std::vector<std::string> column_names = {"index", "title", "content"};

  std::shared_ptr<Dataset> ds = SogouNews(dataset_dir, "test", 0, ShuffleMode::kFalse, 2, 0);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("index"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
      {"1","Make history","Su Bingtian's 100m breakthrough\\n 9.83"},
      {"4","Tesla price","Tesla reduced its price by 70000 yuan"},
      {"1","Opening ceremony of the 14th National Games","On the evening of September 15, Beijing time, "
      "the 14th games of the people's Republic of China opened in Xi'an Olympic Sports Center Stadium, "
      "Shaanxi Province. Yang Qian, the first gold medalist of the Chinese delegation in the Tokyo Olympic"
      " Games and a Post-00 shooter, lit the main torch platform. From then on, to September 27, the 14th"
      " National Games flame will burn here for 12 days."}
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

  // Expect 2 samples
  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Error Test.
/// Description: Test the wrong input.
/// Expectation: Unable to read in data.
TEST_F(MindDataTestPipeline, TestSogouNewsDatasetFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSogouNewsDatasetFail.";

  std::string dataset_dir = datasets_root_path_ + "/testSogouNews/";
  std::string invalid_csv_file = "./NotExistFile";
  std::vector<std::string> column_names = {"index", "title", "content"};

  std::shared_ptr<Dataset> ds0 = SogouNews("", "test", 0);
  EXPECT_NE(ds0, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter0 = ds0->CreateIterator();
  // Expect failure: invalid SogouNews input
  EXPECT_EQ(iter0, nullptr);

  // Create a SogouNews Dataset with invalid usage
  std::shared_ptr<Dataset> ds1 = SogouNews(invalid_csv_file);
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid SogouNews input
  EXPECT_EQ(iter1, nullptr);

  // Test invalid num_samples < -1
  std::shared_ptr<Dataset> ds2 = SogouNews(dataset_dir, "test", -1, ShuffleMode::kFalse);
  EXPECT_NE(ds2, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid SogouNews input
  EXPECT_EQ(iter2, nullptr);

  // Test invalid num_shards < 1
  std::shared_ptr<Dataset> ds3 = SogouNews(dataset_dir, "test", 0, ShuffleMode::kFalse, 0);
  EXPECT_NE(ds3, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid SogouNews input
  EXPECT_EQ(iter3, nullptr);

  // Test invalid shard_id >= num_shards
  std::shared_ptr<Dataset> ds4 = SogouNews(dataset_dir, "test", 0, ShuffleMode::kFalse, 2, 2);
  EXPECT_NE(ds4, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter4 = ds4->CreateIterator();
  // Expect failure: invalid SogouNews input
  EXPECT_EQ(iter4, nullptr);
}

/// Feature: Test SogouNews Dataset(ShuffleMode=kFiles).
/// Description: Test SogouNews Dataset interface with different ShuffleMode.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestSogouNewsDatasetShuffleFilesA) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSogouNewsDatasetShuffleFilesA.";

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(130);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  std::string dataset_dir = datasets_root_path_ + "/testSogouNews/";
  std::vector<std::string> column_names = {"index", "title", "content"};

  std::shared_ptr<Dataset> ds = SogouNews(dataset_dir, "all" , 0, ShuffleMode::kFiles);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("index"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
      {"1","Make history","Su Bingtian's 100m breakthrough\\n 9.83"},
      {"1","Jefferson commented on thick eyebrow: he has the top five talents in the league, but he is not the"
      " top five","They say he has the talent of the top five in the league. The talent of the top five in the"
      " league is one of the most disrespectful statements. I say he has the talent of the top five in the league,"
      " but he is not the top five players because the top five players play every night."},
      {"4","Tesla price","Tesla reduced its price by 70000 yuan"},
      {"3","Group pictures: Liu Shishi's temperament in early autumn released a large piece of micro curly long"
      " hair, elegant, lazy, gentle and capable","Liu Shishi's latest group of cover magazine blockbusters are"
      " released. In the photos, Liu Shishi's long hair is slightly curly, or camel colored belted woolen coat,"
      " or plaid suit, which is gentle and elegant and beautiful to a new height."},
      {"1","Opening ceremony of the 14th National Games","On the evening of September 15, Beijing time, "
      "the 14th games of the people's Republic of China opened in Xi'an Olympic Sports Center Stadium, "
      "Shaanxi Province. Yang Qian, the first gold medalist of the Chinese delegation in the Tokyo Olympic"
      " Games and a Post-00 shooter, lit the main torch platform. From then on, to September 27, the 14th"
      " National Games flame will burn here for 12 days."},
      {"3","Ni Ni deduces elegant retro style in different styles","Ni Ni's latest group of magazine cover"
      " blockbusters released that wearing gift hats is cool, retro, unique and full of fashion expression."}
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

/// Feature: Test SogouNews Dataset(ShuffleMode=kGlobal).
/// Description: Test SogouNews Dataset interface with different ShuffleMode.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestSogouNewsDatasetShuffleFilesGlobal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSogouNewsDatasetShuffleFilesGlobal.";

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(130);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  std::string dataset_dir = datasets_root_path_ + "/testSogouNews/";
  std::vector<std::string> column_names = {"index", "title", "content"};

  std::shared_ptr<Dataset> ds = SogouNews(dataset_dir, "all" , 0, ShuffleMode::kGlobal);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("index"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
      {"1","Make history","Su Bingtian's 100m breakthrough\\n 9.83"},
      {"1","Jefferson commented on thick eyebrow: he has the top five talents in the league, but he is not the"
      " top five","They say he has the talent of the top five in the league. The talent of the top five in the"
      " league is one of the most disrespectful statements. I say he has the talent of the top five in the league,"
      " but he is not the top five players because the top five players play every night."},
      {"4","Tesla price","Tesla reduced its price by 70000 yuan"},
      {"3","Ni Ni deduces elegant retro style in different styles","Ni Ni's latest group of magazine cover"
      " blockbusters released that wearing gift hats is cool, retro, unique and full of fashion expression."},
      {"3","Group pictures: Liu Shishi's temperament in early autumn released a large piece of micro curly long"
      " hair, elegant, lazy, gentle and capable","Liu Shishi's latest group of cover magazine blockbusters are"
      " released. In the photos, Liu Shishi's long hair is slightly curly, or camel colored belted woolen coat,"
      " or plaid suit, which is gentle and elegant and beautiful to a new height."},
      {"1","Opening ceremony of the 14th National Games","On the evening of September 15, Beijing time, "
      "the 14th games of the people's Republic of China opened in Xi'an Olympic Sports Center Stadium, "
      "Shaanxi Province. Yang Qian, the first gold medalist of the Chinese delegation in the Tokyo Olympic"
      " Games and a Post-00 shooter, lit the main torch platform. From then on, to September 27, the 14th"
      " National Games flame will burn here for 12 days."}
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