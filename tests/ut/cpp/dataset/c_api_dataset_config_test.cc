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
#include "minddata/dataset/include/config.h"
#include "minddata/dataset/include/datasets.h"

using namespace mindspore::dataset;
using mindspore::dataset::ShuffleMode;
using mindspore::dataset::Tensor;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestPipeline, TestConfigSetting) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestConfigSetting.";
  // Test basic configuration setting

  // Save original configuration values
  auto original_num_parallel_workers = config::get_num_parallel_workers();
  auto original_prefetch_size = config::get_prefetch_size();
  auto original_seed = config::get_seed();
  auto original_monitor_sampling_interval = config::get_monitor_sampling_interval();

  // Load configuration from file
  std::string config_file_path = datasets_root_path_ + "/declient.cfg";
  auto load_status = config::load(config_file_path);
  EXPECT_EQ(load_status, true);

  // Test configuration loaded
  EXPECT_EQ(config::get_num_parallel_workers(), 8);
  EXPECT_EQ(config::get_prefetch_size(), 16);
  EXPECT_EQ(config::get_seed(), 5489);
  EXPECT_EQ(config::get_monitor_sampling_interval(), 15);

  // Set configuration
  auto status_set_num_parallel_workers = config::set_num_parallel_workers(2);
  auto status_set_prefetch_size = config::set_prefetch_size(4);
  auto status_set_seed = config::set_seed(5);
  auto status_set_monitor_sampling_interval = config::set_monitor_sampling_interval(45);
  EXPECT_EQ(status_set_num_parallel_workers, true);
  EXPECT_EQ(status_set_prefetch_size, true);
  EXPECT_EQ(status_set_seed, true);
  EXPECT_EQ(status_set_monitor_sampling_interval, true);

  // Test configuration set
  EXPECT_EQ(config::get_num_parallel_workers(), 2);
  EXPECT_EQ(config::get_prefetch_size(), 4);
  EXPECT_EQ(config::get_seed(), 5);
  EXPECT_EQ(config::get_monitor_sampling_interval(), 45);

  // Restore original configuration values
  config::set_num_parallel_workers(original_num_parallel_workers);
  config::set_prefetch_size(original_prefetch_size);
  config::set_seed(original_seed);
  config::set_monitor_sampling_interval(original_monitor_sampling_interval);
}

TEST_F(MindDataTestPipeline, TestConfigParamCheck) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestConfigParamCheck.";
  // Test configuration setting with wrong parameter

  // Save original configuration values
  auto original_num_parallel_workers = config::get_num_parallel_workers();
  auto original_prefetch_size = config::get_prefetch_size();
  auto original_seed = config::get_seed();
  auto original_monitor_sampling_interval = config::get_monitor_sampling_interval();

  // Load configuration from file with wrong path
  std::string config_file_path = datasets_root_path_ + "/not_exist.cfg";
  auto load_status = config::load(config_file_path);
  EXPECT_EQ(load_status, false);

  // Set configuration with wrong parameter
  auto status_set_num_parallel_workers = config::set_num_parallel_workers(0);
  auto status_set_prefetch_size = config::set_prefetch_size(0);
  auto status_set_seed = config::set_seed(-1);
  auto status_set_monitor_sampling_interval = config::set_monitor_sampling_interval(0);
  EXPECT_EQ(status_set_num_parallel_workers, false);
  EXPECT_EQ(status_set_prefetch_size, false);
  EXPECT_EQ(status_set_seed, false);
  EXPECT_EQ(status_set_monitor_sampling_interval, false);

  // Restore original configuration values
  config::set_num_parallel_workers(original_num_parallel_workers);
  config::set_prefetch_size(original_prefetch_size);
  config::set_seed(original_seed);
  config::set_monitor_sampling_interval(original_monitor_sampling_interval);
}

TEST_F(MindDataTestPipeline, TestShuffleWithSeed) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestShuffleWithSeed.";
  // Test deterministic shuffle with setting the seed

  // Save and set the seed
  uint32_t original_seed = config::get_seed();
  uint32_t original_num_parallel_workers = config::get_num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  config::set_seed(654);
  config::set_num_parallel_workers(1);

  // Create a TextFile Dataset with single text file which has three samples
  std::string text_file = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({text_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Shuffle the dataset with buffer_size=3
  ds = ds->Shuffle(3);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);
  EXPECT_NE(row.find("text"), row.end());

  std::vector<std::string> expected_result = {"Good luck to everyone.", "Be happy every day.", "This is a text file."};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["text"];

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
  config::set_seed(original_seed);
  config::set_num_parallel_workers(original_num_parallel_workers);
}

TEST_F(MindDataTestPipeline, TestCallShuffleTwice) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCallShuffleTwice.";
  // Test shuffle and repeat with setting the seed.
  // The second copy will be different from the first one because results will be different when calling shuffle twice.

  // Save and set the seed
  uint32_t original_seed = config::get_seed();
  uint32_t original_num_parallel_workers = config::get_num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  config::set_seed(654);
  config::set_num_parallel_workers(1);

  // Create a TextFile Dataset with single text file which has three samples
  std::string text_file = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({text_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Shuffle the dataset with buffer_size=3
  ds = ds->Shuffle(3);
  EXPECT_NE(ds, nullptr);

  // Repeat the dataset twice
  ds = ds->Repeat(2);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);
  EXPECT_NE(row.find("text"), row.end());

  std::vector<std::string> first_copy;
  std::vector<std::string> second_copy;

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["text"];
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    de_text->GetItemAt(&sv, {0});
    std::string ss(sv);
    MS_LOG(INFO) << "Text length: " << ss.length() << ", Text: " << ss.substr(0, 50);

    // The first three samples are the first copy and the rest are the second
    if (i < 3) {
      first_copy.push_back(ss);
    } else {
      second_copy.push_back(ss);
    }

    i++;
    iter->GetNextRow(&row);
  }

  // Expect 6 samples
  EXPECT_EQ(i, 6);

  // Compare the two copies which are deterministic difference
  for (int j = 0; j < 3; j++) {
    EXPECT_STRNE(first_copy.at(j).c_str(), second_copy.at(j).c_str());
  }

  // Manually terminate the pipeline
  iter->Stop();

  // Restore configuration
  config::set_seed(original_seed);
  config::set_num_parallel_workers(original_num_parallel_workers);
}
