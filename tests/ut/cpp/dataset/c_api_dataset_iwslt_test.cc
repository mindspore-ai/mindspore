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

/// Feature: Test IWSLT2016 Dataset.
/// Description: Read IWSLT2016Dataset data and get data.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestIWSLT2016DatasetBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestIWSLT2016DatasetBasic.";

  std::string dataset_dir = datasets_root_path_ + "/testIWSLT/IWSLT2016";

  std::shared_ptr<Dataset> ds =
    IWSLT2016(dataset_dir, "train", {"de", "en"}, "tst2013", "tst2014", 0, ShuffleMode::kFalse);
  std::vector<std::string> column_names = {"text", "translation"};

  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
    {"Code schreiben macht Freude.", "Writing code is a joy."},
    {"Ich hoffe in Zukunft weniger Überstunden machen zu können.", "I hope to work less overtime in the future."}};
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

/// Feature: Test IWSLT2016 Dataset.
/// Description: Read IWSLT2016Dataset data and get data (usage=valid).
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestIWSLT2016DatasetUsageValidBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestIWSLT2016DatasetUsageValidBasic.";

  std::string dataset_dir = datasets_root_path_ + "/testIWSLT/IWSLT2016";

  std::shared_ptr<Dataset> ds =
    IWSLT2016(dataset_dir, "valid", {"de", "en"}, "tst2013", "tst2014", 0, ShuffleMode::kFalse);
  std::vector<std::string> column_names = {"text", "translation"};

  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::vector<std::string>> expected_result = {{"heute hat es geregnet.", "it rained today."},
                                                           {"Leih mir ein Stück Papier.", "Lend me a piece of paper."}};
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

/// Feature: Test IWSLT2016 Dataset.
/// Description: Read IWSLT2016Dataset data and get data (usage=test).
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestIWSLT2016DatasetUsageTestBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestIWSLT2016DatasetUsageTestBasic.";

  std::string dataset_dir = datasets_root_path_ + "/testIWSLT/IWSLT2016";

  std::shared_ptr<Dataset> ds =
    IWSLT2016(dataset_dir, "test", {"de", "en"}, "tst2013", "tst2014", 0, ShuffleMode::kFalse);
  std::vector<std::string> column_names = {"text", "translation"};

  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
    {"Ich mag dich.", "I like you."}, {"Ich gebe dir eine Schultasche.", "I will give you a schoolbag."}};
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

/// Feature: Test IWSLT2016 Dataset.
/// Description: Read IWSLT2016Dataset data and get data (usage=all).
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestIWSLT2016DatasetUsageAllBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestIWSLT2016DatasetUsageAllBasic.";

  std::string dataset_dir = datasets_root_path_ + "/testIWSLT/IWSLT2016";

  std::shared_ptr<Dataset> ds =
    IWSLT2016(dataset_dir, "all", {"de", "en"}, "tst2013", "tst2014", 0, ShuffleMode::kFalse);
  std::vector<std::string> column_names = {"text", "translation"};
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
    {"Code schreiben macht Freude.", "Writing code is a joy."},
    {"heute hat es geregnet.", "it rained today."},
    {"Ich mag dich.", "I like you."},
    {"Ich hoffe in Zukunft weniger Überstunden machen zu können.", "I hope to work less overtime in the future."},
    {"Leih mir ein Stück Papier.", "Lend me a piece of paper."},
    {"Ich gebe dir eine Schultasche.", "I will give you a schoolbag."}};

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

/// Feature: Test IWSLT2016 Dataset.
/// Description: Includes tests for shape, type, size.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestIWSLT2016DatasetGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestIWSLT2016DatasetGetters.";

  std::string dataset_dir = datasets_root_path_ + "/testIWSLT/IWSLT2016";
  std::shared_ptr<Dataset> ds =
    IWSLT2016(dataset_dir, "train", {"de", "en"}, "tst2013", "tst2014", 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::vector<DataType> types = ToDETypes(ds->GetOutputTypes());
  std::vector<TensorShape> shapes = ToTensorShapeVec(ds->GetOutputShapes());
  EXPECT_EQ(types.size(), 2);
  EXPECT_EQ(types[0].ToString(), "string");
  EXPECT_EQ(types[1].ToString(), "string");
  EXPECT_EQ(shapes.size(), 2);
  EXPECT_EQ(shapes[0].ToString(), "<>");
  EXPECT_EQ(shapes[1].ToString(), "<>");

  std::vector<std::string> column_names = {"text", "translation"};
  EXPECT_EQ(ds->GetColumnNames(), column_names);
  EXPECT_EQ(ds->GetDatasetSize(), 2);
}

/// Feature: Test IWSLT2016 Dataset.
/// Description: Test whether the interface meets expectations when NumSamples is equal to 2.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestIWSLT2016DatasetNumSamples) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestIWSLT2016DatasetNumSamples.";

  std::string dataset_dir = datasets_root_path_ + "/testIWSLT/IWSLT2016";
  std::shared_ptr<Dataset> ds =
    IWSLT2016(dataset_dir, "train", {"de", "en"}, "tst2013", "tst2014", 2, ShuffleMode::kFalse);
  std::vector<std::string> column_names = {"text", "translation"};
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
    {"Code schreiben macht Freude.", "Writing code is a joy."},
    {"Ich hoffe in Zukunft weniger Überstunden machen zu können.", "I hope to work less overtime in the future."}};

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

/// Feature: Test IWSLT2016 Dataset.
/// Description: Test interface in a distributed state.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestIWSLT2016DatasetDistribution) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestIWSLT2016DatasetDistribution.";

  // Create a IWSLT2016Dataset.
  std::string dataset_dir = datasets_root_path_ + "/testIWSLT/IWSLT2016";
  std::shared_ptr<Dataset> ds =
    IWSLT2016(dataset_dir, "train", {"de", "en"}, "tst2013", "tst2014", 0, ShuffleMode::kFalse, 2);
  std::vector<std::string> column_names = {"text", "translation"};
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
    {"Code schreiben macht Freude.", "Writing code is a joy."},
    {"Ich hoffe in Zukunft weniger Überstunden machen zu können.", "I hope to work less overtime in the future."}};

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

/// Feature: Test IWSLT2016 Dataset.
/// Description: Test the wrong input.
/// Expectation: Unable to read in data.
TEST_F(MindDataTestPipeline, TestIWSLT2016DatasetFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestIWSLT2016DatasetFail.";

  std::string dataset_dir = datasets_root_path_ + "/testIWSLT/IWSLT2016";

  // Create a IWSLT2016 Dataset with not exist file.
  std::shared_ptr<Dataset> ds0 = IWSLT2016("invalid_dir", "train", {"de", "en"}, "tst2013", "tst2014");
  EXPECT_NE(ds0, nullptr);
  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter0 = ds0->CreateIterator();
  // Expect failure: invalid IWSLT input.
  EXPECT_EQ(iter0, nullptr);
  // Create a IWSLT2016 Dataset with invalid usage.
  std::shared_ptr<Dataset> ds1 = IWSLT2016(dataset_dir, "invalid_usage", {"de", "en"}, "tst2013", "tst2014");
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid IWSLT input.
  EXPECT_EQ(iter1, nullptr);

  // Create a IWSLT2016 Dataset with invalid language_pair[0] (src_language).
  std::shared_ptr<Dataset> ds2 = IWSLT2016(dataset_dir, "train", {"invalid", "en"}, "tst2013", "tst2014");
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid IWSLT input.
  EXPECT_EQ(iter2, nullptr);

  // Create a IWSLT2016 Dataset with invalid language_pair[1] (target_language).
  std::shared_ptr<Dataset> ds3 = IWSLT2016(dataset_dir, "train", {"de", "invalid"}, "tst2013", "tst2014");
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid IWSLT input
  EXPECT_EQ(iter3, nullptr);

  // Create a IWSLT2016 Dataset with invalid valid_set.
  std::shared_ptr<Dataset> ds4 = IWSLT2016(dataset_dir, "train", {"de", "en"}, "invalid", "tst2014");
  EXPECT_NE(ds4, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter4 = ds4->CreateIterator();
  // Expect failure: invalid IWSLT input
  EXPECT_EQ(iter4, nullptr);

  // Create a IWSLT2016 Dataset with invalid test_set.
  std::shared_ptr<Dataset> ds5 = IWSLT2016(dataset_dir, "train", {"de", "en"}, "tst2013", "invalid");
  EXPECT_NE(ds5, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter5 = ds5->CreateIterator();
  // Expect failure: invalid IWSLT input.
  EXPECT_EQ(iter5, nullptr);

  // Test invalid num_samples < -1.
  std::shared_ptr<Dataset> ds6 = IWSLT2016(dataset_dir, "train", {"de", "en"}, "tst2013", "tst2014", -1);
  EXPECT_NE(ds2, nullptr);
  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter6 = ds6->CreateIterator();
  // Expect failure: invalid IWSLT input.
  EXPECT_EQ(iter6, nullptr);

  // Test invalid num_shards < 1.
  std::shared_ptr<Dataset> ds7 =
    IWSLT2016(dataset_dir, "train", {"de", "en"}, "tst2013", "tst2014", 0, ShuffleMode::kFalse, 0);
  EXPECT_NE(ds7, nullptr);
  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter7 = ds7->CreateIterator();
  // Expect failure: invalid IWSLT input.
  EXPECT_EQ(iter7, nullptr);

  // Test invalid shard_id >= num_shards.
  std::shared_ptr<Dataset> ds8 =
    IWSLT2016(dataset_dir, "train", {"de", "en"}, "tst2013", "tst2014", 0, ShuffleMode::kFalse, 2, 2);
  EXPECT_NE(ds8, nullptr);
  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter8 = ds8->CreateIterator();
  // Expect failure: invalid IWSLT input.
  EXPECT_EQ(iter8, nullptr);
}

/// Feature: Test IWSLT2016 Dataset.
/// Description: Test IWSLT2016 Dataset interface in pipeline.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestIWSLT2016DatasetBasicWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestIWSLT2016DatasetBasicWithPipeline.";

  // Create two IWSLT2016 Dataset, with single IWSLT2016 file.
  std::string dataset_dir = datasets_root_path_ + "/testIWSLT/IWSLT2016";

  std::shared_ptr<Dataset> ds1 =
    IWSLT2016(dataset_dir, "train", {"de", "en"}, "tst2013", "tst2014", 0, ShuffleMode::kFalse);
  std::shared_ptr<Dataset> ds2 =
    IWSLT2016(dataset_dir, "train", {"de", "en"}, "tst2013", "tst2014", 0, ShuffleMode::kFalse);
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
  std::vector<std::string> column_project = {"text"};
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

  EXPECT_NE(row.find("text"), row.end());
  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["text"];
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  // Expect 10 samples.
  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: Test IWSLT2016 Dataset.
/// Description: Test IWSLT2016 Dataset interface with different ShuffleMode.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestIWSLT2016DatasetShuffleFilesA) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestIWSLT2016DatasetShuffleFilesA.";

  // Set configuration.
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(130);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  std::string dataset_dir = datasets_root_path_ + "/testIWSLT/IWSLT2016";
  std::vector<std::string> column_names = {"text", "translation"};

  std::shared_ptr<Dataset> ds =
    IWSLT2016(dataset_dir, "all", {"de", "en"}, "tst2013", "tst2014", 0, ShuffleMode::kFiles);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
    {"Ich mag dich.", "I like you."},
    {"Code schreiben macht Freude.", "Writing code is a joy."},
    {"heute hat es geregnet.", "it rained today."},
    {"Ich gebe dir eine Schultasche.", "I will give you a schoolbag."},
    {"Ich hoffe in Zukunft weniger Überstunden machen zu können.", "I hope to work less overtime in the future."},
    {"Leih mir ein Stück Papier.", "Lend me a piece of paper."}};

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

/// Feature: Test IWSLT2016 Dataset.
/// Description: Test IWSLT2016 Dataset interface with different ShuffleMode.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TesIWSLT2016DatasetShuffleFilesGlobal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TesIWSLT2016DatasetShuffleFilesGlobal.";

  // Set configuration.
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(130);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  std::string dataset_dir = datasets_root_path_ + "/testIWSLT/IWSLT2016";
  std::vector<std::string> column_names = {"text", "translation"};

  std::shared_ptr<Dataset> ds =
    IWSLT2016(dataset_dir, "all", {"de", "en"}, "tst2013", "tst2014", 0, ShuffleMode::kGlobal);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
    {"Ich mag dich.", "I like you."},
    {"Code schreiben macht Freude.", "Writing code is a joy."},
    {"heute hat es geregnet.", "it rained today."},
    {"Leih mir ein Stück Papier.", "Lend me a piece of paper."},
    {"Ich gebe dir eine Schultasche.", "I will give you a schoolbag."},
    {"Ich hoffe in Zukunft weniger Überstunden machen zu können.", "I hope to work less overtime in the future."}};

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

/// Feature: Test IWSLT2017 Dataset.
/// Description: Read IWSLT2017Dataset data and get data.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestIWSLT2017DatasetBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestIWSLT2017DatasetBasic.";

  std::string dataset_dir = datasets_root_path_ + "/testIWSLT/IWSLT2017";
  std::shared_ptr<Dataset> ds = IWSLT2017(dataset_dir, "train", {"de", "en"}, 0, ShuffleMode::kFalse);
  std::vector<std::string> column_names = {"text", "translation"};
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
    {"Schönes Wetter heute.", "The weather is nice today."},
    {"Ich bin heute gut gelaunt.", "I am in a good mood today."}};
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

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: Test IWSLT2017 Dataset.
/// Description: Read IWSLT2017Dataset data and get data (usage=valid).
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestIWSLT2017DatasetUsageValidBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestIWSLT2017DatasetUsageValidBasic.";

  std::string dataset_dir = datasets_root_path_ + "/testIWSLT/IWSLT2017";

  std::shared_ptr<Dataset> ds = IWSLT2017(dataset_dir, "valid", {"de", "en"}, 0, ShuffleMode::kFalse);
  std::vector<std::string> column_names = {"text", "translation"};

  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
    {"Ich kann meinen Code nicht zu Ende schreiben.", "I can't finish writing my code."},
    {"Vielleicht muss ich Überstunden machen.", "I might have to work overtime."}};
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

/// Feature: Test IWSLT2017 Dataset.
/// Description: Read IWSLT2017Dataset data and get data (usage=test).
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestIWSLT2017DatasetUsageTestBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestIWSLT2017DatasetUsageTestBasic.";

  std::string dataset_dir = datasets_root_path_ + "/testIWSLT/IWSLT2017";

  std::shared_ptr<Dataset> ds = IWSLT2017(dataset_dir, "test", {"de", "en"}, 0, ShuffleMode::kFalse);
  std::vector<std::string> column_names = {"text", "translation"};

  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
    {"Heute gehe ich ins Labor.", "Today i'm going to the lab."},
    {"Ich schlafe jetzt wieder ein.", "I am going back to sleep now."}};
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

/// Feature: Test IWSLT2017 Dataset.
/// Description: Read IWSLT2017Dataset data and get data (usage=all).
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestIWSLT2017DatasetUsageAllBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestIWSLT2017DatasetUsageAllBasic.";

  std::string dataset_dir = datasets_root_path_ + "/testIWSLT/IWSLT2017";
  std::shared_ptr<Dataset> ds = IWSLT2017(dataset_dir, "all", {"de", "en"}, 0, ShuffleMode::kFalse);
  std::vector<std::string> column_names = {"text", "translation"};
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::vector<std::string>> expected_result = {
    {"Schönes Wetter heute.", "The weather is nice today."},
    {"Ich kann meinen Code nicht zu Ende schreiben.", "I can't finish writing my code."},
    {"Heute gehe ich ins Labor.", "Today i'm going to the lab."},
    {"Ich bin heute gut gelaunt.", "I am in a good mood today."},
    {"Vielleicht muss ich Überstunden machen.", "I might have to work overtime."},
    {"Ich schlafe jetzt wieder ein.", "I am going back to sleep now."}};

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

/// Feature: Test IWSLT2017 Dataset.
/// Description: Test the wrong input.
/// Expectation: Unable to read in data.
TEST_F(MindDataTestPipeline, TestIWSLT2017DatasetFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestIWSLT2017DatasetFail.";

  std::string dataset_dir = datasets_root_path_ + "/testIWSLT/IWSLT2017";

  // Create a IWSLT2017 Dataset with not exist file.
  std::shared_ptr<Dataset> ds0 = IWSLT2017("invalid_dir", "train", {"de", "en"});
  EXPECT_NE(ds0, nullptr);
  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter0 = ds0->CreateIterator();
  // Expect failure: invalid IWSLT input.
  EXPECT_EQ(iter0, nullptr);

  // Create a IWSLT2017 Dataset with invalid usage.
  std::shared_ptr<Dataset> ds1 = IWSLT2017(dataset_dir, "invalid_usage", {"de", "en"});
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid IWSLT input.
  EXPECT_EQ(iter1, nullptr);

  // Create a IWSLT2017 Dataset with invalid language_pair[0](src_language).
  std::shared_ptr<Dataset> ds2 = IWSLT2017(dataset_dir, "train", {"invalid", "en"});
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid IWSLT input
  EXPECT_EQ(iter2, nullptr);

  // Create a IWSLT2017 Dataset with invalid language_pair[1](target_language.
  std::shared_ptr<Dataset> ds3 = IWSLT2016(dataset_dir, "train", {"de", "invalid"}, "tst2013", "tst2014");
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid IWSLT input.
  EXPECT_EQ(iter3, nullptr);

  // Test invalid num_samples < -1.
  std::shared_ptr<Dataset> ds4 = IWSLT2016(dataset_dir, "train", {"de", "en"}, "tst2013", "tst2014", -1);
  EXPECT_NE(ds4, nullptr);
  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter4 = ds4->CreateIterator();
  // Expect failure: invalid IWSLT input.
  EXPECT_EQ(iter4, nullptr);

  // Test invalid num_shards < 1.
  std::shared_ptr<Dataset> ds5 =
    IWSLT2016(dataset_dir, "train", {"de", "en"}, "tst2013", "tst2014", 0, ShuffleMode::kFalse, 0);
  EXPECT_NE(ds5, nullptr);
  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter5 = ds5->CreateIterator();
  // Expect failure: invalid IWSLT input.
  EXPECT_EQ(iter5, nullptr);

  // Test invalid shard_id >= num_shards.
  std::shared_ptr<Dataset> ds6 =
    IWSLT2016(dataset_dir, "train", {"de", "en"}, "tst2013", "tst2014", 0, ShuffleMode::kFalse, 2, 2);
  EXPECT_NE(ds6, nullptr);
  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter6 = ds6->CreateIterator();
  // Expect failure: invalid IWSLT input.
  EXPECT_EQ(iter6, nullptr);
}

/// Feature: Test IWSLT2017 Dataset.
/// Description: Test IWSLT2017 Dataset interface in pipeline.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestIWSLT2017DatasetBasicWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestIWSLT2017DatasetBasicWithPipeline.";

  // Create two IWSLT2017 Dataset, with single IWSLT2017 file.
  std::string dataset_dir = datasets_root_path_ + "/testIWSLT/IWSLT2017";

  std::shared_ptr<Dataset> ds1 = IWSLT2017(dataset_dir, "train", {"de", "en"}, 0, ShuffleMode::kFalse);
  std::shared_ptr<Dataset> ds2 = IWSLT2017(dataset_dir, "train", {"de", "en"}, 0, ShuffleMode::kFalse);
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
  std::vector<std::string> column_project = {"text"};
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

  EXPECT_NE(row.find("text"), row.end());
  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["text"];
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  // Expect 10 samples.
  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline.
  iter->Stop();
}
