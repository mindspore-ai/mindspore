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
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/include/datasets.h"

using namespace mindspore::dataset;
using mindspore::dataset::GlobalContext;
using mindspore::dataset::ShuffleMode;
using mindspore::dataset::Tensor;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestPipeline, TestCLUEDatasetAFQMC) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCLUEDatasetAFQMC.";

  // Create a CLUEFile Dataset, with single CLUE file
  std::string train_file = datasets_root_path_ + "/testCLUE/afqmc/train.json";
  std::string test_file = datasets_root_path_ + "/testCLUE/afqmc/test.json";
  std::string eval_file = datasets_root_path_ + "/testCLUE/afqmc/dev.json";
  std::string task = "AFQMC";
  std::string usage = "train";
  std::shared_ptr<Dataset> ds = CLUE({train_file}, task, usage, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  EXPECT_NE(row.find("sentence1"), row.end());
  std::vector<std::string> expected_result = {"蚂蚁借呗等额还款能否换成先息后本", "蚂蚁花呗说我违约了",
                                               "帮我看看本月花呗账单结清了没"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["sentence1"];
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    de_text->GetItemAt(&sv, {0});
    std::string ss(sv);
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    iter->GetNextRow(&row);
    i++;
  }

  // Expect 3 samples
  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();

  // test
  usage = "test";
  expected_result = {"借呗取消的时间", "网商贷用什么方法转变成借呗", "我的借呗为什么开通不了"};
  ds = CLUE({test_file}, task, usage, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  iter->GetNextRow(&row);
  EXPECT_NE(row.find("sentence1"), row.end());
  i = 0;
  while (row.size() != 0) {
    auto text = row["sentence1"];
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    de_text->GetItemAt(&sv, {0});
    std::string ss(sv);
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());
    iter->GetNextRow(&row);
    i++;
  }
  iter->Stop();

  // eval
  usage = "eval";
  expected_result = {"你有花呗吗", "吃饭能用花呗吗", "蚂蚁花呗支付金额有什么限制"};
  ds = CLUE({eval_file}, task, usage, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  iter->GetNextRow(&row);
  EXPECT_NE(row.find("sentence1"), row.end());
  i = 0;
  while (row.size() != 0) {
    auto text = row["sentence1"];
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    de_text->GetItemAt(&sv, {0});
    std::string ss(sv);
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());
    iter->GetNextRow(&row);
    i++;
  }
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestCLUEDatasetBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCLUEDatasetBasic.";

  // Create a CLUEFile Dataset, with single CLUE file
  std::string clue_file = datasets_root_path_ + "/testCLUE/afqmc/train.json";
  std::string task = "AFQMC";
  std::string usage = "train";
  std::shared_ptr<Dataset> ds = CLUE({clue_file}, task, usage, 2);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  EXPECT_NE(row.find("sentence1"), row.end());
  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["sentence1"];
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    i++;
    iter->GetNextRow(&row);
  }

  // Expect 2 samples
  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestCLUEDatasetBasicWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCLUEDatasetBasicWithPipeline.";

  // Create two CLUEFile Dataset, with single CLUE file
  std::string clue_file = datasets_root_path_ + "/testCLUE/afqmc/train.json";
  std::string task = "AFQMC";
  std::string usage = "train";
  std::shared_ptr<Dataset> ds1 = CLUE({clue_file}, task, usage, 2);
  std::shared_ptr<Dataset> ds2 = CLUE({clue_file}, task, usage, 2);
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
  std::vector<std::string> column_project = {"sentence1"};
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
  iter->GetNextRow(&row);

  EXPECT_NE(row.find("sentence1"), row.end());
  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["sentence1"];
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    i++;
    iter->GetNextRow(&row);
  }

  // Expect 10 samples
  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestCLUEGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCLUEGetters.";

  // Create a CLUEFile Dataset, with single CLUE file
  std::string clue_file = datasets_root_path_ + "/testCLUE/afqmc/train.json";
  std::string task = "AFQMC";
  std::string usage = "train";
  std::shared_ptr<Dataset> ds = CLUE({clue_file}, task, usage, 2);
  std::vector<std::string> column_names = {"label", "sentence1", "sentence2"};
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 2);
  EXPECT_EQ(ds->GetColumnNames(), column_names);
}

TEST_F(MindDataTestPipeline, TestCLUEDatasetCMNLI) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCLUEDatasetCMNLI.";

  // Create a CLUEFile Dataset, with single CLUE file
  std::string clue_file = datasets_root_path_ + "/testCLUE/cmnli/train.json";
  std::string task = "CMNLI";
  std::string usage = "train";
  std::shared_ptr<Dataset> ds = CLUE({clue_file}, task, usage, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  EXPECT_NE(row.find("sentence1"), row.end());
  std::vector<std::string> expected_result = {"你应该给这件衣服定一个价格。", "我怎么知道他要说什么", "向左。"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["sentence1"];
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    de_text->GetItemAt(&sv, {0});
    std::string ss(sv);
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    iter->GetNextRow(&row);
    i++;
  }

  // Expect 3 samples
  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestCLUEDatasetCSL) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCLUEDatasetCSL.";

  // Create a CLUEFile Dataset, with single CLUE file
  std::string clue_file = datasets_root_path_ + "/testCLUE/csl/train.json";
  std::string task = "CSL";
  std::string usage = "train";
  std::shared_ptr<Dataset> ds = CLUE({clue_file}, task, usage, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  EXPECT_NE(row.find("abst"), row.end());
  std::vector<std::string> expected_result = {"这是一段长文本", "这是一段长文本", "这是一段长文本"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["abst"];
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    de_text->GetItemAt(&sv, {0});
    std::string ss(sv);
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    iter->GetNextRow(&row);
    i++;
  }

  // Expect 3 samples
  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestCLUEDatasetDistribution) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCLUEDatasetDistribution.";

  // Create a CLUEFile Dataset, with single CLUE file
  std::string clue_file = datasets_root_path_ + "/testCLUE/afqmc/train.json";
  std::string task = "AFQMC";
  std::string usage = "train";
  std::shared_ptr<Dataset> ds = CLUE({clue_file}, task, usage, 0, ShuffleMode::kGlobal, 3, 0);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  EXPECT_NE(row.find("sentence1"), row.end());
  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["sentence1"];
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    i++;
    iter->GetNextRow(&row);
  }

  // Expect 1 samples
  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestCLUEDatasetFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCLUEDatasetFail.";
  // Create a CLUE Dataset
  std::string clue_file = datasets_root_path_ + "/testCLUE/wsc/train.json";
  std::string task = "WSC";
  std::string usage = "train";
  std::string invalid_clue_file = "./NotExistFile";

  std::shared_ptr<Dataset> ds0 = CLUE({}, task, usage);
  EXPECT_NE(ds0, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter0 = ds0->CreateIterator();
  // Expect failure: invalid CLUE input
  EXPECT_EQ(iter0, nullptr);

  std::shared_ptr<Dataset> ds1 = CLUE({invalid_clue_file}, task, usage);
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid CLUE input
  EXPECT_EQ(iter1, nullptr);

  std::shared_ptr<Dataset> ds2 = CLUE({clue_file}, "invalid_task", usage);
  EXPECT_NE(ds2, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid CLUE input
  EXPECT_EQ(iter2, nullptr);

  std::shared_ptr<Dataset> ds3 = CLUE({clue_file}, task, "invalid_usage");
  EXPECT_NE(ds3, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid CLUE input
  EXPECT_EQ(iter3, nullptr);

  std::shared_ptr<Dataset> ds4 = CLUE({clue_file}, task, usage, 0, ShuffleMode::kGlobal, 2, 2);
  EXPECT_NE(ds4, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter4 = ds4->CreateIterator();
  // Expect failure: invalid CLUE input
  EXPECT_EQ(iter4, nullptr);

  std::shared_ptr<Dataset> ds5 = CLUE({clue_file}, task, usage, -1, ShuffleMode::kGlobal);
  EXPECT_NE(ds5, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter5 = ds5->CreateIterator();
  // Expect failure: invalid CLUE input
  EXPECT_EQ(iter5, nullptr);

  std::shared_ptr<Dataset> ds6 = CLUE({clue_file}, task, usage, 0, ShuffleMode::kGlobal, -1);
  EXPECT_NE(ds6, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter6 = ds6->CreateIterator();
  // Expect failure: invalid CLUE input
  EXPECT_EQ(iter6, nullptr);

  std::shared_ptr<Dataset> ds7 = CLUE({clue_file}, task, usage, 0, ShuffleMode::kGlobal, 0, -1);
  EXPECT_NE(ds7, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter7 = ds7->CreateIterator();
  // Expect failure: invalid CLUE input
  EXPECT_EQ(iter7, nullptr);
}

TEST_F(MindDataTestPipeline, TestCLUEDatasetIFLYTEK) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCLUEDatasetIFLYTEK.";

  // Create a CLUEFile Dataset, with single CLUE file
  std::string clue_file = datasets_root_path_ + "/testCLUE/iflytek/train.json";
  std::string task = "IFLYTEK";
  std::string usage = "train";
  std::shared_ptr<Dataset> ds = CLUE({clue_file}, task, usage, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  EXPECT_NE(row.find("sentence"), row.end());
  std::vector<std::string> expected_result = {"第一个文本", "第二个文本", "第三个文本"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["sentence"];
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    de_text->GetItemAt(&sv, {0});
    std::string ss(sv);
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    iter->GetNextRow(&row);
    i++;
  }

  // Expect 3 samples
  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestCLUEDatasetShuffleFilesA) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCLUEDatasetShuffleFilesA.";
  // Test CLUE Dataset with files shuffle, num_parallel_workers=1

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(135);
  GlobalContext::config_manager()->set_num_parallel_workers(1);

  // Create a CLUE Dataset, with two text files, dev.json and train.json, in lexicographical order
  // Note: train.json has 3 rows
  // Note: dev.json has 3 rows
  // Use default of all samples
  // They have the same keywords
  // Set shuffle to files shuffle
  std::string clue_file1 = datasets_root_path_ + "/testCLUE/afqmc/train.json";
  std::string clue_file2 = datasets_root_path_ + "/testCLUE/afqmc/dev.json";
  std::string task = "AFQMC";
  std::string usage = "train";
  std::shared_ptr<Dataset> ds = CLUE({clue_file2, clue_file1}, task, usage, 0, ShuffleMode::kFiles);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  EXPECT_NE(row.find("sentence1"), row.end());
  std::vector<std::string> expected_result = {"你有花呗吗",
                                              "吃饭能用花呗吗",
                                              "蚂蚁花呗支付金额有什么限制",
                                              "蚂蚁借呗等额还款能否换成先息后本",
                                              "蚂蚁花呗说我违约了",
                                              "帮我看看本月花呗账单结清了没"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["sentence1"];
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

  // Expect 3 + 3 = 6 samples
  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();

  // Restore configuration
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

TEST_F(MindDataTestPipeline, TestCLUEDatasetShuffleFilesB) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCLUEDatasetShuffleFilesB.";
  // Test CLUE Dataset with files shuffle, num_parallel_workers=1

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(135);
  GlobalContext::config_manager()->set_num_parallel_workers(1);

  // Create a CLUE Dataset, with two text files, train.json and dev.json, in non-lexicographical order
  // Note: train.json has 3 rows
  // Note: dev.json has 3 rows
  // Use default of all samples
  // They have the same keywords
  // Set shuffle to files shuffle
  std::string clue_file1 = datasets_root_path_ + "/testCLUE/afqmc/train.json";
  std::string clue_file2 = datasets_root_path_ + "/testCLUE/afqmc/dev.json";
  std::string task = "AFQMC";
  std::string usage = "train";
  std::shared_ptr<Dataset> ds = CLUE({clue_file1, clue_file2}, task, usage, 0, ShuffleMode::kFiles);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  EXPECT_NE(row.find("sentence1"), row.end());
  std::vector<std::string> expected_result = {"你有花呗吗",
                                              "吃饭能用花呗吗",
                                              "蚂蚁花呗支付金额有什么限制",
                                              "蚂蚁借呗等额还款能否换成先息后本",
                                              "蚂蚁花呗说我违约了",
                                              "帮我看看本月花呗账单结清了没"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["sentence1"];
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    de_text->GetItemAt(&sv, {0});
    std::string ss(sv);
    // Compare against expected result
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());
    i++;
    iter->GetNextRow(&row);
  }

  // Expect 3 + 3 = 6 samples
  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();

  // Restore configuration
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

TEST_F(MindDataTestPipeline, TestCLUEDatasetShuffleGlobal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCLUEDatasetShuffleGlobal.";
  // Test CLUE Dataset with GLOBLE shuffle

  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(135);
  GlobalContext::config_manager()->set_num_parallel_workers(4);

  // Create a CLUEFile Dataset, with single CLUE file
  std::string clue_file = datasets_root_path_ + "/testCLUE/afqmc/train.json";
  std::string task = "AFQMC";
  std::string usage = "train";
  std::shared_ptr<Dataset> ds = CLUE({clue_file}, task, usage, 0, ShuffleMode::kGlobal);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  EXPECT_NE(row.find("sentence1"), row.end());
  std::vector<std::string> expected_result = {"蚂蚁花呗说我违约了", "帮我看看本月花呗账单结清了没",
                                              "蚂蚁借呗等额还款能否换成先息后本"};
  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["sentence1"];
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    de_text->GetItemAt(&sv, {0});
    std::string ss(sv);
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    i++;
    iter->GetNextRow(&row);
  }

  // Expect 3 samples
  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();

  // Restore configuration
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

TEST_F(MindDataTestPipeline, TestCLUEDatasetTNEWS) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCLUEDatasetTNEWS.";

  // Create a CLUEFile Dataset, with single CLUE file
  std::string clue_file = datasets_root_path_ + "/testCLUE/tnews/train.json";
  std::string task = "TNEWS";
  std::string usage = "train";
  std::shared_ptr<Dataset> ds = CLUE({clue_file}, task, usage, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  EXPECT_NE(row.find("sentence"), row.end());
  std::vector<std::string> expected_result = {"新闻1", "新闻2", "新闻3"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["sentence"];
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    de_text->GetItemAt(&sv, {0});
    std::string ss(sv);
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    iter->GetNextRow(&row);
    i++;
  }

  // Expect 3 samples
  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestCLUEDatasetWSC) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCLUEDatasetWSC.";

  // Create a CLUEFile Dataset, with single CLUE file
  std::string clue_file = datasets_root_path_ + "/testCLUE/wsc/train.json";
  std::string task = "WSC";
  std::string usage = "train";
  std::shared_ptr<Dataset> ds = CLUE({clue_file}, task, usage, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  EXPECT_NE(row.find("text"), row.end());
  std::vector<std::string> expected_result = {"小明呢，他在哪？", "小红刚刚看到小明，他在操场",
                                              "等小明回来，小张你叫他交作业"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto text = row["text"];
    std::shared_ptr<Tensor> de_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(text, &de_text));
    std::string_view sv;
    de_text->GetItemAt(&sv, {0});
    std::string ss(sv);
    EXPECT_STREQ(ss.c_str(), expected_result[i].c_str());
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    iter->GetNextRow(&row);
    i++;
  }

  // Expect 3 samples
  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}
