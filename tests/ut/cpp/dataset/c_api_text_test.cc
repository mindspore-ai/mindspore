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
#include <memory>
#include <vector>
#include <string>

#include "common/common.h"
#include "include/api/status.h"
#include "minddata/dataset/include/config.h"
#include "minddata/dataset/include/datasets.h"
#include "minddata/dataset/include/text.h"
#include "minddata/dataset/include/transforms.h"
#include "minddata/dataset/text/vocab.h"

using namespace mindspore::dataset;
using mindspore::Status;
using mindspore::dataset::ShuffleMode;
using mindspore::dataset::Tensor;
using mindspore::dataset::Vocab;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestPipeline, TestBasicTokenizerSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBasicTokenizerSuccess1.";
  // Test BasicTokenizer with default parameters

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/basic_tokenizer.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create Take operation on ds
  ds = ds->Take(6);
  EXPECT_NE(ds, nullptr);

  // Create BasicTokenizer operation on ds
  std::shared_ptr<TensorTransform> basic_tokenizer = std::make_shared<text::BasicTokenizer>();
  EXPECT_NE(basic_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({basic_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::vector<std::string>> expected = {
    {"Welcome", "to", "Beijing", "北", "京", "欢", "迎", "您"},
    {"長", "風", "破", "浪", "會", "有", "時", "，", "直", "掛", "雲", "帆", "濟", "滄", "海"},
    {"😀", "嘿", "嘿", "😃", "哈", "哈", "😄", "大", "笑", "😁", "嘻", "嘻"},
    {"明", "朝", "（", "1368", "—",  "1644", "年", "）", "和", "清", "朝", "（", "1644", "—",  "1911", "年", "）",
     "，", "是", "中", "国",   "封", "建",   "王", "朝", "史", "上", "最", "后", "两",   "个", "朝",   "代"},
    {"明", "代",   "（", "1368",     "-",  "1644", "）",      "と", "清", "代",    "（", "1644",
     "-",  "1911", "）", "は",       "、", "中",   "国",      "の", "封", "建",    "王", "朝",
     "の", "歴",   "史", "における", "最", "後",   "の2つの", "王", "朝", "でした"},
    {"명나라", "(", "1368", "-",    "1644", ")",      "와",       "청나라", "(",  "1644",    "-",
     "1911",   ")", "는",   "중국", "봉건", "왕조의", "역사에서", "마지막", "두", "왕조였다"}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestBasicTokenizerSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBasicTokenizerSuccess2.";
  // Test BasicTokenizer with lower_case true

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/basic_tokenizer.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create Skip operation on ds
  ds = ds->Skip(6);
  EXPECT_NE(ds, nullptr);

  // Create BasicTokenizer operation on ds
  std::shared_ptr<TensorTransform> basic_tokenizer = std::make_shared<text::BasicTokenizer>(true);
  EXPECT_NE(basic_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({basic_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::string> expected = {"this", "is", "a", "funky", "string"};
  std::shared_ptr<Tensor> de_expected_tensor;
  ASSERT_OK(Tensor::CreateFromVector(expected, &de_expected_tensor));
  mindspore::MSTensor expected_tensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestBasicTokenizerSuccess3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBasicTokenizerSuccess3.";
  // Test BasicTokenizer with with_offsets true and lower_case true

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/basic_tokenizer.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create Skip operation on ds
  ds = ds->Skip(6);
  EXPECT_NE(ds, nullptr);

  // Create BasicTokenizer operation on ds
  std::shared_ptr<TensorTransform> basic_tokenizer =
    std::make_shared<text::BasicTokenizer>(true, false, NormalizeForm::kNone, true, true);
  EXPECT_NE(basic_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({basic_tokenizer}, {"text"}, {"token", "offsets_start", "offsets_limit"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::string> expected_tokens = {"this", "is", "a", "funky", "string"};
  std::vector<uint32_t> expected_offsets_start = {0, 5, 8, 10, 16};
  std::vector<uint32_t> expected_offsets_limit = {4, 7, 9, 15, 22};

  std::shared_ptr<Tensor> de_expected_tokens;
  ASSERT_OK(Tensor::CreateFromVector(expected_tokens, &de_expected_tokens));
  mindspore::MSTensor ms_expected_tokens =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tokens));

  std::shared_ptr<Tensor> de_expected_offsets_start;
  ASSERT_OK(Tensor::CreateFromVector(expected_offsets_start, &de_expected_offsets_start));
  mindspore::MSTensor ms_expected_offsets_start =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_offsets_start));

  std::shared_ptr<Tensor> de_expected_offsets_limit;
  ASSERT_OK(Tensor::CreateFromVector(expected_offsets_limit, &de_expected_offsets_limit));
  mindspore::MSTensor ms_expected_offsets_limit =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_offsets_limit));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["token"];
    EXPECT_MSTENSOR_EQ(ind, ms_expected_tokens);

    auto start = row["offsets_start"];
    EXPECT_MSTENSOR_EQ(start, ms_expected_offsets_start);

    auto limit = row["offsets_limit"];
    EXPECT_MSTENSOR_EQ(limit, ms_expected_offsets_limit);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

std::vector<std::string> list = {
  "床", "前", "明",    "月",    "光",    "疑",    "是",      "地",        "上",        "霜",   "举",    "头",
  "望", "低", "思",    "故",    "乡",    "繁",    "體",      "字",        "嘿",        "哈",   "大",    "笑",
  "嘻", "i",  "am",    "mak",   "make",  "small", "mistake", "##s",       "during",    "work", "##ing", "hour",
  "😀",  "😃",  "😄",     "😁",     "+",     "/",     "-",       "=",         "12",        "28",   "40",    "16",
  " ",  "I",  "[CLS]", "[SEP]", "[UNK]", "[PAD]", "[MASK]",  "[unused1]", "[unused10]"};

TEST_F(MindDataTestPipeline, TestBertTokenizerSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBertTokenizerSuccess1.";
  // Test BertTokenizer with default parameters

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/bert_tokenizer.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create Take operation on ds
  ds = ds->Take(4);
  EXPECT_NE(ds, nullptr);

  // Create a vocab from vector
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  Status s = Vocab::BuildFromVector(list, {}, true, &vocab);
  EXPECT_EQ(s, Status::OK());

  // Create BertTokenizer operation on ds
  std::shared_ptr<TensorTransform> bert_tokenizer = std::make_shared<text::BertTokenizer>(vocab);
  EXPECT_NE(bert_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({bert_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::vector<std::string>> expected = {{"床", "前", "明", "月", "光"},
                                                    {"疑", "是", "地", "上", "霜"},
                                                    {"举", "头", "望", "明", "月"},
                                                    {"低", "头", "思", "故", "乡"}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestBertTokenizerSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBertTokenizerSuccess2.";
  // Test BertTokenizer with lower_case true

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/bert_tokenizer.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create Skip operation on ds
  ds = ds->Skip(4);
  EXPECT_NE(ds, nullptr);

  // Create Take operation on ds
  ds = ds->Take(1);
  EXPECT_NE(ds, nullptr);

  // Create a vocab from vector
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  Status s = Vocab::BuildFromVector(list, {}, true, &vocab);
  EXPECT_EQ(s, Status::OK());

  // Create BertTokenizer operation on ds
  std::shared_ptr<TensorTransform> bert_tokenizer =
    std::make_shared<text::BertTokenizer>(vocab, "##", 100, "[UNK]", true);
  EXPECT_NE(bert_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({bert_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::string> expected = {"i",   "am",     "mak",  "##ing", "small", "mistake",
                                       "##s", "during", "work", "##ing", "hour",  "##s"};
  std::shared_ptr<Tensor> de_expected_tensor;
  ASSERT_OK(Tensor::CreateFromVector(expected, &de_expected_tensor));
  mindspore::MSTensor expected_tensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestBertTokenizerSuccess3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBertTokenizerSuccess3.";
  // Test BertTokenizer with normalization_form NFKC

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/bert_tokenizer.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create Skip operation on ds
  ds = ds->Skip(5);
  EXPECT_NE(ds, nullptr);

  // Create Take operation on ds
  ds = ds->Take(2);
  EXPECT_NE(ds, nullptr);

  // Create a vocab from vector
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  Status s = Vocab::BuildFromVector(list, {}, true, &vocab);
  EXPECT_EQ(s, Status::OK());

  // Create BertTokenizer operation on ds
  std::shared_ptr<TensorTransform> bert_tokenizer =
    std::make_shared<text::BertTokenizer>(vocab, "##", 100, "[UNK]", false, false, NormalizeForm::kNfc);
  EXPECT_NE(bert_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({bert_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::vector<std::string>> expected = {
    {"😀", "嘿", "嘿", "😃", "哈", "哈", "😄", "大", "笑", "😁", "嘻", "嘻"}, {"繁", "體", "字"}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestBertTokenizerSuccess4) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBertTokenizerSuccess4.";
  // Test BertTokenizer with keep_whitespace true

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/bert_tokenizer.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create Skip operation on ds
  ds = ds->Skip(7);
  EXPECT_NE(ds, nullptr);

  // Create Take operation on ds
  ds = ds->Take(1);
  EXPECT_NE(ds, nullptr);

  // Create a vocab from vector
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  Status s = Vocab::BuildFromVector(list, {}, true, &vocab);
  EXPECT_EQ(s, Status::OK());

  // Create BertTokenizer operation on ds
  std::shared_ptr<TensorTransform> bert_tokenizer =
    std::make_shared<text::BertTokenizer>(vocab, "##", 100, "[UNK]", false, true);
  EXPECT_NE(bert_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({bert_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::string> expected = {"[UNK]", " ", "[CLS]"};
  std::shared_ptr<Tensor> de_expected_tensor;
  ASSERT_OK(Tensor::CreateFromVector(expected, &de_expected_tensor));
  mindspore::MSTensor expected_tensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestBertTokenizerSuccess5) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBertTokenizerSuccess5.";
  // Test BertTokenizer with unknown_token empty and keep_whitespace true

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/bert_tokenizer.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create Skip operation on ds
  ds = ds->Skip(7);
  EXPECT_NE(ds, nullptr);

  // Create Take operation on ds
  ds = ds->Take(1);
  EXPECT_NE(ds, nullptr);

  // Create a vocab from vector
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  Status s = Vocab::BuildFromVector(list, {}, true, &vocab);
  EXPECT_EQ(s, Status::OK());

  // Create BertTokenizer operation on ds
  std::shared_ptr<TensorTransform> bert_tokenizer =
    std::make_shared<text::BertTokenizer>(vocab, "##", 100, "", false, true);
  EXPECT_NE(bert_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({bert_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::string> expected = {"unused", " ", "[CLS]"};
  std::shared_ptr<Tensor> de_expected_tensor;
  ASSERT_OK(Tensor::CreateFromVector(expected, &de_expected_tensor));
  mindspore::MSTensor expected_tensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestBertTokenizerSuccess6) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBertTokenizerSuccess6.";
  // Test BertTokenizer with preserve_unused_token false, unknown_token empty and keep_whitespace true

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/bert_tokenizer.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create Skip operation on ds
  ds = ds->Skip(7);
  EXPECT_NE(ds, nullptr);

  // Create Take operation on ds
  ds = ds->Take(1);
  EXPECT_NE(ds, nullptr);

  // Create a vocab from vector
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  Status s = Vocab::BuildFromVector(list, {}, true, &vocab);
  EXPECT_EQ(s, Status::OK());

  // Create BertTokenizer operation on ds
  std::shared_ptr<TensorTransform> bert_tokenizer =
    std::make_shared<text::BertTokenizer>(vocab, "##", 100, "", false, true, NormalizeForm::kNone, false);
  EXPECT_NE(bert_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({bert_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::string> expected = {"unused", " ", "[", "CLS", "]"};
  std::shared_ptr<Tensor> de_expected_tensor;
  ASSERT_OK(Tensor::CreateFromVector(expected, &de_expected_tensor));
  mindspore::MSTensor expected_tensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestBertTokenizerSuccess7) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBertTokenizerSuccess7.";
  // Test BertTokenizer with with_offsets true and lower_case true

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/bert_tokenizer.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create Skip operation on ds
  ds = ds->Skip(4);
  EXPECT_NE(ds, nullptr);

  // Create Take operation on ds
  ds = ds->Take(1);
  EXPECT_NE(ds, nullptr);

  // Create a vocab from vector
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  Status s = Vocab::BuildFromVector(list, {}, true, &vocab);
  EXPECT_EQ(s, Status::OK());

  // Create BertTokenizer operation on ds
  std::shared_ptr<TensorTransform> bert_tokenizer =
    std::make_shared<text::BertTokenizer>(vocab, "##", 100, "[UNK]", true, false, NormalizeForm::kNone, true, true);
  EXPECT_NE(bert_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({bert_tokenizer}, {"text"}, {"token", "offsets_start", "offsets_limit"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::string> expected_tokens = {"i",   "am",     "mak",  "##ing", "small", "mistake",
                                              "##s", "during", "work", "##ing", "hour",  "##s"};
  std::vector<uint32_t> expected_offsets_start = {0, 2, 5, 8, 12, 18, 25, 27, 34, 38, 42, 46};
  std::vector<uint32_t> expected_offsets_limit = {1, 4, 8, 11, 17, 25, 26, 33, 38, 41, 46, 47};

  std::shared_ptr<Tensor> de_expected_tokens;
  ASSERT_OK(Tensor::CreateFromVector(expected_tokens, &de_expected_tokens));
  mindspore::MSTensor ms_expected_tokens =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tokens));

  std::shared_ptr<Tensor> de_expected_offsets_start;
  ASSERT_OK(Tensor::CreateFromVector(expected_offsets_start, &de_expected_offsets_start));
  mindspore::MSTensor ms_expected_offsets_start =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_offsets_start));

  std::shared_ptr<Tensor> de_expected_offsets_limit;
  ASSERT_OK(Tensor::CreateFromVector(expected_offsets_limit, &de_expected_offsets_limit));
  mindspore::MSTensor ms_expected_offsets_limit =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_offsets_limit));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["token"];
    EXPECT_MSTENSOR_EQ(ind, ms_expected_tokens);

    auto start = row["offsets_start"];
    EXPECT_MSTENSOR_EQ(start, ms_expected_offsets_start);

    auto limit = row["offsets_limit"];
    EXPECT_MSTENSOR_EQ(limit, ms_expected_offsets_limit);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestBertTokenizerFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBertTokenizerFail1.";
  // Test BertTokenizer with nullptr vocab

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/bert_tokenizer.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create BertTokenizer operation on ds
  std::shared_ptr<TensorTransform> bert_tokenizer = std::make_shared<text::BertTokenizer>(nullptr);
  EXPECT_NE(bert_tokenizer, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({bert_tokenizer});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid BertTokenizer input with nullptr vocab
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestBertTokenizerFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBertTokenizerFail2.";
  // Test BertTokenizer with negative max_bytes_per_token

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/bert_tokenizer.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create a vocab from vector
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  Status s = Vocab::BuildFromVector(list, {}, true, &vocab);
  EXPECT_EQ(s, Status::OK());

  // Create BertTokenizer operation on ds
  std::shared_ptr<TensorTransform> bert_tokenizer = std::make_shared<text::BertTokenizer>(vocab, "##", -1);
  EXPECT_NE(bert_tokenizer, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({bert_tokenizer});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid BertTokenizer input with nullptr vocab
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestCaseFoldSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCaseFoldSuccess.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create casefold operation on ds
  std::shared_ptr<TensorTransform> casefold = std::make_shared<text::CaseFold>();
  EXPECT_NE(casefold, nullptr);

  // Create Map operation on ds
  ds = ds->Map({casefold}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::string> expected = {"welcome to beijing!", "北京欢迎您!", "我喜欢english!", "  "};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateScalar(expected[i], &de_expected_tensor));
    mindspore::MSTensor ms_expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_tensor);
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestJiebaTokenizerSuccess) {
  // Testing the parameter of JiebaTokenizer interface when the mode is JiebaMode::kMp and the with_offsets is false.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestJiebaTokenizerSuccess.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testJiebaDataset/3.txt";
  std::string hmm_path = datasets_root_path_ + "/jiebadict/hmm_model.utf8";
  std::string mp_path = datasets_root_path_ + "/jiebadict/jieba.dict.utf8";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create jieba_tokenizer operation on ds
  std::shared_ptr<TensorTransform> jieba_tokenizer =
    std::make_shared<text::JiebaTokenizer>(hmm_path, mp_path, JiebaMode::kMp);
  EXPECT_NE(jieba_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({jieba_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::string> expected = {"今天天气", "太好了", "我们", "一起", "去", "外面", "玩吧"};
  std::shared_ptr<Tensor> de_expected_tensor;
  ASSERT_OK(Tensor::CreateFromVector(expected, &de_expected_tensor));
  mindspore::MSTensor expected_tensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestJiebaTokenizerSuccess1) {
  // Testing the parameter of JiebaTokenizer interface when the mode is JiebaMode::kHmm and the with_offsets is false.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestJiebaTokenizerSuccess1.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testJiebaDataset/3.txt";
  std::string hmm_path = datasets_root_path_ + "/jiebadict/hmm_model.utf8";
  std::string mp_path = datasets_root_path_ + "/jiebadict/jieba.dict.utf8";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create jieba_tokenizer operation on ds
  std::shared_ptr<TensorTransform> jieba_tokenizer =
    std::make_shared<text::JiebaTokenizer>(hmm_path, mp_path, JiebaMode::kHmm);
  EXPECT_NE(jieba_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({jieba_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::string> expected = {"今天", "天气", "太", "好", "了", "我们", "一起", "去", "外面", "玩", "吧"};
  std::shared_ptr<Tensor> de_expected_tensor;
  ASSERT_OK(Tensor::CreateFromVector(expected, &de_expected_tensor));
  mindspore::MSTensor expected_tensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestJiebaTokenizerSuccess2) {
  // Testing the parameter of JiebaTokenizer interface when the mode is JiebaMode::kMp and the with_offsets is true.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestJiebaTokenizerSuccess2.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testJiebaDataset/3.txt";
  std::string hmm_path = datasets_root_path_ + "/jiebadict/hmm_model.utf8";
  std::string mp_path = datasets_root_path_ + "/jiebadict/jieba.dict.utf8";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create jieba_tokenizer operation on ds
  std::shared_ptr<TensorTransform> jieba_tokenizer =
    std::make_shared<text::JiebaTokenizer>(hmm_path, mp_path, JiebaMode::kMp, true);
  EXPECT_NE(jieba_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({jieba_tokenizer}, {"text"}, {"token", "offsets_start", "offsets_limit"},
               {"token", "offsets_start", "offsets_limit"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::string> expected_tokens = {"今天天气", "太好了", "我们", "一起", "去", "外面", "玩吧"};
  std::vector<uint32_t> expected_offsets_start = {0, 12, 21, 27, 33, 36, 42};
  std::vector<uint32_t> expected_offsets_limit = {12, 21, 27, 33, 36, 42, 48};
  
  std::shared_ptr<Tensor> de_expected_tokens;
  ASSERT_OK(Tensor::CreateFromVector(expected_tokens, &de_expected_tokens));
  mindspore::MSTensor ms_expected_tokens =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tokens));

  std::shared_ptr<Tensor> de_expected_offsets_start;
  ASSERT_OK(Tensor::CreateFromVector(expected_offsets_start, &de_expected_offsets_start));
  mindspore::MSTensor ms_expected_offsets_start =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_offsets_start));

  std::shared_ptr<Tensor> de_expected_offsets_limit;
  ASSERT_OK(Tensor::CreateFromVector(expected_offsets_limit, &de_expected_offsets_limit));
  mindspore::MSTensor ms_expected_offsets_limit =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_offsets_limit));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["token"];
    EXPECT_MSTENSOR_EQ(ind, ms_expected_tokens);

    auto start = row["offsets_start"];
    EXPECT_MSTENSOR_EQ(start, ms_expected_offsets_start);

    auto limit = row["offsets_limit"];
    EXPECT_MSTENSOR_EQ(limit, ms_expected_offsets_limit);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestJiebaTokenizerFail1) {
  // Testing the incorrect parameter of JiebaTokenizer interface.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestJiebaTokenizerFail1.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testJiebaDataset/3.txt";
  std::string mp_path = datasets_root_path_ + "/jiebadict/jieba.dict.utf8";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create jieba_tokenizer operation on ds
  // Testing the parameter hmm_path is empty
  std::shared_ptr<TensorTransform> jieba_tokenizer =
    std::make_shared<text::JiebaTokenizer>("", mp_path, JiebaMode::kMp);
  EXPECT_NE(jieba_tokenizer, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({jieba_tokenizer});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid JiebaTokenizer input (parameter hmm_path is empty)
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestJiebaTokenizerFail2) {
  // Testing the incorrect parameter of JiebaTokenizer interface.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestJiebaTokenizerFail2.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testJiebaDataset/3.txt";
  std::string hmm_path = datasets_root_path_ + "/jiebadict/hmm_model.utf8";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create jieba_tokenizer operation on ds
  // Testing the parameter mp_path is empty
  std::shared_ptr<TensorTransform> jieba_tokenizer =
    std::make_shared<text::JiebaTokenizer>(hmm_path, "", JiebaMode::kMp);
  EXPECT_NE(jieba_tokenizer, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({jieba_tokenizer});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid JiebaTokenizer input (parameter mp_path is empty)
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestJiebaTokenizerFail3) {
  // Testing the incorrect parameter of JiebaTokenizer interface.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestJiebaTokenizerFail3.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testJiebaDataset/3.txt";
  std::string hmm_path_invalid = datasets_root_path_ + "/jiebadict/1.txt";
  std::string mp_path = datasets_root_path_ + "/jiebadict/jieba.dict.utf8";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create jieba_tokenizer operation on ds
  // Testing the parameter hmm_path is invalid path
  std::shared_ptr<TensorTransform> jieba_tokenizer =
    std::make_shared<text::JiebaTokenizer>(hmm_path_invalid, mp_path, JiebaMode::kMp);
  EXPECT_NE(jieba_tokenizer, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({jieba_tokenizer});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid JiebaTokenizer input (parameter hmm_path is invalid path)
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestJiebaTokenizerFail4) {
  // Testing the incorrect parameter of JiebaTokenizer interface.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestJiebaTokenizerFail4.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testJiebaDataset/3.txt";
  std::string hmm_path = datasets_root_path_ + "/jiebadict/hmm_model.utf8";
  std::string mp_path_invalid = datasets_root_path_ + "/jiebadict/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create jieba_tokenizer operation on ds
  // Testing the parameter mp_path is invalid path
  std::shared_ptr<TensorTransform> jieba_tokenizer =
    std::make_shared<text::JiebaTokenizer>(hmm_path, mp_path_invalid, JiebaMode::kMp);
  EXPECT_NE(jieba_tokenizer, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({jieba_tokenizer});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid JiebaTokenizer input (parameter mp_path is invalid path)
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestJiebaTokenizerAddWord) {
  // Testing the parameter AddWord of JiebaTokenizer when the freq is not provided (default 0).
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestJiebaTokenizerAddWord.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testJiebaDataset/4.txt";
  std::string hmm_path = datasets_root_path_ + "/jiebadict/hmm_model.utf8";
  std::string mp_path = datasets_root_path_ + "/jiebadict/jieba.dict.utf8";
  std::shared_ptr<Dataset> ds = TextFile({data_file});
  EXPECT_NE(ds, nullptr);

  // Create jieba_tokenizer operation on ds
  std::shared_ptr<text::JiebaTokenizer> jieba_tokenizer =
    std::make_shared<text::JiebaTokenizer>(hmm_path, mp_path, JiebaMode::kMp);
  EXPECT_NE(jieba_tokenizer, nullptr);

  // Add word with freq not provided (default 0)
  jieba_tokenizer->AddWord("男默女泪");

  // Create Map operation on ds
  ds = ds->Map({jieba_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::string> expected = {"男默女泪", "市", "长江大桥"};
  std::shared_ptr<Tensor> de_expected_tensor;
  ASSERT_OK(Tensor::CreateFromVector(expected, &de_expected_tensor));
  mindspore::MSTensor expected_tensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestJiebaTokenizerAddWord1) {
  // Testing the parameter AddWord of JiebaTokenizer when the freq is set explicitly to 0.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestJiebaTokenizerAddWord1.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testJiebaDataset/4.txt";
  std::string hmm_path = datasets_root_path_ + "/jiebadict/hmm_model.utf8";
  std::string mp_path = datasets_root_path_ + "/jiebadict/jieba.dict.utf8";
  std::shared_ptr<Dataset> ds = TextFile({data_file});
  EXPECT_NE(ds, nullptr);

  // Create jieba_tokenizer operation on ds
  std::shared_ptr<text::JiebaTokenizer> jieba_tokenizer =
    std::make_shared<text::JiebaTokenizer>(hmm_path, mp_path, JiebaMode::kMp);
  EXPECT_NE(jieba_tokenizer, nullptr);

  // Add word with freq is set explicitly to 0
  jieba_tokenizer->AddWord("男默女泪", 0);

  // Create Map operation on ds
  ds = ds->Map({jieba_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::string> expected = {"男默女泪", "市", "长江大桥"};
  std::shared_ptr<Tensor> de_expected_tensor;
  ASSERT_OK(Tensor::CreateFromVector(expected, &de_expected_tensor));
  mindspore::MSTensor expected_tensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestJiebaTokenizerAddWord2) {
  // Testing the parameter AddWord of JiebaTokenizer when the freq is 10.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestJiebaTokenizerAddWord2.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testJiebaDataset/4.txt";
  std::string hmm_path = datasets_root_path_ + "/jiebadict/hmm_model.utf8";
  std::string mp_path = datasets_root_path_ + "/jiebadict/jieba.dict.utf8";
  std::shared_ptr<Dataset> ds = TextFile({data_file});
  EXPECT_NE(ds, nullptr);

  // Create jieba_tokenizer operation on ds
  std::shared_ptr<text::JiebaTokenizer> jieba_tokenizer =
    std::make_shared<text::JiebaTokenizer>(hmm_path, mp_path, JiebaMode::kMp);
  EXPECT_NE(jieba_tokenizer, nullptr);

  // Add word with freq 10
  jieba_tokenizer->AddWord("男默女泪", 10);

  // Create Map operation on ds
  ds = ds->Map({jieba_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::string> expected = {"男默女泪", "市", "长江大桥"};
  std::shared_ptr<Tensor> de_expected_tensor;
  ASSERT_OK(Tensor::CreateFromVector(expected, &de_expected_tensor));
  mindspore::MSTensor expected_tensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestJiebaTokenizerAddWord3) {
  // Testing the parameter AddWord of JiebaTokenizer when the freq is 20000 which affects the result of segmentation.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestJiebaTokenizerAddWord3.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testJiebaDataset/6.txt";
  std::string hmm_path = datasets_root_path_ + "/jiebadict/hmm_model.utf8";
  std::string mp_path = datasets_root_path_ + "/jiebadict/jieba.dict.utf8";
  std::shared_ptr<Dataset> ds = TextFile({data_file});
  EXPECT_NE(ds, nullptr);

  // Create jieba_tokenizer operation on ds
  std::shared_ptr<text::JiebaTokenizer> jieba_tokenizer =
    std::make_shared<text::JiebaTokenizer>(hmm_path, mp_path, JiebaMode::kMp);
  EXPECT_NE(jieba_tokenizer, nullptr);

  // Add word with freq 20000
  jieba_tokenizer->AddWord("江大桥", 20000);

  // Create Map operation on ds
  ds = ds->Map({jieba_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::string> expected = {"江州", "市长", "江大桥", "参加", "了", "长江大桥", "的", "通车", "仪式"};
  std::shared_ptr<Tensor> de_expected_tensor;
  ASSERT_OK(Tensor::CreateFromVector(expected, &de_expected_tensor));
  mindspore::MSTensor expected_tensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestJiebaTokenizerAddWordFail) {
  // Testing the incorrect parameter of AddWord in JiebaTokenizer.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestJiebaTokenizerAddWordFail.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testJiebaDataset/3.txt";
  std::string hmm_path = datasets_root_path_ + "/jiebadict/hmm_model.utf8";
  std::string mp_path = datasets_root_path_ + "/jiebadict/jieba.dict.utf8";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Testing the parameter word of AddWord is empty
  std::shared_ptr<text::JiebaTokenizer> jieba_tokenizer =
    std::make_shared<text::JiebaTokenizer>(hmm_path, mp_path, JiebaMode::kMp);
  EXPECT_NE(jieba_tokenizer, nullptr);
  EXPECT_NE(jieba_tokenizer->AddWord("", 10), Status::OK());
  // Testing the parameter freq of AddWord is negative
  std::shared_ptr<text::JiebaTokenizer> jieba_tokenizer1 =
    std::make_shared<text::JiebaTokenizer>(hmm_path, mp_path, JiebaMode::kMp);
  EXPECT_NE(jieba_tokenizer1, nullptr);
  EXPECT_NE(jieba_tokenizer1->AddWord("我们", -1), Status::OK());
}

TEST_F(MindDataTestPipeline, TestSlidingWindowSuccess) {
  // Testing the parameter of SlidingWindow interface when the axis is 0.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSlidingWindowSuccess.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create white_tokenizer operation on ds
  std::shared_ptr<TensorTransform> white_tokenizer = std::make_shared<text::WhitespaceTokenizer>();
  EXPECT_NE(white_tokenizer, nullptr);
  // Create sliding_window operation on ds
  std::shared_ptr<TensorTransform> sliding_window = std::make_shared<text::SlidingWindow>(3, 0);
  EXPECT_NE(sliding_window, nullptr);

  // Create Map operation on ds
  ds = ds->Map({white_tokenizer, sliding_window}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::vector<std::string>> expected = {{"This", "is", "a", "is", "a", "text", "a", "text", "file."},
                                                    {"Be", "happy", "every", "happy", "every", "day."},
                                                    {"Good", "luck", "to", "luck", "to", "everyone."}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];

    std::shared_ptr<Tensor> de_expected_tensor;
    int x = expected[i].size() / 3;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], TensorShape({x, 3}), &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestSlidingWindowSuccess1) {
  // Testing the parameter of SlidingWindow interface when the axis is -1.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSlidingWindowSuccess1.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create white_tokenizer operation on ds
  std::shared_ptr<TensorTransform> white_tokenizer = std::make_shared<text::WhitespaceTokenizer>();
  EXPECT_NE(white_tokenizer, nullptr);
  // Create sliding_window operation on ds
  std::shared_ptr<TensorTransform> sliding_window = std::make_shared<text::SlidingWindow>(2, -1);
  EXPECT_NE(sliding_window, nullptr);

  // Create Map operation on ds
  ds = ds->Map({white_tokenizer, sliding_window}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::vector<std::string>> expected = {{"This", "is", "is", "a", "a", "text", "text", "file."},
                                                    {"Be", "happy", "happy", "every", "every", "day."},
                                                    {"Good", "luck", "luck", "to", "to", "everyone."}};
  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];

    std::shared_ptr<Tensor> de_expected_tensor;
    int x = expected[i].size() / 2;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], TensorShape({x, 2}), &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestSlidingWindowFail1) {
  // Testing the incorrect parameter of SlidingWindow interface.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSlidingWindowFail1.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create sliding_window operation on ds
  // Testing the parameter width less than or equal to 0
  // The parameter axis support 0 or -1 only for now
  std::shared_ptr<TensorTransform> sliding_window = std::make_shared<text::SlidingWindow>(0, 0);
  EXPECT_NE(sliding_window, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({sliding_window});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid SlidingWindow input (width less than or equal to 0)
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestSlidingWindowFail2) {
  // Testing the incorrect parameter of SlidingWindow interface.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSlidingWindowFail2.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create sliding_window operation on ds
  // Testing the parameter width less than or equal to 0
  // The parameter axis support 0 or -1 only for now
  std::shared_ptr<TensorTransform> sliding_window = std::make_shared<text::SlidingWindow>(-2, 0);
  EXPECT_NE(sliding_window, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({sliding_window});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid SlidingWindow input (width less than or equal to 0)
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestToNumberSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestToNumberSuccess1.";
  // Test ToNumber with integer numbers

  std::string data_file = datasets_root_path_ + "/testTokenizerData/to_number.txt";

  // Create a TextFile dataset
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create a Take operation on ds
  ds = ds->Take(8);
  EXPECT_NE(ds, nullptr);

  // Create ToNumber operation on ds
  std::shared_ptr<TensorTransform> to_number = std::make_shared<text::ToNumber>("int64");
  EXPECT_NE(to_number, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({to_number}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<int64_t> expected = {-121, 14, -2219, 7623, -8162536, 162371864, -1726483716, 98921728421};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateScalar(expected[i], &de_expected_tensor));
    mindspore::MSTensor ms_expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_tensor);
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 8);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestToNumberSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestToNumberSuccess2.";
  // Test ToNumber with float numbers

  std::string data_file = datasets_root_path_ + "/testTokenizerData/to_number.txt";

  // Create a TextFile dataset
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create a Skip operation on ds
  ds = ds->Skip(8);
  EXPECT_NE(ds, nullptr);

  // Create a Take operation on ds
  ds = ds->Take(6);
  EXPECT_NE(ds, nullptr);

  // Create ToNumber operation on ds
  std::shared_ptr<TensorTransform> to_number = std::make_shared<text::ToNumber>("float64");
  EXPECT_NE(to_number, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({to_number}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<double_t> expected = {-1.1, 1.4, -2219.321, 7623.453, -816256.234282, 162371864.243243};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateScalar(expected[i], &de_expected_tensor));
    mindspore::MSTensor ms_expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_tensor);
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestToNumberFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestToNumberFail1.";
  // Test ToNumber with overflow integer numbers

  std::string data_file = datasets_root_path_ + "/testTokenizerData/to_number.txt";

  // Create a TextFile dataset
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create a Skip operation on ds
  ds = ds->Skip(2);
  EXPECT_NE(ds, nullptr);

  // Create a Take operation on ds
  ds = ds->Take(6);
  EXPECT_NE(ds, nullptr);

  // Create ToNumber operation on ds
  std::shared_ptr<TensorTransform> to_number = std::make_shared<text::ToNumber>("int8");
  EXPECT_NE(to_number, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({to_number}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;

  // Expect error: input out of bounds of int8
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    iter->GetNextRow(&row);
    i++;
  }

  // Expect failure: GetNextRow fail and return nothing
  EXPECT_EQ(i, 0);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestToNumberFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestToNumberFail2.";
  // Test ToNumber with overflow float numbers

  std::string data_file = datasets_root_path_ + "/testTokenizerData/to_number.txt";

  // Create a TextFile dataset
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create a Skip operation on ds
  ds = ds->Skip(12);
  EXPECT_NE(ds, nullptr);

  // Create a Take operation on ds
  ds = ds->Take(2);
  EXPECT_NE(ds, nullptr);

  // Create ToNumber operation on ds
  std::shared_ptr<TensorTransform> to_number = std::make_shared<text::ToNumber>("float16");
  EXPECT_NE(to_number, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({to_number}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;

  // Expect error: input out of bounds of float16
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    iter->GetNextRow(&row);
    i++;
  }

  // Expect failure: GetNextRow fail and return nothing
  EXPECT_EQ(i, 0);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestToNumberFail3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestToNumberFail3.";
  // Test ToNumber with non numerical input

  std::string data_file = datasets_root_path_ + "/testTokenizerData/to_number.txt";

  // Create a TextFile dataset
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create a Skip operation on ds
  ds = ds->Skip(14);
  EXPECT_NE(ds, nullptr);

  // Create ToNumber operation on ds
  std::shared_ptr<TensorTransform> to_number = std::make_shared<text::ToNumber>("int64");
  EXPECT_NE(to_number, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({to_number}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;

  // Expect error: invalid input which is non numerical
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    iter->GetNextRow(&row);
    i++;
  }

  // Expect failure: GetNextRow fail and return nothing
  EXPECT_EQ(i, 0);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestToNumberFail4) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestToNumberFail4.";
  // Test ToNumber with non numerical data type

  std::string data_file = datasets_root_path_ + "/testTokenizerData/to_number.txt";

  // Create a TextFile dataset
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create ToNumber operation on ds
  std::shared_ptr<TensorTransform> to_number = std::make_shared<text::ToNumber>("string");
  EXPECT_NE(to_number, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({to_number}, {"text"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid parameter with non numerical data type
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestToNumberFail5) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestToNumberFail5.";
  // Test ToNumber with non numerical data type

  std::string data_file = datasets_root_path_ + "/testTokenizerData/to_number.txt";

  // Create a TextFile dataset
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create ToNumber operation on ds
  std::shared_ptr<TensorTransform> to_number = std::make_shared<text::ToNumber>("bool");
  EXPECT_NE(to_number, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({to_number}, {"text"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid parameter with non numerical data type
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestTruncateSequencePairSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTruncateSequencePairSuccess1.";
  // Testing basic TruncateSequencePair

  // Set seed for RandomDataset
  auto original_seed = config::get_seed();
  bool status_set_seed = config::set_seed(0);
  EXPECT_EQ(status_set_seed, true);

  // Set num_parallel_workers for RandomDataset
  auto original_worker = config::get_num_parallel_workers();
  bool status_set_worker = config::set_num_parallel_workers(1);
  EXPECT_EQ(status_set_worker, true);

  // Create a RandomDataset which has column names "col1" and "col2"
  std::shared_ptr<SchemaObj> schema = Schema();
  schema->add_column("col1", mindspore::DataType::kNumberTypeInt16, {5});
  schema->add_column("col2", mindspore::DataType::kNumberTypeInt32, {3});
  std::shared_ptr<Dataset> ds = RandomData(3, schema);
  EXPECT_NE(ds, nullptr);

  // Create a truncate_sequence_pair operation on ds
  std::shared_ptr<TensorTransform> truncate_sequence_pair = std::make_shared<text::TruncateSequencePair>(4);
  EXPECT_NE(truncate_sequence_pair, nullptr);

  // Create Map operation on ds
  ds = ds->Map({truncate_sequence_pair}, {"col1", "col2"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::vector<int16_t>> expected1 = {{-29556, -29556}, {-18505, -18505}, {-25958, -25958}};
  std::vector<std::vector<int32_t>> expected2 = {
    {-1751672937, -1751672937}, {-656877352, -656877352}, {-606348325, -606348325}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind1 = row["col1"];
    auto ind2 = row["col2"];

    std::shared_ptr<Tensor> de_expected_tensor1;
    ASSERT_OK(Tensor::CreateFromVector(expected1[i], &de_expected_tensor1));
    mindspore::MSTensor expected_tensor1 =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor1));
    EXPECT_MSTENSOR_EQ(ind1, expected_tensor1);

    std::shared_ptr<Tensor> de_expected_tensor2;
    ASSERT_OK(Tensor::CreateFromVector(expected2[i], &de_expected_tensor2));
    mindspore::MSTensor expected_tensor2 =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor2));
    EXPECT_MSTENSOR_EQ(ind2, expected_tensor2);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();

  // Restore original seed and num_parallel_workers
  status_set_seed = config::set_seed(original_seed);
  EXPECT_EQ(status_set_seed, true);
  status_set_worker = config::set_num_parallel_workers(original_worker);
  EXPECT_EQ(status_set_worker, true);
}

TEST_F(MindDataTestPipeline, TestTruncateSequencePairSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTruncateSequencePairSuccess2.";
  // Testing basic TruncateSequencePair with odd max_length

  // Set seed for RandomDataset
  auto original_seed = config::get_seed();
  bool status_set_seed = config::set_seed(1);
  EXPECT_EQ(status_set_seed, true);

  // Set num_parallel_workers for RandomDataset
  auto original_worker = config::get_num_parallel_workers();
  bool status_set_worker = config::set_num_parallel_workers(1);
  EXPECT_EQ(status_set_worker, true);

  // Create a RandomDataset which has column names "col1" and "col2"
  std::shared_ptr<SchemaObj> schema = Schema();
  schema->add_column("col1", mindspore::DataType::kNumberTypeInt32, {4});
  schema->add_column("col2", mindspore::DataType::kNumberTypeInt64, {4});
  std::shared_ptr<Dataset> ds = RandomData(4, schema);
  EXPECT_NE(ds, nullptr);

  // Create a truncate_sequence_pair operation on ds
  std::shared_ptr<TensorTransform> truncate_sequence_pair = std::make_shared<text::TruncateSequencePair>(5);
  EXPECT_NE(truncate_sequence_pair, nullptr);

  // Create Map operation on ds
  ds = ds->Map({truncate_sequence_pair}, {"col1", "col2"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::vector<int32_t>> expected1 = {{1785358954, 1785358954, 1785358954},
                                                 {-1195853640, -1195853640, -1195853640},
                                                 {0, 0, 0},
                                                 {1296911693, 1296911693, 1296911693}};
  std::vector<std::vector<int64_t>> expected2 = {
    {-1, -1}, {-1229782938247303442, -1229782938247303442}, {2314885530818453536, 2314885530818453536}, {-1, -1}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind1 = row["col1"];
    auto ind2 = row["col2"];

    std::shared_ptr<Tensor> de_expected_tensor1;
    ASSERT_OK(Tensor::CreateFromVector(expected1[i], &de_expected_tensor1));
    mindspore::MSTensor expected_tensor1 =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor1));
    EXPECT_MSTENSOR_EQ(ind1, expected_tensor1);

    std::shared_ptr<Tensor> de_expected_tensor2;
    ASSERT_OK(Tensor::CreateFromVector(expected2[i], &de_expected_tensor2));
    mindspore::MSTensor expected_tensor2 =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor2));
    EXPECT_MSTENSOR_EQ(ind2, expected_tensor2);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();

  // Restore original seed and num_parallel_workers
  status_set_seed = config::set_seed(original_seed);
  EXPECT_EQ(status_set_seed, true);
  status_set_worker = config::set_num_parallel_workers(original_worker);
  EXPECT_EQ(status_set_worker, true);
}

TEST_F(MindDataTestPipeline, TestTruncateSequencePairFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTruncateSequencePairFail.";
  // Testing TruncateSequencePair with negative max_length

  // Create a RandomDataset which has column names "col1" and "col2"
  std::shared_ptr<SchemaObj> schema = Schema();
  schema->add_column("col1", mindspore::DataType::kNumberTypeInt8, {3});
  schema->add_column("col2", mindspore::DataType::kNumberTypeInt8, {3});
  std::shared_ptr<Dataset> ds = RandomData(3, schema);
  EXPECT_NE(ds, nullptr);

  // Create a truncate_sequence_pair operation on ds
  std::shared_ptr<TensorTransform> truncate_sequence_pair = std::make_shared<text::TruncateSequencePair>(-1);
  EXPECT_NE(truncate_sequence_pair, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({truncate_sequence_pair});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid TruncateSequencePair input (invalid parameter with negative max_length)
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestNgramSuccess) {
  // Testing the parameter of Ngram interface.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNgramSuccess.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create white_tokenizer operation on ds
  std::shared_ptr<TensorTransform> white_tokenizer = std::make_shared<text::WhitespaceTokenizer>();
  EXPECT_NE(white_tokenizer, nullptr);
  // Create sliding_window operation on ds
  std::shared_ptr<TensorTransform> ngram_op(new text::Ngram({2}, {"_", 1}, {"_", 1}, " "));
  EXPECT_NE(ngram_op, nullptr);

  // Create Map operation on ds
  ds = ds->Map({white_tokenizer, ngram_op}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::vector<std::string>> expected = {{"_ This", "This is", "is a", "a text", "text file.", "file. _"},
                                                    {"_ Be", "Be happy", "happy every", "every day.", "day. _"},
                                                    {"_ Good", "Good luck", "luck to", "to everyone.", "everyone. _"}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];

    std::shared_ptr<Tensor> de_expected_tensor;
    int x = expected[i].size();
    ASSERT_OK(Tensor::CreateFromVector(expected[i], TensorShape({x}), &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestNgramSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNgramSuccess1.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create white_tokenizer operation on ds
  std::shared_ptr<TensorTransform> white_tokenizer = std::make_shared<text::WhitespaceTokenizer>();
  EXPECT_NE(white_tokenizer, nullptr);
  // Create sliding_window operation on ds
  std::shared_ptr<TensorTransform> ngram_op(new text::Ngram({2, 3}, {"&", 2}, {"&", 2}, "-"));
  EXPECT_NE(ngram_op, nullptr);

  // Create Map operation on ds
  ds = ds->Map({white_tokenizer, ngram_op}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::vector<std::string>> expected = {
    {"&-This", "This-is", "is-a", "a-text", "text-file.", "file.-&", "&-&-This", "&-This-is", "This-is-a",
     "is-a-text",
     "a-text-file.", "text-file.-&", "file.-&-&"},
    {"&-Be", "Be-happy", "happy-every", "every-day.", "day.-&", "&-&-Be", "&-Be-happy", "Be-happy-every",
     "happy-every-day.", "every-day.-&", "day.-&-&"},
    {"&-Good", "Good-luck", "luck-to", "to-everyone.", "everyone.-&", "&-&-Good", "&-Good-luck", "Good-luck-to",
     "luck-to-everyone.", "to-everyone.-&", "everyone.-&-&"}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];

    std::shared_ptr<Tensor> de_expected_tensor;
    int x = expected[i].size();
    ASSERT_OK(Tensor::CreateFromVector(expected[i], TensorShape({x}), &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestNgramFail1) {
  // Testing the incorrect parameter of Ngram interface.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNgramFail1.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create sliding_window operation on ds
  // Testing the vector of ngram is empty
  std::shared_ptr<TensorTransform> ngram_op(new text::Ngram({}));
  EXPECT_NE(ngram_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({ngram_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Ngram input (the vector of ngram is empty)
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestNgramFail2) {
  // Testing the incorrect parameter of Ngram interface.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNgramFail2.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create sliding_window operation on ds
  // Testing the value of ngrams vector less than and equal to 0
  std::shared_ptr<TensorTransform> ngram_op(new text::Ngram({0}));
  EXPECT_NE(ngram_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({ngram_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Ngram input (the value of ngrams vector less than and equal to 0)
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestNgramFail3) {
  // Testing the incorrect parameter of Ngram interface.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNgramFail3.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create sliding_window operation on ds
  // Testing the value of ngrams vector less than and equal to 0
  std::shared_ptr<TensorTransform> ngram_op(new text::Ngram({-2}));
  EXPECT_NE(ngram_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({ngram_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Ngram input (the value of ngrams vector less than and equal to 0)
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestNgramFail4) {
  // Testing the incorrect parameter of Ngram interface.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNgramFail4.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create sliding_window operation on ds
  // Testing the second parameter pad_width in left_pad vector less than 0
  std::shared_ptr<TensorTransform> ngram_op(new text::Ngram({2}, {"", -1}));
  EXPECT_NE(ngram_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({ngram_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Ngram input (the second parameter pad_width in left_pad vector less than 0)
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestNgramFail5) {
  // Testing the incorrect parameter of Ngram interface.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNgramFail5.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create sliding_window operation on ds
  // Testing the second parameter pad_width in right_pad vector less than 0
  std::shared_ptr<TensorTransform> ngram_op(new text::Ngram({2}, {"", 1}, {"", -1}));
  EXPECT_NE(ngram_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({ngram_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Ngram input (the second parameter pad_width in left_pad vector less than 0)
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestNormalizeUTF8Success) {
  // Testing the parameter of NormalizeUTF8 interface when the normalize_form is NormalizeForm::kNfkc.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNormalizeUTF8Success.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/normalize.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create normalizeutf8 operation on ds
  std::shared_ptr<TensorTransform> normalizeutf8 = std::make_shared<text::NormalizeUTF8>(NormalizeForm::kNfkc);
  EXPECT_NE(normalizeutf8, nullptr);

  // Create Map operation on ds
  ds = ds->Map({normalizeutf8}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::string> expected = {"ṩ", "ḍ̇", "q̣̇", "fi", "25", "ṩ"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateScalar(expected[i], &de_expected_tensor));
    mindspore::MSTensor ms_expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_tensor);
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestNormalizeUTF8Success1) {
  // Testing the parameter of NormalizeUTF8 interface when the normalize_form is NormalizeForm::kNfc.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNormalizeUTF8Success1.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/normalize.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create normalizeutf8 operation on ds
  std::shared_ptr<TensorTransform> normalizeutf8 = std::make_shared<text::NormalizeUTF8>(NormalizeForm::kNfc);
  EXPECT_NE(normalizeutf8, nullptr);

  // Create Map operation on ds
  ds = ds->Map({normalizeutf8}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::string> expected = {"ṩ", "ḍ̇", "q̣̇", "ﬁ", "2⁵", "ẛ̣"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateScalar(expected[i], &de_expected_tensor));
    mindspore::MSTensor ms_expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_tensor);
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestNormalizeUTF8Success2) {
  // Testing the parameter of NormalizeUTF8 interface when the normalize_form is NormalizeForm::kNfd.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNormalizeUTF8Success2.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/normalize.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create normalizeutf8 operation on ds
  std::shared_ptr<TensorTransform> normalizeutf8 = std::make_shared<text::NormalizeUTF8>(NormalizeForm::kNfd);
  EXPECT_NE(normalizeutf8, nullptr);

  // Create Map operation on ds
  ds = ds->Map({normalizeutf8}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::string> expected = {"ṩ", "ḍ̇", "q̣̇", "ﬁ", "2⁵", "ẛ̣"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateScalar(expected[i], &de_expected_tensor));
    mindspore::MSTensor ms_expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_tensor);
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestNormalizeUTF8Success3) {
  // Testing the parameter of NormalizeUTF8 interface when the normalize_form is NormalizeForm::kNfkd.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNormalizeUTF8Success3.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/normalize.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create normalizeutf8 operation on ds
  std::shared_ptr<TensorTransform> normalizeutf8 = std::make_shared<text::NormalizeUTF8>(NormalizeForm::kNfkd);
  EXPECT_NE(normalizeutf8, nullptr);

  // Create Map operation on ds
  ds = ds->Map({normalizeutf8}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::string> expected = {"ṩ", "ḍ̇", "q̣̇", "fi", "25", "ṩ"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateScalar(expected[i], &de_expected_tensor));
    mindspore::MSTensor ms_expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_tensor);
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRegexReplaceSuccess) {
  // Testing the parameter of RegexReplace interface when the replace_all is true.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRegexReplaceSuccess.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/regex_replace.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create regex_replace operation on ds
  std::shared_ptr<TensorTransform> regex_replace = std::make_shared<text::RegexReplace>("\\s+", "_", true);
  EXPECT_NE(regex_replace, nullptr);

  // Create Map operation on ds
  ds = ds->Map({regex_replace}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::string> expected = {"Hello_World", "Let's_Go",          "1:hello",        "2:world",
                                       "31:beijing",  "Welcome_to_China!", "_我_不想_长大_", "Welcome_to_Shenzhen!"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateScalar(expected[i], &de_expected_tensor));
    mindspore::MSTensor ms_expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_tensor);
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 8);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRegexReplaceSuccess1) {
  // Testing the parameter of RegexReplace interface when the replace_all is false.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRegexReplaceSuccess1.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/regex_replace.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create regex_replace operation on ds
  std::shared_ptr<TensorTransform> regex_replace = std::make_shared<text::RegexReplace>("\\s+", "_", false);
  EXPECT_NE(regex_replace, nullptr);

  // Create Map operation on ds
  ds = ds->Map({regex_replace}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::string> expected = {"Hello_World", "Let's_Go",          "1:hello",          "2:world",
                                       "31:beijing",  "Welcome_to China!", "_我	不想  长大	", "Welcome_to Shenzhen!"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateScalar(expected[i], &de_expected_tensor));
    mindspore::MSTensor ms_expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_tensor);
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 8);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRegexTokenizerSuccess) {
  // Testing the parameter of RegexTokenizer interface when the with_offsets is false.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRegexTokenizerSuccess.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/regex_replace.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create regex_tokenizer operation on ds
  std::shared_ptr<TensorTransform> regex_tokenizer = std::make_shared<text::RegexTokenizer>("\\s+", "\\s+", false);
  EXPECT_NE(regex_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({regex_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::vector<std::string>> expected = {{"Hello", " ", "World"},
                                                    {"Let's", " ", "Go"},
                                                    {"1:hello"},
                                                    {"2:world"},
                                                    {"31:beijing"},
                                                    {"Welcome", " ", "to", " ", "China!"},
                                                    {"  ", "我", "	", "不想", "  ", "长大", "	"},
                                                    {"Welcome", " ", "to", " ", "Shenzhen!"}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];

    std::shared_ptr<Tensor> de_expected_tensor;
    int x = expected[i].size();
    ASSERT_OK(Tensor::CreateFromVector(expected[i], TensorShape({x}), &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 8);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRegexTokenizerSuccess1) {
  // Testing the parameter of RegexTokenizer interface when the with_offsets is true.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRegexTokenizerSuccess1.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/regex_replace.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create regex_tokenizer operation on ds
  std::shared_ptr<TensorTransform> regex_tokenizer = std::make_shared<text::RegexTokenizer>("\\s+", "\\s+", true);
  EXPECT_NE(regex_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({regex_tokenizer}, {"text"}, {"token", "offsets_start", "offsets_limit"},
               {"token", "offsets_start", "offsets_limit"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::vector<std::string>> expected_tokens = {{"Hello", " ", "World"},
                                                           {"Let's", " ", "Go"},
                                                           {"1:hello"},
                                                           {"2:world"},
                                                           {"31:beijing"},
                                                           {"Welcome", " ", "to", " ", "China!"},
                                                           {"  ", "我", "	", "不想", "  ", "长大", "	"},
                                                           {"Welcome", " ", "to", " ", "Shenzhen!"}};

  std::vector<std::vector<uint32_t>> expected_offsets_start = {
    {0, 5, 6}, {0, 5, 6}, {0}, {0}, {0}, {0, 7, 8, 10, 11}, {0, 2, 5, 6, 12, 14, 20}, {0, 7, 8, 10, 11}};
  std::vector<std::vector<uint32_t>> expected_offsets_limit = {
    {5, 6, 11}, {5, 6, 8}, {7}, {7}, {10}, {7, 8, 10, 11, 17}, {2, 5, 6, 12, 14, 20, 21}, {7, 8, 10, 11, 20}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto token = row["token"];
    auto start = row["offsets_start"];
    auto limit = row["offsets_limit"];

    std::shared_ptr<Tensor> de_expected_tokens;
    int x = expected_tokens[i].size();
    ASSERT_OK(Tensor::CreateFromVector(expected_tokens[i], TensorShape({x}), &de_expected_tokens));
    mindspore::MSTensor ms_expected_tokens =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tokens));
    EXPECT_MSTENSOR_EQ(token, ms_expected_tokens);

    std::shared_ptr<Tensor> de_expected_offsets_start;
    ASSERT_OK(Tensor::CreateFromVector(expected_offsets_start[i], TensorShape({x}), &de_expected_offsets_start));
    mindspore::MSTensor ms_expected_offsets_start =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_offsets_start));
    EXPECT_MSTENSOR_EQ(start, ms_expected_offsets_start);

    std::shared_ptr<Tensor> de_expected_offsets_limit;
    ASSERT_OK(Tensor::CreateFromVector(expected_offsets_limit[i], TensorShape({x}), &de_expected_offsets_limit));
    mindspore::MSTensor ms_expected_offsets_limit =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_offsets_limit));
    EXPECT_MSTENSOR_EQ(limit, ms_expected_offsets_limit);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 8);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestUnicodeCharTokenizerSuccess) {
  // Testing the parameter of UnicodeCharTokenizer interface when the with_offsets is default.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUnicodeCharTokenizerSuccess.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create unicodechar_tokenizer operation on ds
  std::shared_ptr<TensorTransform> unicodechar_tokenizer = std::make_shared<text::UnicodeCharTokenizer>();
  EXPECT_NE(unicodechar_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({unicodechar_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::vector<std::string>> expected = {
    {"W", "e", "l", "c", "o", "m", "e", " ", "t", "o", " ", "B", "e", "i", "j", "i", "n", "g", "!"},
    {"北", "京", "欢", "迎", "您", "！"},
    {"我", "喜", "欢", "E", "n", "g", "l", "i", "s", "h", "!"},
    {" ", " "}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];

    std::shared_ptr<Tensor> de_expected_tensor;
    int x = expected[i].size();
    ASSERT_OK(Tensor::CreateFromVector(expected[i], TensorShape({x}), &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestUnicodeCharTokenizerSuccess1) {
  // Testing the parameter of UnicodeCharTokenizer interface when the with_offsets is true.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUnicodeCharTokenizerSuccess1.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create unicodechar_tokenizer operation on ds
  std::shared_ptr<TensorTransform> unicodechar_tokenizer = std::make_shared<text::UnicodeCharTokenizer>(true);
  EXPECT_NE(unicodechar_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({unicodechar_tokenizer}, {"text"}, {"token", "offsets_start", "offsets_limit"},
               {"token", "offsets_start", "offsets_limit"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::vector<std::string>> expected_tokens = {
    {"W", "e", "l", "c", "o", "m", "e", " ", "t", "o", " ", "B", "e", "i", "j", "i", "n", "g", "!"},
    {"北", "京", "欢", "迎", "您", "！"},
    {"我", "喜", "欢", "E", "n", "g", "l", "i", "s", "h", "!"},
    {" ", " "}};

  std::vector<std::vector<uint32_t>> expected_offsets_start = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
    {0, 3, 6, 9, 12, 15},
    {0, 3, 6, 9, 10, 11, 12, 13, 14, 15, 16},
    {0, 1}};

  std::vector<std::vector<uint32_t>> expected_offsets_limit = {
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
    {3, 6, 9, 12, 15, 18},
    {3, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17},
    {1, 2}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto token = row["token"];
    auto start = row["offsets_start"];
    auto limit = row["offsets_limit"];

    std::shared_ptr<Tensor> de_expected_tokens;
    int x = expected_tokens[i].size();
    ASSERT_OK(Tensor::CreateFromVector(expected_tokens[i], TensorShape({x}), &de_expected_tokens));
    mindspore::MSTensor ms_expected_tokens =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tokens));
    EXPECT_MSTENSOR_EQ(token, ms_expected_tokens);

    std::shared_ptr<Tensor> de_expected_offsets_start;
    ASSERT_OK(Tensor::CreateFromVector(expected_offsets_start[i], TensorShape({x}), &de_expected_offsets_start));
    mindspore::MSTensor ms_expected_offsets_start =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_offsets_start));
    EXPECT_MSTENSOR_EQ(start, ms_expected_offsets_start);

    std::shared_ptr<Tensor> de_expected_offsets_limit;
    ASSERT_OK(Tensor::CreateFromVector(expected_offsets_limit[i], TensorShape({x}), &de_expected_offsets_limit));
    mindspore::MSTensor ms_expected_offsets_limit =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_offsets_limit));
    EXPECT_MSTENSOR_EQ(limit, ms_expected_offsets_limit);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestUnicodeScriptTokenizerSuccess) {
  // Testing the parameter of UnicodeScriptTokenizer interface when the with_offsets and the keep_whitespace is default.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUnicodeScriptTokenizerSuccess.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create unicodescript_tokenizer operation on ds
  std::shared_ptr<TensorTransform> unicodescript_tokenizer = std::make_shared<text::UnicodeScriptTokenizer>();
  EXPECT_NE(unicodescript_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({unicodescript_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::vector<std::string>> expected = {
    {"Welcome", "to", "Beijing", "!"}, {"北京欢迎您", "！"}, {"我喜欢", "English", "!"}, {""}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];

    std::shared_ptr<Tensor> de_expected_tensor;
    int x = expected[i].size();
    ASSERT_OK(Tensor::CreateFromVector(expected[i], TensorShape({x}), &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestUnicodeScriptTokenizerSuccess1) {
  // Testing the parameter of UnicodeScriptTokenizer interface when the keep_whitespace is true and the with_offsets is
  // false.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUnicodeScriptTokenizerSuccess1.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create unicodescript_tokenizer operation on ds
  std::shared_ptr<TensorTransform> unicodescript_tokenizer = std::make_shared<text::UnicodeScriptTokenizer>(true);
  EXPECT_NE(unicodescript_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({unicodescript_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::vector<std::string>> expected = {
    {"Welcome", " ", "to", " ", "Beijing", "!"}, {"北京欢迎您", "！"}, {"我喜欢", "English", "!"}, {"  "}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];

    std::shared_ptr<Tensor> de_expected_tensor;
    int x = expected[i].size();
    ASSERT_OK(Tensor::CreateFromVector(expected[i], TensorShape({x}), &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestUnicodeScriptTokenizerSuccess2) {
  // Testing the parameter of UnicodeScriptTokenizer interface when the keep_whitespace is false and the with_offsets is
  // true.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUnicodeScriptTokenizerSuccess2.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create unicodescript_tokenizer operation on ds
  std::shared_ptr<TensorTransform> unicodescript_tokenizer =
    std::make_shared<text::UnicodeScriptTokenizer>(false, true);
  EXPECT_NE(unicodescript_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({unicodescript_tokenizer}, {"text"}, {"token", "offsets_start", "offsets_limit"},
               {"token", "offsets_start", "offsets_limit"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::vector<std::string>> expected_tokens = {
    {"Welcome", "to", "Beijing", "!"}, {"北京欢迎您", "！"}, {"我喜欢", "English", "!"}, {""}};

  std::vector<std::vector<uint32_t>> expected_offsets_start = {{0, 8, 11, 18}, {0, 15}, {0, 9, 16}, {0}};
  std::vector<std::vector<uint32_t>> expected_offsets_limit = {{7, 10, 18, 19}, {15, 18}, {9, 16, 17}, {0}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto token = row["token"];
    auto start = row["offsets_start"];
    auto limit = row["offsets_limit"];

    std::shared_ptr<Tensor> de_expected_tokens;
    int x = expected_tokens[i].size();
    ASSERT_OK(Tensor::CreateFromVector(expected_tokens[i], TensorShape({x}), &de_expected_tokens));
    mindspore::MSTensor ms_expected_tokens =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tokens));
    EXPECT_MSTENSOR_EQ(token, ms_expected_tokens);

    std::shared_ptr<Tensor> de_expected_offsets_start;
    ASSERT_OK(Tensor::CreateFromVector(expected_offsets_start[i], TensorShape({x}), &de_expected_offsets_start));
    mindspore::MSTensor ms_expected_offsets_start =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_offsets_start));
    EXPECT_MSTENSOR_EQ(start, ms_expected_offsets_start);

    std::shared_ptr<Tensor> de_expected_offsets_limit;
    ASSERT_OK(Tensor::CreateFromVector(expected_offsets_limit[i], TensorShape({x}), &de_expected_offsets_limit));
    mindspore::MSTensor ms_expected_offsets_limit =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_offsets_limit));
    EXPECT_MSTENSOR_EQ(limit, ms_expected_offsets_limit);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestUnicodeScriptTokenizerSuccess3) {
  // Testing the parameter of UnicodeScriptTokenizer interface when the keep_whitespace is true and the with_offsets is
  // true.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUnicodeScriptTokenizerSuccess3.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create unicodescript_tokenizer operation on ds
  std::shared_ptr<TensorTransform> unicodescript_tokenizer = std::make_shared<text::UnicodeScriptTokenizer>(true, true);
  EXPECT_NE(unicodescript_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({unicodescript_tokenizer}, {"text"}, {"token", "offsets_start", "offsets_limit"},
               {"token", "offsets_start", "offsets_limit"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::vector<std::string>> expected_tokens = {
    {"Welcome", " ", "to", " ", "Beijing", "!"}, {"北京欢迎您", "！"}, {"我喜欢", "English", "!"}, {"  "}};

  std::vector<std::vector<uint32_t>> expected_offsets_start = {{0, 7, 8, 10, 11, 18}, {0, 15}, {0, 9, 16}, {0}};
  std::vector<std::vector<uint32_t>> expected_offsets_limit = {{7, 8, 10, 11, 18, 19}, {15, 18}, {9, 16, 17}, {2}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto token = row["token"];
    auto start = row["offsets_start"];
    auto limit = row["offsets_limit"];

    std::shared_ptr<Tensor> de_expected_tokens;
    int x = expected_tokens[i].size();
    ASSERT_OK(Tensor::CreateFromVector(expected_tokens[i], TensorShape({x}), &de_expected_tokens));
    mindspore::MSTensor ms_expected_tokens =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tokens));
    EXPECT_MSTENSOR_EQ(token, ms_expected_tokens);

    std::shared_ptr<Tensor> de_expected_offsets_start;
    ASSERT_OK(Tensor::CreateFromVector(expected_offsets_start[i], TensorShape({x}), &de_expected_offsets_start));
    mindspore::MSTensor ms_expected_offsets_start =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_offsets_start));
    EXPECT_MSTENSOR_EQ(start, ms_expected_offsets_start);

    std::shared_ptr<Tensor> de_expected_offsets_limit;
    ASSERT_OK(Tensor::CreateFromVector(expected_offsets_limit[i], TensorShape({x}), &de_expected_offsets_limit));
    mindspore::MSTensor ms_expected_offsets_limit =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_offsets_limit));
    EXPECT_MSTENSOR_EQ(limit, ms_expected_offsets_limit);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestWhitespaceTokenizerSuccess) {
  // Testing the parameter of WhitespaceTokenizer interface when the with_offsets is default.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestWhitespaceTokenizerSuccess.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create white_tokenizer operation on ds
  std::shared_ptr<TensorTransform> white_tokenizer = std::make_shared<text::WhitespaceTokenizer>();
  EXPECT_NE(white_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({white_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::vector<std::string>> expected = {
    {"This", "is", "a", "text", "file."}, {"Be", "happy", "every", "day."}, {"Good", "luck", "to", "everyone."}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];

    std::shared_ptr<Tensor> de_expected_tensor;
    int x = expected[i].size();
    ASSERT_OK(Tensor::CreateFromVector(expected[i], TensorShape({x}), &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestWhitespaceTokenizerSuccess1) {
  // Testing the parameter of WhitespaceTokenizer interface when the with_offsets is true.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestWhitespaceTokenizerSuccess1.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create white_tokenizer operation on ds
  std::shared_ptr<TensorTransform> white_tokenizer = std::make_shared<text::WhitespaceTokenizer>(true);
  EXPECT_NE(white_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({white_tokenizer}, {"text"}, {"token", "offsets_start", "offsets_limit"},
               {"token", "offsets_start", "offsets_limit"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<std::vector<std::string>> expected_tokens = {
    {"Welcome", "to", "Beijing!"}, {"北京欢迎您！"}, {"我喜欢English!"}, {""}};

  std::vector<std::vector<uint32_t>> expected_offsets_start = {{0, 8, 11}, {0}, {0}, {0}};
  std::vector<std::vector<uint32_t>> expected_offsets_limit = {{7, 10, 19}, {18}, {17}, {0}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto token = row["token"];
    auto start = row["offsets_start"];
    auto limit = row["offsets_limit"];

    std::shared_ptr<Tensor> de_expected_tokens;
    int x = expected_tokens[i].size();
    ASSERT_OK(Tensor::CreateFromVector(expected_tokens[i], TensorShape({x}), &de_expected_tokens));
    mindspore::MSTensor ms_expected_tokens =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tokens));
    EXPECT_MSTENSOR_EQ(token, ms_expected_tokens);

    std::shared_ptr<Tensor> de_expected_offsets_start;
    ASSERT_OK(Tensor::CreateFromVector(expected_offsets_start[i], TensorShape({x}), &de_expected_offsets_start));
    mindspore::MSTensor ms_expected_offsets_start =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_offsets_start));
    EXPECT_MSTENSOR_EQ(start, ms_expected_offsets_start);

    std::shared_ptr<Tensor> de_expected_offsets_limit;
    ASSERT_OK(Tensor::CreateFromVector(expected_offsets_limit[i], TensorShape({x}), &de_expected_offsets_limit));
    mindspore::MSTensor ms_expected_offsets_limit =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_offsets_limit));
    EXPECT_MSTENSOR_EQ(limit, ms_expected_offsets_limit);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}
