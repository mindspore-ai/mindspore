/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include <string>
#include <vector>

#include "common/common.h"
#include "include/api/status.h"
#include "minddata/dataset/include/dataset/config.h"
#include "minddata/dataset/include/dataset/datasets.h"
#include "minddata/dataset/include/dataset/text.h"
#include "minddata/dataset/include/dataset/transforms.h"
#include "minddata/dataset/text/char_n_gram.h"
#include "minddata/dataset/text/fast_text.h"
#include "minddata/dataset/text/glove.h"
#include "minddata/dataset/text/vectors.h"

using namespace mindspore::dataset;
using mindspore::Status;
using mindspore::dataset::CharNGram;
using mindspore::dataset::FastText;
using mindspore::dataset::GloVe;
using mindspore::dataset::ShuffleMode;
using mindspore::dataset::Tensor;
using mindspore::dataset::Vectors;
using mindspore::dataset::Vocab;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

/// Feature: BasicTokenizer op
/// Description: Test BasicTokenizer op on TextFileDataset with default inputs
/// Expectation: Output is equal to the expected output
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
  ASSERT_OK(iter->GetNextRow(&row));

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

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: BasicTokenizer op
/// Description: Test BasicTokenizer op on TextFileDataset with lower_case=true
/// Expectation: Output is equal to the expected output
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
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::string> expected = {"this", "is", "a", "funky", "string"};
  std::shared_ptr<Tensor> de_expected_tensor;
  ASSERT_OK(Tensor::CreateFromVector(expected, &de_expected_tensor));
  mindspore::MSTensor expected_tensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: BasicTokenizer op
/// Description: Test BasicTokenizer op on TextFileDataset with with_offsets=true and lower_case=true
/// Expectation: Output is equal to the expected output
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
  ASSERT_OK(iter->GetNextRow(&row));

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

    ASSERT_OK(iter->GetNextRow(&row));
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

/// Feature: BertTokenizer op
/// Description: Test BertTokenizer op on TextFileDataset with default parameters
/// Expectation: Output is equal to the expected output
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
  ASSERT_OK(iter->GetNextRow(&row));

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

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: BertTokenizer op
/// Description: Test BertTokenizer op on TextFileDataset with lower_case=true
/// Expectation: Output is equal to the expected output
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
  ASSERT_OK(iter->GetNextRow(&row));

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
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: BertTokenizer op
/// Description: Test BertTokenizer op on TextFileDataset with NormalizeForm::kNfc
/// Expectation: Output is equal to the expected output
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
  ASSERT_OK(iter->GetNextRow(&row));

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

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: BertTokenizer op
/// Description: Test BertTokenizer op on TextFileDataset with keep_whitespace=true
/// Expectation: Output is equal to the expected output
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
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::string> expected = {"[UNK]", " ", "[CLS]"};
  std::shared_ptr<Tensor> de_expected_tensor;
  ASSERT_OK(Tensor::CreateFromVector(expected, &de_expected_tensor));
  mindspore::MSTensor expected_tensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: BertTokenizer op
/// Description: Test BertTokenizer op on TextFileDataset with empty unknown_token and keep_whitespace=true
/// Expectation: Output is equal to the expected output
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
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::string> expected = {"unused", " ", "[CLS]"};
  std::shared_ptr<Tensor> de_expected_tensor;
  ASSERT_OK(Tensor::CreateFromVector(expected, &de_expected_tensor));
  mindspore::MSTensor expected_tensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: BertTokenizer op
/// Description: Test BertTokenizer op with preserve_unused_token=false, empty unknown_token, and keep_whitespace=true
/// Expectation: Output is equal to the expected output
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
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::string> expected = {"unused", " ", "[", "CLS", "]"};
  std::shared_ptr<Tensor> de_expected_tensor;
  ASSERT_OK(Tensor::CreateFromVector(expected, &de_expected_tensor));
  mindspore::MSTensor expected_tensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: BertTokenizer op
/// Description: Test BertTokenizer op with with_offsets=true and lower_case=true
/// Expectation: Output is equal to the expected output
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
  ASSERT_OK(iter->GetNextRow(&row));

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

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: BertTokenizer op
/// Description: Test BertTokenizer op with nullptr vocab
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
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

/// Feature: BertTokenizer op
/// Description: Test BertTokenizer op with negative max_bytes_per_token
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
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

/// Feature: CaseFold op
/// Description: Test CaseFold op on TextFileDataset with default parameters
/// Expectation: Output is equal to the expected output
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
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::string> expected = {"welcome to beijing!", "北京欢迎您!", "我喜欢english!", "  "};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateScalar(expected[i], &de_expected_tensor));
    mindspore::MSTensor ms_expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: FilterWikipediaXML op
/// Description: Test FilterWikipediaXML op in pipeline mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestFilterWikipediaXMLSuccess) {
  // Testing the parameter of FilterWikipediaXML interface .
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFilterWikipediaXMLSuccess.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/2.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create filter_wikipedia_xml operation on ds
  std::shared_ptr<TensorTransform> filter_wikipedia_xml = std::make_shared<text::FilterWikipediaXML>();
  EXPECT_NE(filter_wikipedia_xml, nullptr);

  // Create Map operation on ds
  ds = ds->Map({filter_wikipedia_xml}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  std::vector<std::string> expected = {"welcome to beijing", "", ""};

  uint64_t i = 0;

  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateScalar(expected[i], &de_expected_tensor));
    mindspore::MSTensor ms_expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: JiebaTokenizer op
/// Description: Test JiebaTokenizer op when the mode is JiebaMode::kMp and with_offsets=false
/// Expectation: Output is equal to the expected output
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
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::string> expected = {"今天天气", "太好了", "我们", "一起", "去", "外面", "玩吧"};
  std::shared_ptr<Tensor> de_expected_tensor;
  ASSERT_OK(Tensor::CreateFromVector(expected, &de_expected_tensor));
  mindspore::MSTensor expected_tensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: JiebaTokenizer op
/// Description: Test JiebaTokenizer op when the mode is JiebaMode::kHmm and with_offsets=false
/// Expectation: Output is equal to the expected output
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
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::string> expected = {"今天", "天气", "太", "好", "了", "我们", "一起", "去", "外面", "玩", "吧"};
  std::shared_ptr<Tensor> de_expected_tensor;
  ASSERT_OK(Tensor::CreateFromVector(expected, &de_expected_tensor));
  mindspore::MSTensor expected_tensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: JiebaTokenizer op
/// Description: Test JiebaTokenizer op when the mode is JiebaMode::kMp and with_offsets=true
/// Expectation: Output is equal to the expected output
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
  ds = ds->Map({jieba_tokenizer}, {"text"}, {"token", "offsets_start", "offsets_limit"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

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

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: JiebaTokenizer op
/// Description: Test JiebaTokenizer op with empty hmm_path
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
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

/// Feature: JiebaTokenizer op
/// Description: Test JiebaTokenizer op with empty mp_path
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
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

/// Feature: JiebaTokenizer op
/// Description: Test JiebaTokenizer op with invalid hmm_path
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
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

/// Feature: JiebaTokenizer op
/// Description: Test JiebaTokenizer op with invalid mp_path
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
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

/// Feature: JiebaTokenizer op
/// Description: Test AddWord of JiebaTokenizer when the freq is not provided (default 0)
/// Expectation: Output is equal to the expected output
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
  ASSERT_OK(jieba_tokenizer->AddWord("男默女泪"));

  // Create Map operation on ds
  ds = ds->Map({jieba_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::string> expected = {"男默女泪", "市", "长江大桥"};
  std::shared_ptr<Tensor> de_expected_tensor;
  ASSERT_OK(Tensor::CreateFromVector(expected, &de_expected_tensor));
  mindspore::MSTensor expected_tensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: JiebaTokenizer op
/// Description: Test AddWord of JiebaTokenizer when the freq is set explicitly to 0
/// Expectation: Output is equal to the expected output
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
  ASSERT_OK(jieba_tokenizer->AddWord("男默女泪", 0));

  // Create Map operation on ds
  ds = ds->Map({jieba_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::string> expected = {"男默女泪", "市", "长江大桥"};
  std::shared_ptr<Tensor> de_expected_tensor;
  ASSERT_OK(Tensor::CreateFromVector(expected, &de_expected_tensor));
  mindspore::MSTensor expected_tensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: JiebaTokenizer op
/// Description: Test AddWord of JiebaTokenizer when the freq is set to 10
/// Expectation: Output is equal to the expected output
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
  ASSERT_OK(jieba_tokenizer->AddWord("男默女泪", 10));

  // Create Map operation on ds
  ds = ds->Map({jieba_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::string> expected = {"男默女泪", "市", "长江大桥"};
  std::shared_ptr<Tensor> de_expected_tensor;
  ASSERT_OK(Tensor::CreateFromVector(expected, &de_expected_tensor));
  mindspore::MSTensor expected_tensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: JiebaTokenizer op
/// Description: Test AddWord of JiebaTokenizer when the freq is 20000 which affects the result of segmentation
/// Expectation: Output is equal to the expected output
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
  ASSERT_OK(jieba_tokenizer->AddWord("江大桥", 20000));

  // Create Map operation on ds
  ds = ds->Map({jieba_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::string> expected = {"江州", "市长", "江大桥", "参加", "了", "长江大桥", "的", "通车", "仪式"};
  std::shared_ptr<Tensor> de_expected_tensor;
  ASSERT_OK(Tensor::CreateFromVector(expected, &de_expected_tensor));
  mindspore::MSTensor expected_tensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: JiebaTokenizer op
/// Description: Test AddWord of JiebaTokenizer with invalid parameters
/// Expectation: Throw correct error and message
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

/// Feature: JiebaTokenizer op
/// Description: Test AddDict of JiebaTokenizer when the input is a vector of word-freq pair
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestJiebaTokenizerAddDict) {
  // Testing AddDict of JiebaTokenizer when the input is a vector of word-freq pair.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestJiebaTokenizerAddDict.";

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
  std::vector<std::pair<std::string, int64_t>> user_dict = {{"江大桥", 20000}};
  ASSERT_OK(jieba_tokenizer->AddDict(user_dict));

  // Create Map operation on ds
  ds = ds->Map({jieba_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::string> expected = {"江州", "市长", "江大桥", "参加", "了", "长江大桥", "的", "通车", "仪式"};
  std::shared_ptr<Tensor> de_expected_tensor;
  ASSERT_OK(Tensor::CreateFromVector(expected, &de_expected_tensor));
  mindspore::MSTensor expected_tensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto txt = row["text"];
    EXPECT_MSTENSOR_EQ(txt, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: JiebaTokenizer op
/// Description: Test AddDict of JiebaTokenizer when the input is a path to dict
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestJiebaTokenizerAddDictFromFile) {
  // Testing AddDict of JiebaTokenizer when the input is a path to dict.
  // Test error scenario for AddDict: invalid path
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestJiebaTokenizerAddDictFromFile.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testJiebaDataset/3.txt";
  std::string hmm_path = datasets_root_path_ + "/jiebadict/hmm_model.utf8";
  std::string mp_path = datasets_root_path_ + "/jiebadict/jieba.dict.utf8";
  std::shared_ptr<Dataset> ds = TextFile({data_file});
  EXPECT_NE(ds, nullptr);

  // Create jieba_tokenizer operation on ds
  std::shared_ptr<text::JiebaTokenizer> jieba_tokenizer =
    std::make_shared<text::JiebaTokenizer>(hmm_path, mp_path, JiebaMode::kMp);
  EXPECT_NE(jieba_tokenizer, nullptr);

  // Load dict from txt file
  std::string user_dict_path = datasets_root_path_ + "/testJiebaDataset/user_dict.txt";
  std::string invalid_path = datasets_root_path_ + "/testJiebaDataset/invalid_path.txt";
  EXPECT_ERROR(jieba_tokenizer->AddDict(invalid_path));
  ASSERT_OK(jieba_tokenizer->AddDict(user_dict_path));

  // Create Map operation on ds
  ds = ds->Map({jieba_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::string> expected = {"今天天气", "太好了", "我们", "一起", "去", "外面", "玩吧"};
  std::shared_ptr<Tensor> de_expected_tensor;
  ASSERT_OK(Tensor::CreateFromVector(expected, &de_expected_tensor));
  mindspore::MSTensor expected_tensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto txt = row["text"];
    EXPECT_MSTENSOR_EQ(txt, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: SlidingWindow op
/// Description: Test SlidingWindow when the axis is 0
/// Expectation: Output is equal to the expected output
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
  ASSERT_OK(iter->GetNextRow(&row));

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

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: SlidingWindow op
/// Description: Test SlidingWindow when the axis is -1
/// Expectation: Output is equal to the expected output
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
  ASSERT_OK(iter->GetNextRow(&row));

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

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: SlidingWindow op
/// Description: Test SlidingWindow when the width=0
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
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

/// Feature: SlidingWindow op
/// Description: Test SlidingWindow when the width=-2
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
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

/// Feature: ToNumber op
/// Description: Test ToNumber with integer numbers
/// Expectation: Output is equal to the expected output
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
  std::shared_ptr<TensorTransform> to_number = std::make_shared<text::ToNumber>(mindspore::DataType::kNumberTypeInt64);
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
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<int64_t> expected = {-121, 14, -2219, 7623, -8162536, 162371864, -1726483716, 98921728421};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateScalar(expected[i], &de_expected_tensor));
    mindspore::MSTensor ms_expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 8);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: ToNumber op
/// Description: Test ToNumber with float numbers
/// Expectation: Output is equal to the expected output
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
  std::shared_ptr<TensorTransform> to_number =
    std::make_shared<text::ToNumber>(mindspore::DataType::kNumberTypeFloat64);
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
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<double_t> expected = {-1.1, 1.4, -2219.321, 7623.453, -816256.234282, 162371864.243243};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateScalar(expected[i], &de_expected_tensor));
    mindspore::MSTensor ms_expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: ToNumber op
/// Description: Test ToNumber with overflow integer numbers
/// Expectation: Throw correct error and message
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
  std::shared_ptr<TensorTransform> to_number = std::make_shared<text::ToNumber>(mindspore::DataType::kNumberTypeInt8);
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
  EXPECT_ERROR(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    EXPECT_ERROR(iter->GetNextRow(&row));
    i++;
  }

  // Expect failure: GetNextRow fail and return nothing
  EXPECT_EQ(i, 0);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: ToNumber op
/// Description: Test ToNumber with overflow float numbers
/// Expectation: Throw correct error and message
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
  std::shared_ptr<TensorTransform> to_number =
    std::make_shared<text::ToNumber>(mindspore::DataType::kNumberTypeFloat16);
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
  EXPECT_ERROR(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    EXPECT_ERROR(iter->GetNextRow(&row));
    i++;
  }

  // Expect failure: GetNextRow fail and return nothing
  EXPECT_EQ(i, 0);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: ToNumber op
/// Description: Test ToNumber with non numerical input
/// Expectation: Throw correct error and message
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
  std::shared_ptr<TensorTransform> to_number = std::make_shared<text::ToNumber>(mindspore::DataType::kNumberTypeInt64);
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
  EXPECT_ERROR(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    EXPECT_ERROR(iter->GetNextRow(&row));
    i++;
  }

  // Expect failure: GetNextRow fail and return nothing
  EXPECT_EQ(i, 0);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: ToNumber op
/// Description: Test ToNumber with non numerical data type (kObjectTypeString)
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestToNumberFail4) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestToNumberFail4.";
  // Test ToNumber with non numerical data type

  std::string data_file = datasets_root_path_ + "/testTokenizerData/to_number.txt";

  // Create a TextFile dataset
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create ToNumber operation on ds
  std::shared_ptr<TensorTransform> to_number = std::make_shared<text::ToNumber>(mindspore::DataType::kObjectTypeString);
  EXPECT_NE(to_number, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({to_number}, {"text"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid parameter with non numerical data type
  EXPECT_EQ(iter, nullptr);
}

/// Feature: ToNumber op
/// Description: Test ToNumber with non numerical data type (kObjectTypeBool)
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestToNumberFail5) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestToNumberFail5.";
  // Test ToNumber with non numerical data type

  std::string data_file = datasets_root_path_ + "/testTokenizerData/to_number.txt";

  // Create a TextFile dataset
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create ToNumber operation on ds
  std::shared_ptr<TensorTransform> to_number = std::make_shared<text::ToNumber>(mindspore::DataType::kNumberTypeBool);
  EXPECT_NE(to_number, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({to_number}, {"text"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid parameter with non numerical data type
  EXPECT_EQ(iter, nullptr);
}

/// Feature: TruncateSequencePair op
/// Description: Test TruncateSequencePair basic usage
/// Expectation: Output is equal to the expected output
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
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeInt16, {5}));
  ASSERT_OK(schema->add_column("col2", mindspore::DataType::kNumberTypeInt32, {3}));
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
  ASSERT_OK(iter->GetNextRow(&row));

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

    ASSERT_OK(iter->GetNextRow(&row));
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

/// Feature: TruncateSequencePair op
/// Description: Test TruncateSequencePair with odd max_length
/// Expectation: Output is equal to the expected output
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
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeInt32, {4}));
  ASSERT_OK(schema->add_column("col2", mindspore::DataType::kNumberTypeInt64, {4}));
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
  ASSERT_OK(iter->GetNextRow(&row));

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

    ASSERT_OK(iter->GetNextRow(&row));
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

/// Feature: TruncateSequencePair op
/// Description: Test TruncateSequencePair with negative max_length
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestTruncateSequencePairFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTruncateSequencePairFail.";
  // Testing TruncateSequencePair with negative max_length

  // Create a RandomDataset which has column names "col1" and "col2"
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeInt8, {3}));
  ASSERT_OK(schema->add_column("col2", mindspore::DataType::kNumberTypeInt8, {3}));
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

/// Feature: Ngram op
/// Description: Test parameters for Ngram interface
/// Expectation: Output is equal to the expected output
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
  auto ngram_op = std::make_shared<text::Ngram>(
    std::vector<int>{2}, std::pair<std::string, int32_t>{"_", 1}, std::pair<std::string, int32_t>{"_", 1}, " ");
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
  ASSERT_OK(iter->GetNextRow(&row));

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

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Ngram op
/// Description: Test Ngram basic usage
/// Expectation: Output is equal to the expected output
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
  auto ngram_op = std::make_shared<text::Ngram>(
    std::vector<int32_t>{2, 3}, std::pair<std::string, int32_t>{"&", 2}, std::pair<std::string, int32_t>{"&", 2}, "-");
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
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::vector<std::string>> expected = {
    {"&-This", "This-is", "is-a", "a-text", "text-file.", "file.-&", "&-&-This", "&-This-is", "This-is-a", "is-a-text",
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

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Ngram op
/// Description: Test Ngram where the vector of ngram is empty
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestNgramFail1) {
  // Testing the incorrect parameter of Ngram interface.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNgramFail1.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create sliding_window operation on ds
  // Testing the vector of ngram is empty
  auto ngram_op = std::make_shared<text::Ngram>(std::vector<int32_t>{});
  EXPECT_NE(ngram_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({ngram_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Ngram input (the vector of ngram is empty)
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Ngram op
/// Description: Test Ngram where value of ngram vector is equal to 0
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestNgramFail2) {
  // Testing the incorrect parameter of Ngram interface.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNgramFail2.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create sliding_window operation on ds
  // Testing the value of ngrams vector less than and equal to 0
  auto ngram_op = std::make_shared<text::Ngram>(std::vector<int32_t>{0});
  EXPECT_NE(ngram_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({ngram_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Ngram input (the value of ngrams vector less than and equal to 0)
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Ngram op
/// Description: Test Ngram where value of ngram vector is less than 0
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestNgramFail3) {
  // Testing the incorrect parameter of Ngram interface.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNgramFail3.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create sliding_window operation on ds
  // Testing the value of ngrams vector less than and equal to 0
  auto ngram_op = std::make_shared<text::Ngram>(std::vector<int32_t>{-2});
  EXPECT_NE(ngram_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({ngram_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Ngram input (the value of ngrams vector less than and equal to 0)
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Ngram op
/// Description: Test Ngram where second parameter pad_width in left_pad vector is less than 0
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestNgramFail4) {
  // Testing the incorrect parameter of Ngram interface.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNgramFail4.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create sliding_window operation on ds
  // Testing the second parameter pad_width in left_pad vector less than 0
  auto ngram_op = std::make_shared<text::Ngram>(std::vector<int32_t>{2}, std::pair<std::string, int32_t>{"", -1});
  EXPECT_NE(ngram_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({ngram_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Ngram input (the second parameter pad_width in left_pad vector less than 0)
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Ngram op
/// Description: Test Ngram where second parameter pad_width in right_pad vector is less than 0
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestNgramFail5) {
  // Testing the incorrect parameter of Ngram interface.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNgramFail5.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create sliding_window operation on ds
  // Testing the second parameter pad_width in right_pad vector less than 0
  auto ngram_op = std::make_shared<text::Ngram>(
    std::vector<int32_t>{2}, std::pair<std::string, int32_t>{"", 1}, std::pair<std::string, int32_t>{"", -1});
  EXPECT_NE(ngram_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({ngram_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Ngram input (the second parameter pad_width in left_pad vector less than 0)
  EXPECT_EQ(iter, nullptr);
}

/// Feature: NormalizeUTF8 op
/// Description: Test NormalizeUTF8 when the normalize_form is NormalizeForm::kNfkc
/// Expectation: Output is equal to the expected output
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
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::string> expected = {"ṩ", "ḍ̇", "q̣̇", "fi", "25", "ṩ"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateScalar(expected[i], &de_expected_tensor));
    mindspore::MSTensor ms_expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: NormalizeUTF8 op
/// Description: Test NormalizeUTF8 when the normalize_form is NormalizeForm::kNfc
/// Expectation: Output is equal to the expected output
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
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::string> expected = {"ṩ", "ḍ̇", "q̣̇", "ﬁ", "2⁵", "ẛ̣"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateScalar(expected[i], &de_expected_tensor));
    mindspore::MSTensor ms_expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: NormalizeUTF8 op
/// Description: Test NormalizeUTF8 when the normalize_form is NormalizeForm::kNfd
/// Expectation: Output is equal to the expected output
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
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::string> expected = {"ṩ", "ḍ̇", "q̣̇", "ﬁ", "2⁵", "ẛ̣"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateScalar(expected[i], &de_expected_tensor));
    mindspore::MSTensor ms_expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: NormalizeUTF8 op
/// Description: Test NormalizeUTF8 when the normalize_form is NormalizeForm::kNfkd
/// Expectation: Output is equal to the expected output
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
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::string> expected = {"ṩ", "ḍ̇", "q̣̇", "fi", "25", "ṩ"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateScalar(expected[i], &de_expected_tensor));
    mindspore::MSTensor ms_expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: RegexReplace op
/// Description: Test RegexReplace when the replace_all=true
/// Expectation: Output is equal to the expected output
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
  ASSERT_OK(iter->GetNextRow(&row));

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
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 8);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: RegexReplace op
/// Description: Test RegexReplace when the replace_all=false
/// Expectation: Output is equal to the expected output
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
  ASSERT_OK(iter->GetNextRow(&row));

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
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 8);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: RegexTokenizer op
/// Description: Test RegexTokenizer when with_offsets=false
/// Expectation: Output is equal to the expected output
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
  ASSERT_OK(iter->GetNextRow(&row));

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

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 8);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: RegexTokenizer op
/// Description: Test RegexTokenizer when with_offsets=true
/// Expectation: Output is equal to the expected output
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
  ds = ds->Map({regex_tokenizer}, {"text"}, {"token", "offsets_start", "offsets_limit"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

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

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 8);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: UnicodeCharTokenizer op
/// Description: Test UnicodeCharTokenizer when with_offsets is default
/// Expectation: Output is equal to the expected output
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
  ASSERT_OK(iter->GetNextRow(&row));

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

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: UnicodeCharTokenizer op
/// Description: Test UnicodeCharTokenizer when with_offsets=true
/// Expectation: Output is equal to the expected output
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
  ds = ds->Map({unicodechar_tokenizer}, {"text"}, {"token", "offsets_start", "offsets_limit"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

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

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

std::vector<std::string> vocab_english = {"book", "cholera", "era", "favor", "##ite", "my",
                                          "is",   "love",    "dur", "##ing", "the"};

std::vector<std::string> vocab_chinese = {"我", "最", "喜", "欢", "的", "书", "是", "霍", "乱", "时", "期", "爱", "情"};

/// Feature: WordpieceTokenizer op
/// Description: Test WordpieceTokenizer with default parameters on English vocab
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestWordpieceTokenizerSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestWordpieceTokenizerSuccess1.";
  // Test WordpieceTokenizer with default parameters on English vocab

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/wordpiece_tokenizer.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create Take operation on ds
  ds = ds->Take(10);
  EXPECT_NE(ds, nullptr);

  // Create a vocab from vector
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  Status s = Vocab::BuildFromVector(vocab_english, {}, true, &vocab);
  EXPECT_EQ(s, Status::OK());

  // Create WordpieceTokenizer operation on ds
  std::shared_ptr<TensorTransform> wordpiece_tokenizer = std::make_shared<text::WordpieceTokenizer>(vocab);
  EXPECT_NE(wordpiece_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({wordpiece_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::vector<std::string>> expected = {
    {"my"}, {"favor", "##ite"}, {"book"}, {"is"}, {"love"}, {"dur", "##ing"}, {"the"}, {"cholera"}, {"era"}, {"[UNK]"}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto txt = row["text"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(txt, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: WordpieceTokenizer op
/// Description: Test WordpieceTokenizer with empty unknown_token
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestWordpieceTokenizerSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestWordpieceTokenizerSuccess2.";
  // Test WordpieceTokenizer with empty unknown_token

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/wordpiece_tokenizer.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create Take operation on ds
  ds = ds->Take(10);
  EXPECT_NE(ds, nullptr);

  // Create a vocab from vector
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  Status s = Vocab::BuildFromVector(vocab_english, {}, true, &vocab);
  EXPECT_EQ(s, Status::OK());

  // Create WordpieceTokenizer operation on ds
  std::shared_ptr<TensorTransform> wordpiece_tokenizer =
    std::make_shared<text::WordpieceTokenizer>(vocab, "##", 100, "", false);
  EXPECT_NE(wordpiece_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({wordpiece_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::vector<std::string>> expected = {
    {"my"}, {"favor", "##ite"}, {"book"}, {"is"}, {"love"}, {"dur", "##ing"}, {"the"}, {"cholera"}, {"era"}, {"what"}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto txt = row["text"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(txt, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: WordpieceTokenizer op
/// Description: Test WordpieceTokenizer with non-default max_bytes_per_token
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestWordpieceTokenizerSuccess3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestWordpieceTokenizerSuccess3.";
  // Test WordpieceTokenizer with non-default max_bytes_per_token

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/wordpiece_tokenizer.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create Take operation on ds
  ds = ds->Take(10);
  EXPECT_NE(ds, nullptr);

  // Create a vocab from vector
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  Status s = Vocab::BuildFromVector(vocab_english, {}, true, &vocab);
  EXPECT_EQ(s, Status::OK());

  // Create WordpieceTokenizer operation on ds
  std::shared_ptr<TensorTransform> wordpiece_tokenizer =
    std::make_shared<text::WordpieceTokenizer>(vocab, "##", 4, "[UNK]", false);
  EXPECT_NE(wordpiece_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({wordpiece_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::vector<std::string>> expected = {{"my"},    {"[UNK]"}, {"book"},  {"is"},  {"love"},
                                                    {"[UNK]"}, {"the"},   {"[UNK]"}, {"era"}, {"[UNK]"}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto txt = row["text"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(txt, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: WordpieceTokenizer op
/// Description: Test WordpieceTokenizer with default parameters on Chinese vocab
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestWordpieceTokenizerSuccess4) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestWordpieceTokenizerSuccess4.";
  // Test WordpieceTokenizer with default parameters on Chinese vocab

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/wordpiece_tokenizer.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create Skip operation on ds
  ds = ds->Skip(10);
  EXPECT_NE(ds, nullptr);

  // Create Take operation on ds
  ds = ds->Take(15);
  EXPECT_NE(ds, nullptr);

  // Create a vocab from vector
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  Status s = Vocab::BuildFromVector(vocab_chinese, {}, true, &vocab);
  EXPECT_EQ(s, Status::OK());

  // Create WordpieceTokenizer operation on ds
  std::shared_ptr<TensorTransform> wordpiece_tokenizer =
    std::make_shared<text::WordpieceTokenizer>(vocab, "##", 100, "[UNK]", false);
  EXPECT_NE(wordpiece_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({wordpiece_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::vector<std::string>> expected = {{"我"}, {"最"}, {"喜"}, {"欢"}, {"的"}, {"书"}, {"是"},   {"霍"},
                                                    {"乱"}, {"时"}, {"期"}, {"的"}, {"爱"}, {"情"}, {"[UNK]"}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto txt = row["text"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(txt, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 15);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: WordpieceTokenizer op
/// Description: Test WordpieceTokenizer with with_offsets=true
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestWordpieceTokenizerSuccess5) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestWordpieceTokenizerSuccess5.";
  // Test WordpieceTokenizer with with_offsets true

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/wordpiece_tokenizer.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create Take operation on ds
  ds = ds->Take(10);
  EXPECT_NE(ds, nullptr);

  // Create a vocab from vector
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  Status s = Vocab::BuildFromVector(vocab_english, {}, true, &vocab);
  EXPECT_EQ(s, Status::OK());

  // Create WordpieceTokenizer operation on ds
  std::shared_ptr<TensorTransform> wordpiece_tokenizer =
    std::make_shared<text::WordpieceTokenizer>(vocab, "##", 100, "[UNK]", true);
  EXPECT_NE(wordpiece_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({wordpiece_tokenizer}, {"text"}, {"token", "offsets_start", "offsets_limit"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::vector<std::string>> expected = {
    {"my"}, {"favor", "##ite"}, {"book"}, {"is"}, {"love"}, {"dur", "##ing"}, {"the"}, {"cholera"}, {"era"}, {"[UNK]"}};
  std::vector<std::vector<uint32_t>> expected_offsets_start = {{0}, {0, 5}, {0}, {0}, {0}, {0, 3}, {0}, {0}, {0}, {0}};
  std::vector<std::vector<uint32_t>> expected_offsets_limit = {{2}, {5, 8}, {4}, {2}, {4}, {3, 6}, {3}, {7}, {3}, {4}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto txt = row["token"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(txt, expected_tensor);

    auto start = row["offsets_start"];
    std::shared_ptr<Tensor> de_expected_start_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected_offsets_start[i], &de_expected_start_tensor));
    mindspore::MSTensor expected_start_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_start_tensor));
    EXPECT_MSTENSOR_EQ(start, expected_start_tensor);

    auto limit = row["offsets_limit"];
    std::shared_ptr<Tensor> de_expected_limit_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected_offsets_limit[i], &de_expected_limit_tensor));
    mindspore::MSTensor expected_limit_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_limit_tensor));
    EXPECT_MSTENSOR_EQ(limit, expected_limit_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: WordpieceTokenizer op
/// Description: Test WordpieceTokenizer with max_bytes_per_token=0
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestWordpieceTokenizerSuccess6) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestWordpieceTokenizerSuccess6.";
  // Test WordpieceTokenizer with max_bytes_per_token equals to 0

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/wordpiece_tokenizer.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create Take operation on ds
  ds = ds->Take(10);
  EXPECT_NE(ds, nullptr);

  // Create a vocab from vector
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  Status s = Vocab::BuildFromVector(vocab_english, {}, true, &vocab);
  EXPECT_EQ(s, Status::OK());

  // Create WordpieceTokenizer operation on ds
  std::shared_ptr<TensorTransform> wordpiece_tokenizer =
    std::make_shared<text::WordpieceTokenizer>(vocab, "##", 0, "[UNK]", true);
  EXPECT_NE(wordpiece_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({wordpiece_tokenizer}, {"text"}, {"token", "offsets_start", "offsets_limit"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::vector<std::string>> expected = {{"[UNK]"}, {"[UNK]"}, {"[UNK]"}, {"[UNK]"}, {"[UNK]"},
                                                    {"[UNK]"}, {"[UNK]"}, {"[UNK]"}, {"[UNK]"}, {"[UNK]"}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto txt = row["token"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(txt, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: WordpieceTokenizer op
/// Description: Test WordpieceTokenizer with nullptr vocab
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestWordpieceTokenizerFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestWordpieceTokenizerFail1.";
  // Test WordpieceTokenizer with nullptr vocab

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/bert_tokenizer.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create WordpieceTokenizer operation on ds
  std::shared_ptr<TensorTransform> wordpiece_tokenizer = std::make_shared<text::WordpieceTokenizer>(nullptr);
  EXPECT_NE(wordpiece_tokenizer, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({wordpiece_tokenizer});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid WordpieceTokenizer input with nullptr vocab
  EXPECT_EQ(iter, nullptr);
}

/// Feature: WordpieceTokenizer op
/// Description: Test WordpieceTokenizer with negative max_bytes_per_token
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestWordpieceTokenizerFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestWordpieceTokenizerFail2.";
  // Test WordpieceTokenizer with negative max_bytes_per_token

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/bert_tokenizer.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create a vocab from vector
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  Status s = Vocab::BuildFromVector(vocab_english, {}, true, &vocab);
  EXPECT_EQ(s, Status::OK());

  // Create WordpieceTokenizer operation on ds
  std::shared_ptr<TensorTransform> wordpiece_tokenizer = std::make_shared<text::WordpieceTokenizer>(vocab, "##", -1);
  EXPECT_NE(wordpiece_tokenizer, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({wordpiece_tokenizer});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid WordpieceTokenizer input with nullptr vocab
  EXPECT_EQ(iter, nullptr);
}

/// Feature: UnicodeScriptTokenizer op
/// Description: Test UnicodeScriptTokenizer when with_offsets and keep_whitespace is default
/// Expectation: Output is equal to the expected output
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
  ASSERT_OK(iter->GetNextRow(&row));

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

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: UnicodeScriptTokenizer op
/// Description: Test UnicodeScriptTokenizer when with_offsets=false and keep_whitespace=true
/// Expectation: Output is equal to the expected output
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
  ASSERT_OK(iter->GetNextRow(&row));

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

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: UnicodeScriptTokenizer op
/// Description: Test UnicodeScriptTokenizer when with_offsets=true and keep_whitespace=false
/// Expectation: Output is equal to the expected output
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
  ds = ds->Map({unicodescript_tokenizer}, {"text"}, {"token", "offsets_start", "offsets_limit"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

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

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: UnicodeScriptTokenizer op
/// Description: Test UnicodeScriptTokenizer when with_offsets=true and keep_whitespace=true
/// Expectation: Output is equal to the expected output
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
  ds = ds->Map({unicodescript_tokenizer}, {"text"}, {"token", "offsets_start", "offsets_limit"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

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

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: WhitespaceTokenizer op
/// Description: Test WhitespaceTokenizer when with_offsets is default
/// Expectation: Output is equal to the expected output
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
  ASSERT_OK(iter->GetNextRow(&row));

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

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: WhitespaceTokenizer op
/// Description: Test WhitespaceTokenizer when with_offsets=true
/// Expectation: Output is equal to the expected output
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
  ds = ds->Map({white_tokenizer}, {"text"}, {"token", "offsets_start", "offsets_limit"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

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

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Vectors
/// Description: Test with default parameter in function BuildFromFile and function Lookup
/// Expectation: Return correct MSTensor which is equal to the expected
TEST_F(MindDataTestPipeline, TestVectorsDefaultParam) {
  // Test with default parameter.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVectorsDefaultParam.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testVectors/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/testVectors/vectors.txt";
  std::shared_ptr<Vectors> vectors;
  Status s = Vectors::BuildFromFile(&vectors, vectors_dir);
  EXPECT_EQ(s, Status::OK());

  std::shared_ptr<TensorTransform> lookup = std::make_shared<text::ToVectors>(vectors);
  EXPECT_NE(lookup, nullptr);

  // Create Map operation on ds
  ds = ds->Map({lookup}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  std::vector<std::vector<float>> expected = {{0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411},
                                              {0, 0, 0, 0, 0, 0},
                                              {0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973},
                                              {0.70853, 0.57088, -0.4716, 0.18048, 0.54449, 0.72603},
                                              {0.68047, -0.039263, 0.30186, -0.17792, 0.42962, 0.032246},
                                              {0.26818, 0.14346, -0.27877, 0.016257, 0.11384, 0.69923},
                                              {0, 0, 0, 0, 0, 0}};
  while (row.size() != 0) {
    auto ind = row["text"];
    MS_LOG(INFO) << ind.Shape();
    TEST_MS_LOG_MSTENSOR(INFO, "ind: ", ind);
    TensorPtr de_expected_item;
    dsize_t dim = 6;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], TensorShape({dim}), &de_expected_item));
    mindspore::MSTensor ms_expected_item =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_item));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_item);

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 7);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Vectors
/// Description: Test with all parameters which include `path` and `max_vector` in function BuildFromFile
/// Expectation: Return correct MSTensor which is equal to the expected
TEST_F(MindDataTestPipeline, TestVectorsAllBuildfromfileParams) {
  // Test with two parameters.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVectorsAllBuildfromfileParams.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testVectors/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/testVectors/vectors.txt";
  std::shared_ptr<Vectors> vectors;
  Status s = Vectors::BuildFromFile(&vectors, vectors_dir, 100);
  EXPECT_EQ(s, Status::OK());

  std::shared_ptr<TensorTransform> lookup = std::make_shared<text::ToVectors>(vectors);
  EXPECT_NE(lookup, nullptr);

  // Create Map operation on ds
  ds = ds->Map({lookup}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  std::vector<std::vector<float>> expected = {{0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411},
                                              {0, 0, 0, 0, 0, 0},
                                              {0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973},
                                              {0.70853, 0.57088, -0.4716, 0.18048, 0.54449, 0.72603},
                                              {0.68047, -0.039263, 0.30186, -0.17792, 0.42962, 0.032246},
                                              {0.26818, 0.14346, -0.27877, 0.016257, 0.11384, 0.69923},
                                              {0, 0, 0, 0, 0, 0}};
  while (row.size() != 0) {
    auto ind = row["text"];
    MS_LOG(INFO) << ind.Shape();
    TEST_MS_LOG_MSTENSOR(INFO, "ind: ", ind);
    TensorPtr de_expected_item;
    dsize_t dim = 6;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], TensorShape({dim}), &de_expected_item));
    mindspore::MSTensor ms_expected_item =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_item));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_item);

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 7);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Vectors
/// Description: Test with all parameters in function BuildFromFile and `unknown_init` in function Lookup
/// Expectation: Return correct MSTensor which is equal to the expected
TEST_F(MindDataTestPipeline, TestVectorsUnknownInit) {
  // Test with two parameters.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVectorsUnknownInit.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testVectors/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/testVectors/vectors.txt";
  std::shared_ptr<Vectors> vectors;
  Status s = Vectors::BuildFromFile(&vectors, vectors_dir, 100);
  EXPECT_EQ(s, Status::OK());

  std::vector<float> unknown_init = {-1, -1, -1, -1, -1, -1};
  std::shared_ptr<TensorTransform> lookup = std::make_shared<text::ToVectors>(vectors, unknown_init);
  EXPECT_NE(lookup, nullptr);

  // Create Map operation on ds
  ds = ds->Map({lookup}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  std::vector<std::vector<float>> expected = {{0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411},
                                              {-1, -1, -1, -1, -1, -1},
                                              {0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973},
                                              {0.70853, 0.57088, -0.4716, 0.18048, 0.54449, 0.72603},
                                              {0.68047, -0.039263, 0.30186, -0.17792, 0.42962, 0.032246},
                                              {0.26818, 0.14346, -0.27877, 0.016257, 0.11384, 0.69923},
                                              {-1, -1, -1, -1, -1, -1}};
  while (row.size() != 0) {
    auto ind = row["text"];
    MS_LOG(INFO) << ind.Shape();
    TEST_MS_LOG_MSTENSOR(INFO, "ind: ", ind);
    TensorPtr de_expected_item;
    dsize_t dim = 6;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], TensorShape({dim}), &de_expected_item));
    mindspore::MSTensor ms_expected_item =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_item));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_item);

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 7);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Vectors
/// Description: Test with all parameters which include `path` and `max_vectors` in function BuildFromFile and `token`,
///     `unknown_init` and `lower_case_backup` in function Lookup. But some tokens have some big letters
/// Expectation: Return correct MSTensor which is equal to the expected
TEST_F(MindDataTestPipeline, TestVectorsAllParams) {
  // Test with all parameters.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVectorsAllParams.";
  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testVectors/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/testVectors/vectors.txt";
  std::shared_ptr<Vectors> vectors;
  Status s = Vectors::BuildFromFile(&vectors, vectors_dir);
  EXPECT_EQ(s, Status::OK());

  std::vector<float> unknown_init = {-1, -1, -1, -1, -1, -1};
  std::shared_ptr<TensorTransform> lookup = std::make_shared<text::ToVectors>(vectors, unknown_init, true);
  EXPECT_NE(lookup, nullptr);

  // Create Map operation on ds
  ds = ds->Map({lookup}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  std::vector<std::vector<float>> expected = {{0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411},
                                              {-1, -1, -1, -1, -1, -1},
                                              {0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973},
                                              {0.70853, 0.57088, -0.4716, 0.18048, 0.54449, 0.72603},
                                              {0.68047, -0.039263, 0.30186, -0.17792, 0.42962, 0.032246},
                                              {0.26818, 0.14346, -0.27877, 0.016257, 0.11384, 0.69923},
                                              {-1, -1, -1, -1, -1, -1}};
  while (row.size() != 0) {
    auto ind = row["text"];
    MS_LOG(INFO) << ind.Shape();
    TEST_MS_LOG_MSTENSOR(INFO, "ind: ", ind);
    TensorPtr de_expected_item;
    dsize_t dim = 6;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], TensorShape({dim}), &de_expected_item));
    mindspore::MSTensor ms_expected_item =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_item));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_item);

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 7);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Vectors
/// Description: Test with pre-vectors set that have the different dimension
/// Expectation: Throw correct error and message
TEST_F(MindDataTestPipeline, TestVectorsDifferentDimension) {
  // Tokens don't have the same number of vectors.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVectorsDifferentDimension.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testVectors/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/testVectors/vectors_dim_different.txt";
  std::shared_ptr<Vectors> vectors;
  Status s = Vectors::BuildFromFile(&vectors, vectors_dir, 100);
  EXPECT_NE(s, Status::OK());
}

/// Feature: Vectors
/// Description: Test with pre-vectors set that has the head-info
/// Expectation: Return correct MSTensor which is equal to the expected
TEST_F(MindDataTestPipeline, TestVectorsWithHeadInfo) {
  // Test with words that has head info.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVectorsWithHeadInfo.";
  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testVectors/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/testVectors/vectors_with_info.txt";
  std::shared_ptr<Vectors> vectors;
  Status s = Vectors::BuildFromFile(&vectors, vectors_dir);
  EXPECT_EQ(s, Status::OK());

  std::vector<float> unknown_init = {-1, -1, -1, -1, -1, -1};
  std::shared_ptr<TensorTransform> lookup = std::make_shared<text::ToVectors>(vectors, unknown_init, true);
  EXPECT_NE(lookup, nullptr);

  // Create Map operation on ds
  ds = ds->Map({lookup}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  std::vector<std::vector<float>> expected = {{0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411},
                                              {-1, -1, -1, -1, -1, -1},
                                              {0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973},
                                              {0.70853, 0.57088, -0.4716, 0.18048, 0.54449, 0.72603},
                                              {0.68047, -0.039263, 0.30186, -0.17792, 0.42962, 0.032246},
                                              {0.26818, 0.14346, -0.27877, 0.016257, 0.11384, 0.69923},
                                              {-1, -1, -1, -1, -1, -1}};
  while (row.size() != 0) {
    auto ind = row["text"];
    MS_LOG(INFO) << ind.Shape();
    TEST_MS_LOG_MSTENSOR(INFO, "ind: ", ind);
    TensorPtr de_expected_item;
    dsize_t dim = 6;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], TensorShape({dim}), &de_expected_item));
    mindspore::MSTensor ms_expected_item =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_item));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_item);

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 7);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Vectors
/// Description: Test with the parameter max_vectors that is <= 0
/// Expectation: Throw correct error and message
TEST_F(MindDataTestPipeline, TestVectorsMaxVectorsLessThanZero) {
  // Test with max_vectors <= 0.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVectorsMaxVectorsLessThanZero.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testVectors/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/testVectors/vectors.txt";
  std::shared_ptr<Vectors> vectors;
  Status s = Vectors::BuildFromFile(&vectors, vectors_dir, -1);
  EXPECT_NE(s, Status::OK());
}

/// Feature: Vectors
/// Description: Test with the pre-vectors file that is empty
/// Expectation: Throw correct error and message
TEST_F(MindDataTestPipeline, TestVectorsWithEmptyFile) {
  // Read empty file.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVectorsWithEmptyFile.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testVectors/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/testVectors/vectors_empty.txt";
  std::shared_ptr<Vectors> vectors;
  Status s = Vectors::BuildFromFile(&vectors, vectors_dir);
  EXPECT_NE(s, Status::OK());
}

/// Feature: Vectors
/// Description: Test with the pre-vectors file that is not exist
/// Expectation: Throw correct error and message
TEST_F(MindDataTestPipeline, TestVectorsWithNotExistFile) {
  // Test with not exist file.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVectorsWithNotExistFile.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testVectors/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/testVectors/no_vectors.txt";
  std::shared_ptr<Vectors> vectors;
  Status s = Vectors::BuildFromFile(&vectors, vectors_dir);
  EXPECT_NE(s, Status::OK());
}

/// Feature: Vectors
/// Description: Test with the pre-vectors set that has a situation that info-head is not the first line in the set
/// Expectation: Throw correct error and message
TEST_F(MindDataTestPipeline, TestVectorsWithWrongInfoFile) {
  // Wrong info.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVectorsWithWrongInfoFile.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testVectors/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/testVectors/vectors_with_wrong_info.txt";
  std::shared_ptr<Vectors> vectors;
  Status s = Vectors::BuildFromFile(&vectors, vectors_dir);
  EXPECT_NE(s, Status::OK());
}

/// Feature: FastText
/// Description: Test with default parameter in function BuildFromFile and function Lookup
/// Expectation: Return correct MSTensor which is equal to the expected
TEST_F(MindDataTestPipeline, TestFastTextDefaultParam) {
  // Test with default parameter.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFastTextDefaultParam.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/test_fast_text/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/test_fast_text/fast_text.vec";
  std::shared_ptr<FastText> fast_text;
  Status s = FastText::BuildFromFile(&fast_text, vectors_dir);
  EXPECT_EQ(s, Status::OK());

  std::shared_ptr<TensorTransform> lookup = std::make_shared<text::ToVectors>(fast_text);
  EXPECT_NE(lookup, nullptr);

  // Create Map operation on ds
  ds = ds->Map({lookup}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  std::vector<std::vector<float>> expected = {{0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411},
                                              {0, 0, 0, 0, 0, 0},
                                              {0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973},
                                              {0.70853, 0.57088, -0.4716, 0.18048, 0.54449, 0.72603},
                                              {0.68047, -0.039263, 0.30186, -0.17792, 0.42962, 0.032246},
                                              {0.26818, 0.14346, -0.27877, 0.016257, 0.11384, 0.69923},
                                              {0, 0, 0, 0, 0, 0}};
  while (row.size() != 0) {
    auto ind = row["text"];
    MS_LOG(INFO) << ind.Shape();
    TEST_MS_LOG_MSTENSOR(INFO, "ind: ", ind);
    TensorPtr de_expected_item;
    dsize_t dim = 6;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], TensorShape({dim}), &de_expected_item));
    mindspore::MSTensor ms_expected_item =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_item));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_item);

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 7);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: FastText
/// Description: Test with all parameters which include `path` and `max_vector` in function BuildFromFile
/// Expectation: Return correct MSTensor which is equal to the expected
TEST_F(MindDataTestPipeline, TestFastTextAllBuildfromfileParams) {
  // Test with two parameters.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFastTextAllBuildfromfileParams.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/test_fast_text/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/test_fast_text/fast_text.vec";
  std::shared_ptr<FastText> fast_text;
  Status s = FastText::BuildFromFile(&fast_text, vectors_dir, 100);
  EXPECT_EQ(s, Status::OK());

  std::shared_ptr<TensorTransform> lookup = std::make_shared<text::ToVectors>(fast_text);
  EXPECT_NE(lookup, nullptr);

  // Create Map operation on ds
  ds = ds->Map({lookup}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  std::vector<std::vector<float>> expected = {{0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411},
                                              {0, 0, 0, 0, 0, 0},
                                              {0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973},
                                              {0.70853, 0.57088, -0.4716, 0.18048, 0.54449, 0.72603},
                                              {0.68047, -0.039263, 0.30186, -0.17792, 0.42962, 0.032246},
                                              {0.26818, 0.14346, -0.27877, 0.016257, 0.11384, 0.69923},
                                              {0, 0, 0, 0, 0, 0}};
  while (row.size() != 0) {
    auto ind = row["text"];
    MS_LOG(INFO) << ind.Shape();
    TEST_MS_LOG_MSTENSOR(INFO, "ind: ", ind);
    TensorPtr de_expected_item;
    dsize_t dim = 6;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], TensorShape({dim}), &de_expected_item));
    mindspore::MSTensor ms_expected_item =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_item));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_item);

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 7);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: FastText
/// Description: Test with all parameters in function BuildFromFile and `unknown_init` in function Lookup
/// Expectation: Return correct MSTensor which is equal to the expected
TEST_F(MindDataTestPipeline, TestFastTextUnknownInit) {
  // Test with two parameters.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFastTextUnknownInit.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/test_fast_text/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/test_fast_text/fast_text.vec";
  std::shared_ptr<FastText> fast_text;
  Status s = FastText::BuildFromFile(&fast_text, vectors_dir, 100);
  EXPECT_EQ(s, Status::OK());

  std::vector<float> unknown_init = {-1, -1, -1, -1, -1, -1};
  std::shared_ptr<TensorTransform> lookup = std::make_shared<text::ToVectors>(fast_text, unknown_init);
  EXPECT_NE(lookup, nullptr);

  // Create Map operation on ds
  ds = ds->Map({lookup}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  std::vector<std::vector<float>> expected = {{0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411},
                                              {-1, -1, -1, -1, -1, -1},
                                              {0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973},
                                              {0.70853, 0.57088, -0.4716, 0.18048, 0.54449, 0.72603},
                                              {0.68047, -0.039263, 0.30186, -0.17792, 0.42962, 0.032246},
                                              {0.26818, 0.14346, -0.27877, 0.016257, 0.11384, 0.69923},
                                              {-1, -1, -1, -1, -1, -1}};
  while (row.size() != 0) {
    auto ind = row["text"];
    MS_LOG(INFO) << ind.Shape();
    TEST_MS_LOG_MSTENSOR(INFO, "ind: ", ind);
    TensorPtr de_expected_item;
    dsize_t dim = 6;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], TensorShape({dim}), &de_expected_item));
    mindspore::MSTensor ms_expected_item =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_item));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_item);

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 7);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: FastText
/// Description: Test with all parameters which include `path` and `max_vectors` in function BuildFromFile and `token`,
///     `unknown_init` and `lower_case_backup` in function Lookup. But some tokens have some big letters
/// Expectation: Return correct MSTensor which is equal to the expected
TEST_F(MindDataTestPipeline, TestFastTextAllParams) {
  // Test with all parameters.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFastTextAllParams.";
  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/test_fast_text/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/test_fast_text/fast_text.vec";
  std::shared_ptr<FastText> fast_text;
  Status s = FastText::BuildFromFile(&fast_text, vectors_dir);
  EXPECT_EQ(s, Status::OK());

  std::vector<float> unknown_init = {-1, -1, -1, -1, -1, -1};
  std::shared_ptr<TensorTransform> lookup = std::make_shared<text::ToVectors>(fast_text, unknown_init, true);
  EXPECT_NE(lookup, nullptr);

  // Create Map operation on ds
  ds = ds->Map({lookup}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  std::vector<std::vector<float>> expected = {{0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411},
                                              {-1, -1, -1, -1, -1, -1},
                                              {0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973},
                                              {0.70853, 0.57088, -0.4716, 0.18048, 0.54449, 0.72603},
                                              {0.68047, -0.039263, 0.30186, -0.17792, 0.42962, 0.032246},
                                              {0.26818, 0.14346, -0.27877, 0.016257, 0.11384, 0.69923},
                                              {-1, -1, -1, -1, -1, -1}};
  while (row.size() != 0) {
    auto ind = row["text"];
    MS_LOG(INFO) << ind.Shape();
    TEST_MS_LOG_MSTENSOR(INFO, "ind: ", ind);
    TensorPtr de_expected_item;
    dsize_t dim = 6;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], TensorShape({dim}), &de_expected_item));
    mindspore::MSTensor ms_expected_item =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_item));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_item);

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 7);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: FastText
/// Description: Test with pre-vectors set that have the different dimension
/// Expectation: Throw correct error and message
TEST_F(MindDataTestPipeline, TestFastTextDifferentDimension) {
  // Tokens don't have the same number of vectors.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFastTextDifferentDimension.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/test_fast_text/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/test_fast_text/fasttext_dim_different.vec";
  std::shared_ptr<FastText> fast_text;
  Status s = FastText::BuildFromFile(&fast_text, vectors_dir, 100);
  EXPECT_NE(s, Status::OK());
}

/// Feature: FastText
/// Description: Test with the parameter max_vectors that is <= 0
/// Expectation: Throw correct error and message
TEST_F(MindDataTestPipeline, TestFastTextMaxVectorsLessThanZero) {
  // Test with max_vectors <= 0.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFastTextMaxVectorsLessThanZero.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/test_fast_text/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/test_fast_text/fast_text.vec";
  std::shared_ptr<FastText> fast_text;
  Status s = FastText::BuildFromFile(&fast_text, vectors_dir, -1);
  EXPECT_NE(s, Status::OK());
}

/// Feature: FastText
/// Description: Test with the pre-vectors file that is empty
/// Expectation: Throw correct error and message
TEST_F(MindDataTestPipeline, TestFastTextWithEmptyFile) {
  // Read empty file.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFastTextWithEmptyFile.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/test_fast_text/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/test_fast_text/fasttext_empty.vec";
  std::shared_ptr<FastText> fast_text;
  Status s = FastText::BuildFromFile(&fast_text, vectors_dir);
  EXPECT_NE(s, Status::OK());
}

/// Feature: FastText
/// Description: Test with the pre-vectors file that is not exist
/// Expectation: Throw correct error and message
TEST_F(MindDataTestPipeline, TestFastTextWithNotExistFile) {
  // Test with not exist file.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFastTextWithNotExistFile.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/test_fast_text/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/test_fast_text/no_fasttext.vec";
  std::shared_ptr<FastText> fast_text;
  Status s = FastText::BuildFromFile(&fast_text, vectors_dir);
  EXPECT_NE(s, Status::OK());
}

/// Feature: FastText
/// Description: Test with the pre-vectors set that has a situation that info-head is not the first line in the set
/// Expectation: Throw correct error and message
TEST_F(MindDataTestPipeline, TestFastTextWithWrongInfoFile) {
  // Wrong info.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFastTextWithWrongInfoFile.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/test_fast_text/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/test_fast_text/fasttext_with_wrong_info.vec";
  std::shared_ptr<FastText> fast_text;
  Status s = FastText::BuildFromFile(&fast_text, vectors_dir);
  EXPECT_NE(s, Status::OK());
}

/// Feature: FastText
/// Description: Test with the pre-vectors set that has a wrong suffix
/// Expectation: Throw correct error and message
TEST_F(MindDataTestPipeline, TestFastTextWithWrongSuffix) {
  // Wrong info.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFastTextWithWrongSuffix.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/test_fast_text/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/test_fast_text/fast_text.txt";
  std::shared_ptr<FastText> fast_text;
  Status s = FastText::BuildFromFile(&fast_text, vectors_dir);
  EXPECT_NE(s, Status::OK());
}

/// Feature: GloVe
/// Description: Test with default parameter in function BuildFromFile and function Lookup
/// Expectation: Return correct MSTensor which is equal to the expected
TEST_F(MindDataTestPipeline, TestGloVeDefaultParam) {
  // Test with default parameter.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGloVeDefaultParam.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testGloVe/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/testGloVe/glove.6B.test.txt";
  std::shared_ptr<GloVe> glove;
  Status s = GloVe::BuildFromFile(&glove, vectors_dir);
  EXPECT_EQ(s, Status::OK());

  std::shared_ptr<TensorTransform> lookup = std::make_shared<text::ToVectors>(glove);
  EXPECT_NE(lookup, nullptr);

  // Create Map operation on ds
  ds = ds->Map({lookup}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  std::vector<std::vector<float>> expected = {{0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411},
                                              {0, 0, 0, 0, 0, 0},
                                              {0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973},
                                              {0.70853, 0.57088, -0.4716, 0.18048, 0.54449, 0.72603},
                                              {0.68047, -0.039263, 0.30186, -0.17792, 0.42962, 0.032246},
                                              {0.26818, 0.14346, -0.27877, 0.016257, 0.11384, 0.69923},
                                              {0, 0, 0, 0, 0, 0}};
  while (row.size() != 0) {
    auto ind = row["text"];
    MS_LOG(INFO) << ind.Shape();
    TEST_MS_LOG_MSTENSOR(INFO, "ind: ", ind);
    TensorPtr de_expected_item;
    dsize_t dim = 6;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], TensorShape({dim}), &de_expected_item));
    mindspore::MSTensor ms_expected_item =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_item));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_item);

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 7);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: GloVe
/// Description: Test with all parameters which include `path` and `max_vector` in function BuildFromFile
/// Expectation: Return correct MSTensor which is equal to the expected
TEST_F(MindDataTestPipeline, TestGloVeAllBuildfromfileParams) {
  // Test with two parameters.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGloVeAllBuildfromfileParams.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testGloVe/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/testGloVe/glove.6B.test.txt";
  std::shared_ptr<GloVe> glove;
  Status s = GloVe::BuildFromFile(&glove, vectors_dir, 100);
  EXPECT_EQ(s, Status::OK());

  std::shared_ptr<TensorTransform> lookup = std::make_shared<text::ToVectors>(glove);
  EXPECT_NE(lookup, nullptr);

  // Create Map operation on ds
  ds = ds->Map({lookup}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  std::vector<std::vector<float>> expected = {{0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411},
                                              {0, 0, 0, 0, 0, 0},
                                              {0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973},
                                              {0.70853, 0.57088, -0.4716, 0.18048, 0.54449, 0.72603},
                                              {0.68047, -0.039263, 0.30186, -0.17792, 0.42962, 0.032246},
                                              {0.26818, 0.14346, -0.27877, 0.016257, 0.11384, 0.69923},
                                              {0, 0, 0, 0, 0, 0}};
  while (row.size() != 0) {
    auto ind = row["text"];
    MS_LOG(INFO) << ind.Shape();
    TEST_MS_LOG_MSTENSOR(INFO, "ind: ", ind);
    TensorPtr de_expected_item;
    dsize_t dim = 6;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], TensorShape({dim}), &de_expected_item));
    mindspore::MSTensor ms_expected_item =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_item));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_item);

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 7);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: GloVe
/// Description: Test with all parameters in function BuildFromFile and `unknown_init` in function Lookup
/// Expectation: Return correct MSTensor which is equal to the expected
TEST_F(MindDataTestPipeline, TestGloVeUnknownInit) {
  // Test with two parameters.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGloVeUnknownInit.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testGloVe/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/testGloVe/glove.6B.test.txt";
  std::shared_ptr<GloVe> glove;
  Status s = GloVe::BuildFromFile(&glove, vectors_dir, 100);
  EXPECT_EQ(s, Status::OK());

  std::vector<float> unknown_init = {-1, -1, -1, -1, -1, -1};
  std::shared_ptr<TensorTransform> lookup = std::make_shared<text::ToVectors>(glove, unknown_init);
  EXPECT_NE(lookup, nullptr);

  // Create Map operation on ds
  ds = ds->Map({lookup}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  std::vector<std::vector<float>> expected = {{0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411},
                                              {-1, -1, -1, -1, -1, -1},
                                              {0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973},
                                              {0.70853, 0.57088, -0.4716, 0.18048, 0.54449, 0.72603},
                                              {0.68047, -0.039263, 0.30186, -0.17792, 0.42962, 0.032246},
                                              {0.26818, 0.14346, -0.27877, 0.016257, 0.11384, 0.69923},
                                              {-1, -1, -1, -1, -1, -1}};
  while (row.size() != 0) {
    auto ind = row["text"];
    MS_LOG(INFO) << ind.Shape();
    TEST_MS_LOG_MSTENSOR(INFO, "ind: ", ind);
    TensorPtr de_expected_item;
    dsize_t dim = 6;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], TensorShape({dim}), &de_expected_item));
    mindspore::MSTensor ms_expected_item =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_item));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_item);

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 7);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: GloVe
/// Description: Test with all parameters which include `path` and `max_vectors` in function BuildFromFile and `token`,
///     `unknown_init` and `lower_case_backup` in function Lookup. But some tokens have some big letters
/// Expectation: Return correct MSTensor which is equal to the expected
TEST_F(MindDataTestPipeline, TestGloVeAllParams) {
  // Test with all parameters.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGloVeAllParams.";
  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testGloVe/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/testGloVe/glove.6B.test.txt";
  std::shared_ptr<GloVe> glove;
  Status s = GloVe::BuildFromFile(&glove, vectors_dir);
  EXPECT_EQ(s, Status::OK());

  std::vector<float> unknown_init = {-1, -1, -1, -1, -1, -1};
  std::shared_ptr<TensorTransform> lookup = std::make_shared<text::ToVectors>(glove, unknown_init, true);
  EXPECT_NE(lookup, nullptr);

  // Create Map operation on ds
  ds = ds->Map({lookup}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  std::vector<std::vector<float>> expected = {{0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411},
                                              {-1, -1, -1, -1, -1, -1},
                                              {0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973},
                                              {0.70853, 0.57088, -0.4716, 0.18048, 0.54449, 0.72603},
                                              {0.68047, -0.039263, 0.30186, -0.17792, 0.42962, 0.032246},
                                              {0.26818, 0.14346, -0.27877, 0.016257, 0.11384, 0.69923},
                                              {-1, -1, -1, -1, -1, -1}};
  while (row.size() != 0) {
    auto ind = row["text"];
    MS_LOG(INFO) << ind.Shape();
    TEST_MS_LOG_MSTENSOR(INFO, "ind: ", ind);
    TensorPtr de_expected_item;
    dsize_t dim = 6;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], TensorShape({dim}), &de_expected_item));
    mindspore::MSTensor ms_expected_item =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_item));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_item);

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 7);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: GloVe
/// Description: Test with pre-vectors set that have the different dimension
/// Expectation: Throw correct error and message
TEST_F(MindDataTestPipeline, TestGloVeDifferentDimension) {
  // Tokens don't have the same number of glove.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGloVeDifferentDimension.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testGloVe/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/testGloVe/glove.6B.dim_different.txt";
  std::shared_ptr<GloVe> glove;
  Status s = GloVe::BuildFromFile(&glove, vectors_dir, 100);
  EXPECT_NE(s, Status::OK());
}

/// Feature: GloVe
/// Description: Test with the parameter max_vectors that is <= 0
/// Expectation: Throw correct error and message
TEST_F(MindDataTestPipeline, TestGloVeMaxVectorsLessThanZero) {
  // Test with max_vectors <= 0.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGloVeMaxVectorsLessThanZero.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testGloVe/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/testGloVe/glove.6B.test.txt";
  std::shared_ptr<GloVe> glove;
  Status s = GloVe::BuildFromFile(&glove, vectors_dir, -1);
  EXPECT_NE(s, Status::OK());
}

/// Feature: GloVe
/// Description: Test with the pre-vectors file that is empty
/// Expectation: Throw correct error and message
TEST_F(MindDataTestPipeline, TestGloVeWithEmptyFile) {
  // Read empty file.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGloVeWithEmptyFile.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testGloVe/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/testGloVe/glove.6B.empty.txt";
  std::shared_ptr<GloVe> glove;
  Status s = GloVe::BuildFromFile(&glove, vectors_dir);
  EXPECT_NE(s, Status::OK());
}

/// Feature: GloVe
/// Description: Test with the pre-vectors file that is not exist
/// Expectation: Throw correct error and message
TEST_F(MindDataTestPipeline, TestGloVeWithNotExistFile) {
  // Test with not exist file.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGloVeWithNotExistFile.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testGloVe/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/testGloVe/glove.6B.empty.txt";
  std::shared_ptr<GloVe> glove;
  Status s = GloVe::BuildFromFile(&glove, vectors_dir);
  EXPECT_NE(s, Status::OK());
}

/// Feature: GloVe
/// Description: Test with the pre-vectors set that has a situation that info-head is not the first line in the set
/// Expectation: Throw correct error and message
TEST_F(MindDataTestPipeline, TestGloVeWithWrongInfoFile) {
  // Wrong info.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGloVeWithWrongInfoFile.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testGloVe/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/testGloVe/glove.6B.with_wrong_info.txt";
  std::shared_ptr<GloVe> glove;
  Status s = GloVe::BuildFromFile(&glove, vectors_dir);
  EXPECT_NE(s, Status::OK());
}

/// Feature: GloVe
/// Description: Test with the pre-vectors set that has a wrong format
/// Expectation: Throw correct error and message
TEST_F(MindDataTestPipeline, TestGloVeWithWrongFormat) {
  // Wrong info.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGloVeWithWrongFormat.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testGloVe/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/testGloVe/glove.6B.tests.vec";
  std::shared_ptr<GloVe> glove;
  Status s = GloVe::BuildFromFile(&glove, vectors_dir);
  EXPECT_NE(s, Status::OK());
}

/// Feature: CharNGram
/// Description: Test with default parameter in function BuildFromFile and function Lookup
/// Expectation: Return correct MSTensor which is equal to the excepted
TEST_F(MindDataTestPipeline, TestCharNGramDefaultParam) {
  // Test with default parameter.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCharNGramDefaultParam.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testVectors/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/testVectors/char_n_gram_20.txt";
  std::shared_ptr<CharNGram> char_n_gram;
  Status s = CharNGram::BuildFromFile(&char_n_gram, vectors_dir);
  EXPECT_EQ(s, Status::OK());
  std::shared_ptr<TensorTransform> lookup = std::make_shared<text::ToVectors>(char_n_gram);
  EXPECT_NE(lookup, nullptr);

  // Create Map operation on ds
  ds = ds->Map({lookup}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  std::vector<std::vector<float>> expected = {{0, 0, 0, 0, 0},
                                              {0, 0, 0, 0, 0},
                                              {0.117336, 0.362446, -0.983326, 0.939264, -0.05648},
                                              {0.657201, 2.11761, -1.59276, 0.432072, 1.21395},
                                              {0, 0, 0, 0, 0},
                                              {-2.26956, 0.288491, -0.740001, 0.661703, 0.147355},
                                              {0, 0, 0, 0, 0}};
  while (row.size() != 0) {
    auto ind = row["text"];
    MS_LOG(INFO) << ind.Shape();
    TEST_MS_LOG_MSTENSOR(INFO, "ind: ", ind);
    TensorPtr de_expected_item;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_item));
    mindspore::MSTensor ms_expected_item =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_item));
    std::vector<int64_t> ind_shape = ind.Shape();
    std::vector<int64_t> ms_expected_shape = ms_expected_item.Shape();
    EXPECT_EQ(ind_shape, ms_expected_shape);

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 7);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: CharNGram.
/// Description: Test with all parameters which include `path` and `max_vector` in function BuildFromFile
/// Expectation: Return correct MSTensor which is equal to the excepted
TEST_F(MindDataTestPipeline, TestCharNGramAllBuildfromfileParams) {
  // Test with two parameters.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCharNGramAllBuildfromfileParams.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testVectors/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/testVectors/char_n_gram_20.txt";
  std::shared_ptr<CharNGram> char_n_gram;
  Status s = CharNGram::BuildFromFile(&char_n_gram, vectors_dir, 18);
  EXPECT_EQ(s, Status::OK());

  std::shared_ptr<TensorTransform> lookup = std::make_shared<text::ToVectors>(char_n_gram);
  EXPECT_NE(lookup, nullptr);

  // Create Map operation on ds
  ds = ds->Map({lookup}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  std::vector<std::vector<float>> expected = {{0, 0, 0, 0, 0},
                                              {0, 0, 0, 0, 0},
                                              {-0.155665, 0.664073, -0.538499, 1.22657, -0.2162},
                                              {0.657201, 2.11761, -1.59276, 0.432072, 1.21395},
                                              {0, 0, 0, 0, 0},
                                              {-2.26956, 0.288491, -0.740001, 0.661703, 0.147355},
                                              {0, 0, 0, 0, 0}};
  while (row.size() != 0) {
    auto ind = row["text"];
    MS_LOG(INFO) << ind.Shape();
    TEST_MS_LOG_MSTENSOR(INFO, "ind: ", ind);
    TensorPtr de_expected_item;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_item));
    mindspore::MSTensor ms_expected_item =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_item));
    std::vector<int64_t> ind_shape = ind.Shape();
    std::vector<int64_t> ms_expected_shape = ms_expected_item.Shape();
    EXPECT_EQ(ind_shape, ms_expected_shape);

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 7);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: CharNGram
/// Description: Test with all parameters in function BuildFromFile and `unknown_init` in function Lookup
/// Expectation: Return correct MSTensor which is equal to the excepted
TEST_F(MindDataTestPipeline, TestCharNGramUnknownInit) {
  // Test with two parameters.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCharNGramUnknownInit.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testVectors/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/testVectors/char_n_gram_20.txt";
  std::shared_ptr<CharNGram> char_n_gram;
  Status s = CharNGram::BuildFromFile(&char_n_gram, vectors_dir, 18);
  EXPECT_EQ(s, Status::OK());

  std::vector<float> unknown_init(5, -1);
  std::shared_ptr<TensorTransform> lookup = std::make_shared<text::ToVectors>(char_n_gram, unknown_init);
  EXPECT_NE(lookup, nullptr);

  // Create Map operation on ds
  ds = ds->Map({lookup}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  std::vector<std::vector<float>> expected = {{-1, -1, -1, -1, -1},
                                              {-1, -1, -1, -1, -1},
                                              {-0.155665, 0.664073, -0.538499, 1.22657, -0.2162},
                                              {0.657201, 2.11761, -1.59276, 0.432072, 1.21395},
                                              {-1, -1, -1, -1, -1},
                                              {-2.26956, 0.288491, -0.740001, 0.661703, 0.147355},
                                              {-1, -1, -1, -1, -1}};
  while (row.size() != 0) {
    auto ind = row["text"];
    MS_LOG(INFO) << ind.Shape();
    TEST_MS_LOG_MSTENSOR(INFO, "ind: ", ind);
    TensorPtr de_expected_item;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_item));
    mindspore::MSTensor ms_expected_item =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_item));
    std::vector<int64_t> ind_shape = ind.Shape();
    std::vector<int64_t> ms_expected_shape = ms_expected_item.Shape();
    EXPECT_EQ(ind_shape, ms_expected_shape);

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 7);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: CharNGram
/// Description: Test with all parameters which include `path` and `max_vectors` in function BuildFromFile and `token`,
///     `unknown_init` and `lower_case_backup` in function Lookup. But some tokens have some big letters
/// Expectation: Return correct MSTensor which is equal to the excepted
TEST_F(MindDataTestPipeline, TestCharNGramAllParams) {
  // Test with all parameters.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCharNGramAllParams.";
  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testVectors/words_with_big_letter.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/testVectors/char_n_gram_20.txt";
  std::shared_ptr<CharNGram> char_n_gram;
  Status s = CharNGram::BuildFromFile(&char_n_gram, vectors_dir);
  EXPECT_EQ(s, Status::OK());

  std::vector<float> unknown_init(5, -1);
  std::shared_ptr<TensorTransform> lookup = std::make_shared<text::ToVectors>(char_n_gram, unknown_init, true);
  EXPECT_NE(lookup, nullptr);

  // Create Map operation on ds
  ds = ds->Map({lookup}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  std::vector<std::vector<float>> expected = {{-1, -1, -1, -1, -1},
                                              {-1, -1, -1, -1, -1},
                                              {0.117336, 0.362446, -0.983326, 0.939264, -0.05648},
                                              {0.657201, 2.11761, -1.59276, 0.432072, 1.21395},
                                              {-1, -1, -1, -1, -1},
                                              {-2.26956, 0.288491, -0.740001, 0.661703, 0.147355},
                                              {-1, -1, -1, -1, -1}};
  while (row.size() != 0) {
    auto ind = row["text"];
    MS_LOG(INFO) << ind.Shape();
    TEST_MS_LOG_MSTENSOR(INFO, "ind: ", ind);
    TensorPtr de_expected_item;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_item));
    mindspore::MSTensor ms_expected_item =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_item));
    std::vector<int64_t> ind_shape = ind.Shape();
    std::vector<int64_t> ms_expected_shape = ms_expected_item.Shape();
    EXPECT_EQ(ind_shape, ms_expected_shape);

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 7);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: CharNGram
/// Description: Test with pre-vectors set that have the different dimension
/// Expectation: Throw correct error and message
TEST_F(MindDataTestPipeline, TestCharNGramDifferentDimension) {
  // Tokens don't have the same number of vectors.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCharNGramDifferentDimension.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testVectors/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/testVectors/char_n_gram_20_dim_different.txt";
  std::shared_ptr<CharNGram> char_n_gram;
  Status s = CharNGram::BuildFromFile(&char_n_gram, vectors_dir);
  EXPECT_NE(s, Status::OK());
}

/// Feature: CharNGram
/// Description: Test with the parameter max_vectors that is <= 0
/// Expectation: Throw correct error and message
TEST_F(MindDataTestPipeline, TestCharNGramMaxVectorsLessThanZero) {
  // Test with max_vectors <= 0.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCharNGramMaxVectorsLessThanZero.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testVectors/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/testVectors/char_n_gram_20.txt";
  std::shared_ptr<CharNGram> char_n_gram;
  Status s = CharNGram::BuildFromFile(&char_n_gram, vectors_dir, -1);
  EXPECT_NE(s, Status::OK());
}

/// Feature: CharNGram
/// Description: Test with the pre-vectors file that is empty
/// Expectation: Throw correct error and message
TEST_F(MindDataTestPipeline, TestCharNGramWithEmptyFile) {
  // Read empty file.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCharNGramWithEmptyFile.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testVectors/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/testVectors/vectors_empty.txt";
  std::shared_ptr<CharNGram> char_n_gram;
  Status s = CharNGram::BuildFromFile(&char_n_gram, vectors_dir);
  EXPECT_NE(s, Status::OK());
}

/// Feature: CharNGram
/// Description: Test with the pre-vectors file that is not exist
/// Expectation: Throw correct error and message
TEST_F(MindDataTestPipeline, TestCharNGramsWithNotExistFile) {
  // Test with not exist file.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCharNGramsWithNotExistFile.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testVectors/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::string vectors_dir = datasets_root_path_ + "/testVectors/no_vectors.txt";
  std::shared_ptr<CharNGram> char_n_gram;
  Status s = CharNGram::BuildFromFile(&char_n_gram, vectors_dir);
  EXPECT_NE(s, Status::OK());
}

/// Feature: AddToken op
/// Description: Test input 1d of AddToken op successfully
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestAddTokenPipelineSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAddTokenPipelineSuccess.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create Take operation on ds
  ds = ds->Take(1);
  EXPECT_NE(ds, nullptr);

  // Create white_tokenizer operation on ds
  std::shared_ptr<TensorTransform> white_tokenizer = std::make_shared<text::WhitespaceTokenizer>();
  EXPECT_NE(white_tokenizer, nullptr);

  // Create add_token operation on ds
  std::shared_ptr<TensorTransform> add_token = std::make_shared<text::AddToken>("TOKEN", true);
  EXPECT_NE(add_token, nullptr);

  // Create Map operation on ds
  ds = ds->Map({white_tokenizer, add_token}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::string> expected = {"TOKEN", "This", "is", "a", "text", "file."};
  std::shared_ptr<Tensor> de_expected_tensor;
  ASSERT_OK(Tensor::CreateFromVector(expected, &de_expected_tensor));
  mindspore::MSTensor expected_tensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Truncate
/// Description: Test Truncate basic usage max_seq_len less length
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestTruncateSuccess1D) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTruncateSuccess1D.";
  // Testing basic Truncate

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create white_tokenizer operation on ds
  std::shared_ptr<TensorTransform> white_tokenizer = std::make_shared<text::WhitespaceTokenizer>();
  EXPECT_NE(white_tokenizer, nullptr);

  // Create a truncate operation on ds
  std::shared_ptr<TensorTransform> truncate = std::make_shared<text::Truncate>(3);
  EXPECT_NE(truncate, nullptr);

  // Create Map operation on ds
  ds = ds->Map({white_tokenizer, truncate}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::vector<std::string>> expected = {
    {"This", "is", "a"}, {"Be", "happy", "every"}, {"Good", "luck", "to"}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];

    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Truncate
/// Description: Test the incorrect parameter of Truncate interface
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestTruncateFail) {
  // Testing the incorrect parameter of Truncate interface.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTruncateFail.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Testing the parameter max_seq_len less than 0
  // Create a truncate operation on ds
  std::shared_ptr<TensorTransform> truncate = std::make_shared<text::Truncate>(-1);
  EXPECT_NE(truncate, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({truncate});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Truncate input (The parameter max_seq_len must be greater than  0)
  EXPECT_EQ(iter, nullptr);
}
