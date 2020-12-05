/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/include/datasets.h"
#include "minddata/dataset/include/status.h"
#include "minddata/dataset/include/transforms.h"
#include "minddata/dataset/include/text.h"

using namespace mindspore::dataset;
using mindspore::dataset::DataType;
using mindspore::dataset::ShuffleMode;
using mindspore::dataset::Status;
using mindspore::dataset::Tensor;
using mindspore::dataset::Vocab;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestPipeline, TestCaseFoldSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCaseFoldSuccess.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create casefold operation on ds
  std::shared_ptr<TensorOperation> casefold = text::CaseFold();
  EXPECT_NE(casefold, nullptr);

  // Create Map operation on ds
  ds = ds->Map({casefold}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  std::vector<std::string> expected = {"welcome to beijing!", "北京欢迎您!", "我喜欢english!", "  "};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> expected_tensor;
    Tensor::CreateScalar(expected[i], &expected_tensor);
    EXPECT_EQ(*ind, *expected_tensor);
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
  std::shared_ptr<TensorOperation> jieba_tokenizer = text::JiebaTokenizer(hmm_path, mp_path, JiebaMode::kMp);
  EXPECT_NE(jieba_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({jieba_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  std::vector<std::string> expected = {"今天天气", "太好了", "我们", "一起", "去", "外面", "玩吧"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> expected_tensor;
    Tensor::CreateFromVector(expected, &expected_tensor);
    EXPECT_EQ(*ind, *expected_tensor);
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
  std::shared_ptr<TensorOperation> jieba_tokenizer = text::JiebaTokenizer(hmm_path, mp_path, JiebaMode::kHmm);
  EXPECT_NE(jieba_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({jieba_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  std::vector<std::string> expected = {"今天", "天气", "太", "好", "了", "我们", "一起", "去", "外面", "玩", "吧"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> expected_tensor;
    Tensor::CreateFromVector(expected, &expected_tensor);
    EXPECT_EQ(*ind, *expected_tensor);
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
  std::shared_ptr<TensorOperation> jieba_tokenizer = text::JiebaTokenizer(hmm_path, mp_path, JiebaMode::kMp, true);
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
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  std::vector<std::string> expected = {"今天天气", "太好了", "我们", "一起", "去", "外面", "玩吧"};

  std::vector<uint32_t> expected_offsets_start = {0, 12, 21, 27, 33, 36, 42};
  std::vector<uint32_t> expected_offsets_limit = {12, 21, 27, 33, 36, 42, 48};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["offsets_start"];
    auto ind1 = row["offsets_limit"];
    auto token = row["token"];
    std::shared_ptr<Tensor> expected_tensor;
    std::shared_ptr<Tensor> expected_tensor_offsets_start;
    std::shared_ptr<Tensor> expected_tensor_offsets_limit;
    Tensor::CreateFromVector(expected, &expected_tensor);
    Tensor::CreateFromVector(expected_offsets_start, &expected_tensor_offsets_start);
    Tensor::CreateFromVector(expected_offsets_limit, &expected_tensor_offsets_limit);
    EXPECT_EQ(*ind, *expected_tensor_offsets_start);
    EXPECT_EQ(*ind1, *expected_tensor_offsets_limit);
    EXPECT_EQ(*token, *expected_tensor);
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestJiebaTokenizerFail) {
  // Testing the incorrect parameter of JiebaTokenizer interface.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestJiebaTokenizerFail.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testJiebaDataset/3.txt";
  std::string hmm_path = datasets_root_path_ + "/jiebadict/hmm_model.utf8";
  std::string mp_path = datasets_root_path_ + "/jiebadict/jieba.dict.utf8";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create jieba_tokenizer operation on ds
  // Testing the parameter hmm_path is empty
  std::shared_ptr<TensorOperation> jieba_tokenizer = text::JiebaTokenizer("", mp_path, JiebaMode::kMp);
  EXPECT_EQ(jieba_tokenizer, nullptr);
  // Testing the parameter mp_path is empty
  std::shared_ptr<TensorOperation> jieba_tokenizer1 = text::JiebaTokenizer(hmm_path, "", JiebaMode::kMp);
  EXPECT_EQ(jieba_tokenizer1, nullptr);
  // Testing the parameter hmm_path is invalid path
  std::string hmm_path_invalid = datasets_root_path_ + "/jiebadict/1.txt";
  std::shared_ptr<TensorOperation> jieba_tokenizer2 = text::JiebaTokenizer(hmm_path_invalid, mp_path, JiebaMode::kMp);
  EXPECT_EQ(jieba_tokenizer2, nullptr);
  // Testing the parameter mp_path is invalid path
  std::string mp_path_invalid = datasets_root_path_ + "/jiebadict/1.txt";
  std::shared_ptr<TensorOperation> jieba_tokenizer3 = text::JiebaTokenizer(hmm_path, mp_path_invalid, JiebaMode::kMp);
  EXPECT_EQ(jieba_tokenizer3, nullptr);
}

TEST_F(MindDataTestPipeline, TestSlidingWindowSuccess) {
  // Testing the parameter of SlidingWindow interface when the axis is 0.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSlidingWindowSuccess.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create white_tokenizer operation on ds
  std::shared_ptr<TensorOperation> white_tokenizer = text::WhitespaceTokenizer();
  EXPECT_NE(white_tokenizer, nullptr);
  // Create sliding_window operation on ds
  std::shared_ptr<TensorOperation> sliding_window = text::SlidingWindow(3, 0);
  EXPECT_NE(sliding_window, nullptr);

  // Create Map operation on ds
  ds = ds->Map({white_tokenizer, sliding_window}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  std::vector<std::vector<std::string>> expected = {{"This", "is", "a", "is", "a", "text", "a", "text", "file."},
                                                    {"Be", "happy", "every", "happy", "every", "day."},
                                                    {"Good", "luck", "to", "luck", "to", "everyone."}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> expected_tensor;
    int x = expected[i].size() / 3;
    Tensor::CreateFromVector(expected[i], TensorShape({x, 3}), &expected_tensor);
    EXPECT_EQ(*ind, *expected_tensor);
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
  std::shared_ptr<TensorOperation> white_tokenizer = text::WhitespaceTokenizer();
  EXPECT_NE(white_tokenizer, nullptr);
  // Create sliding_window operation on ds
  std::shared_ptr<TensorOperation> sliding_window = text::SlidingWindow(2, -1);
  EXPECT_NE(sliding_window, nullptr);

  // Create Map operation on ds
  ds = ds->Map({white_tokenizer, sliding_window}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  std::vector<std::vector<std::string>> expected = {{"This", "is", "is", "a", "a", "text", "text", "file."},
                                                    {"Be", "happy", "happy", "every", "every", "day."},
                                                    {"Good", "luck", "luck", "to", "to", "everyone."}};
  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> expected_tensor;
    int x = expected[i].size() / 2;
    Tensor::CreateFromVector(expected[i], TensorShape({x, 2}), &expected_tensor);
    EXPECT_EQ(*ind, *expected_tensor);
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestSlidingWindowFail) {
  // Testing the incorrect parameter of SlidingWindow interface.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSlidingWindowFail.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create sliding_window operation on ds
  // Testing the parameter width less than or equal to 0
  // The parameter axis support 0 or -1 only for now
  std::shared_ptr<TensorOperation> sliding_window = text::SlidingWindow(0, 0);
  EXPECT_EQ(sliding_window, nullptr);
  // Testing the parameter width less than or equal to 0
  // The parameter axis support 0 or -1 only for now
  std::shared_ptr<TensorOperation> sliding_window1 = text::SlidingWindow(-2, 0);
  EXPECT_EQ(sliding_window1, nullptr);
}

TEST_F(MindDataTestPipeline, TestNgramSuccess) {
  // Testing the parameter of Ngram interface.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNgramSuccess.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create white_tokenizer operation on ds
  std::shared_ptr<TensorOperation> white_tokenizer = text::WhitespaceTokenizer();
  EXPECT_NE(white_tokenizer, nullptr);
  // Create sliding_window operation on ds
  std::shared_ptr<TensorOperation> ngram_op = text::Ngram({2}, {"_", 1}, {"_", 1}, " ");
  EXPECT_NE(ngram_op, nullptr);

  // Create Map operation on ds
  ds = ds->Map({white_tokenizer, ngram_op}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  std::vector<std::vector<std::string>> expected = {{"_ This", "This is", "is a", "a text", "text file.", "file. _"},
                                                    {"_ Be", "Be happy", "happy every", "every day.", "day. _"},
                                                    {"_ Good", "Good luck", "luck to", "to everyone.", "everyone. _"}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> expected_tensor;
    int x = expected[i].size();
    Tensor::CreateFromVector(expected[i], TensorShape({x}), &expected_tensor);
    EXPECT_EQ(*ind, *expected_tensor);
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
  std::shared_ptr<TensorOperation> white_tokenizer = text::WhitespaceTokenizer();
  EXPECT_NE(white_tokenizer, nullptr);
  // Create sliding_window operation on ds
  std::shared_ptr<TensorOperation> ngram_op = text::Ngram({2, 3}, {"&", 2}, {"&", 2}, "-");
  EXPECT_NE(ngram_op, nullptr);

  // Create Map operation on ds
  ds = ds->Map({white_tokenizer, ngram_op}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

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
    std::shared_ptr<Tensor> expected_tensor;
    int x = expected[i].size();
    Tensor::CreateFromVector(expected[i], TensorShape({x}), &expected_tensor);
    EXPECT_EQ(*ind, *expected_tensor);
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestNgramFail) {
  // Testing the incorrect parameter of Ngram interface.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNgramFail.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTextFileDataset/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create sliding_window operation on ds
  // Testing the vector of ngram is empty
  std::shared_ptr<TensorOperation> ngram_op = text::Ngram({});
  EXPECT_EQ(ngram_op, nullptr);
  // Testing the value of ngrams vector less than and equal to 0
  std::shared_ptr<TensorOperation> ngram_op1 = text::Ngram({0});
  EXPECT_EQ(ngram_op1, nullptr);
  // Testing the value of ngrams vector less than and equal to 0
  std::shared_ptr<TensorOperation> ngram_op2 = text::Ngram({-2});
  EXPECT_EQ(ngram_op2, nullptr);
  // Testing the second parameter pad_width in left_pad vector less than 0
  std::shared_ptr<TensorOperation> ngram_op3 = text::Ngram({2}, {"", -1});
  EXPECT_EQ(ngram_op3, nullptr);
  // Testing the second parameter pad_width in right_pad vector less than 0
  std::shared_ptr<TensorOperation> ngram_op4 = text::Ngram({2}, {"", 1}, {"", -1});
  EXPECT_EQ(ngram_op4, nullptr);
}

TEST_F(MindDataTestPipeline, TestTextOperationName) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTextOperationName.";

  // Create object for the tensor op, and check the name
  std::string data_file = datasets_root_path_ + "/testVocab/words.txt";
  std::shared_ptr<TensorOperation> sentence_piece_tokenizer_op =
    text::SentencePieceTokenizer(data_file, SPieceTokenizerOutType::kString);
  std::string correct_name = "SentencepieceTokenizer";
  EXPECT_EQ(correct_name, sentence_piece_tokenizer_op->Name());
}

TEST_F(MindDataTestPipeline, TestNormalizeUTF8Success) {
  // Testing the parameter of NormalizeUTF8 interface when the normalize_form is NormalizeForm::kNfkc.
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNormalizeUTF8Success.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/normalize.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create normalizeutf8 operation on ds
  std::shared_ptr<TensorOperation> normalizeutf8 = text::NormalizeUTF8(NormalizeForm::kNfkc);
  EXPECT_NE(normalizeutf8, nullptr);

  // Create Map operation on ds
  ds = ds->Map({normalizeutf8}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  std::vector<std::string> expected = {"ṩ", "ḍ̇", "q̣̇", "fi", "25", "ṩ"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> expected_tensor;
    Tensor::CreateScalar(expected[i], &expected_tensor);
    EXPECT_EQ(*ind, *expected_tensor);
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
  std::shared_ptr<TensorOperation> normalizeutf8 = text::NormalizeUTF8(NormalizeForm::kNfc);
  EXPECT_NE(normalizeutf8, nullptr);

  // Create Map operation on ds
  ds = ds->Map({normalizeutf8}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  std::vector<std::string> expected = {"ṩ", "ḍ̇", "q̣̇", "ﬁ", "2⁵", "ẛ̣"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> expected_tensor;
    Tensor::CreateScalar(expected[i], &expected_tensor);
    EXPECT_EQ(*ind, *expected_tensor);
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
  std::shared_ptr<TensorOperation> normalizeutf8 = text::NormalizeUTF8(NormalizeForm::kNfd);
  EXPECT_NE(normalizeutf8, nullptr);

  // Create Map operation on ds
  ds = ds->Map({normalizeutf8}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  std::vector<std::string> expected = {"ṩ", "ḍ̇", "q̣̇", "ﬁ", "2⁵", "ẛ̣"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> expected_tensor;
    Tensor::CreateScalar(expected[i], &expected_tensor);
    EXPECT_EQ(*ind, *expected_tensor);
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
  std::shared_ptr<TensorOperation> normalizeutf8 = text::NormalizeUTF8(NormalizeForm::kNfkd);
  EXPECT_NE(normalizeutf8, nullptr);

  // Create Map operation on ds
  ds = ds->Map({normalizeutf8}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  std::vector<std::string> expected = {"ṩ", "ḍ̇", "q̣̇", "fi", "25", "ṩ"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> expected_tensor;
    Tensor::CreateScalar(expected[i], &expected_tensor);
    EXPECT_EQ(*ind, *expected_tensor);
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
  std::shared_ptr<TensorOperation> regex_replace = text::RegexReplace("\\s+", "_", true);
  EXPECT_NE(regex_replace, nullptr);

  // Create Map operation on ds
  ds = ds->Map({regex_replace}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  std::vector<std::string> expected = {"Hello_World", "Let's_Go",          "1:hello",        "2:world",
                                       "31:beijing",  "Welcome_to_China!", "_我_不想_长大_", "Welcome_to_Shenzhen!"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> expected_tensor;
    Tensor::CreateScalar(expected[i], &expected_tensor);
    EXPECT_EQ(*ind, *expected_tensor);
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
  std::shared_ptr<TensorOperation> regex_replace = text::RegexReplace("\\s+", "_", false);
  EXPECT_NE(regex_replace, nullptr);

  // Create Map operation on ds
  ds = ds->Map({regex_replace}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  std::vector<std::string> expected = {"Hello_World", "Let's_Go",          "1:hello",          "2:world",
                                       "31:beijing",  "Welcome_to China!", "_我	不想  长大	", "Welcome_to Shenzhen!"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> expected_tensor;
    Tensor::CreateScalar(expected[i], &expected_tensor);
    EXPECT_EQ(*ind, *expected_tensor);
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
  std::shared_ptr<TensorOperation> regex_tokenizer = text::RegexTokenizer("\\s+", "\\s+", false);
  EXPECT_NE(regex_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({regex_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
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
    std::shared_ptr<Tensor> expected_tensor;
    int x = expected[i].size();
    Tensor::CreateFromVector(expected[i], TensorShape({x}), &expected_tensor);
    EXPECT_EQ(*ind, *expected_tensor);
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
  std::shared_ptr<TensorOperation> regex_tokenizer = text::RegexTokenizer("\\s+", "\\s+", true);
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
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  std::vector<std::vector<std::string>> expected = {{"Hello", " ", "World"},
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
    auto ind = row["offsets_start"];
    auto ind1 = row["offsets_limit"];
    auto token = row["token"];
    std::shared_ptr<Tensor> expected_tensor;
    std::shared_ptr<Tensor> expected_tensor_offsets_start;
    std::shared_ptr<Tensor> expected_tensor_offsets_limit;
    int x = expected[i].size();
    Tensor::CreateFromVector(expected[i], TensorShape({x}), &expected_tensor);
    Tensor::CreateFromVector(expected_offsets_start[i], TensorShape({x}), &expected_tensor_offsets_start);
    Tensor::CreateFromVector(expected_offsets_limit[i], TensorShape({x}), &expected_tensor_offsets_limit);
    EXPECT_EQ(*ind, *expected_tensor_offsets_start);
    EXPECT_EQ(*ind1, *expected_tensor_offsets_limit);
    EXPECT_EQ(*token, *expected_tensor);

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
  std::shared_ptr<TensorOperation> unicodechar_tokenizer = text::UnicodeCharTokenizer();
  EXPECT_NE(unicodechar_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({unicodechar_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  std::vector<std::vector<std::string>> expected = {
    {"W", "e", "l", "c", "o", "m", "e", " ", "t", "o", " ", "B", "e", "i", "j", "i", "n", "g", "!"},
    {"北", "京", "欢", "迎", "您", "！"},
    {"我", "喜", "欢", "E", "n", "g", "l", "i", "s", "h", "!"},
    {" ", " "}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> expected_tensor;
    int x = expected[i].size();
    Tensor::CreateFromVector(expected[i], TensorShape({x}), &expected_tensor);
    EXPECT_EQ(*ind, *expected_tensor);
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
  std::shared_ptr<TensorOperation> unicodechar_tokenizer = text::UnicodeCharTokenizer(true);
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
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  std::vector<std::vector<std::string>> expected = {
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
    auto ind = row["offsets_start"];
    auto ind1 = row["offsets_limit"];
    auto token = row["token"];
    std::shared_ptr<Tensor> expected_tensor;
    std::shared_ptr<Tensor> expected_tensor_offsets_start;
    std::shared_ptr<Tensor> expected_tensor_offsets_limit;
    int x = expected[i].size();
    Tensor::CreateFromVector(expected[i], TensorShape({x}), &expected_tensor);
    Tensor::CreateFromVector(expected_offsets_start[i], TensorShape({x}), &expected_tensor_offsets_start);
    Tensor::CreateFromVector(expected_offsets_limit[i], TensorShape({x}), &expected_tensor_offsets_limit);
    EXPECT_EQ(*ind, *expected_tensor_offsets_start);
    EXPECT_EQ(*ind1, *expected_tensor_offsets_limit);
    EXPECT_EQ(*token, *expected_tensor);

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
  std::shared_ptr<TensorOperation> unicodescript_tokenizer = text::UnicodeScriptTokenizer();
  EXPECT_NE(unicodescript_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({unicodescript_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  std::vector<std::vector<std::string>> expected = {
    {"Welcome", "to", "Beijing", "!"}, {"北京欢迎您", "！"}, {"我喜欢", "English", "!"}, {""}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> expected_tensor;
    int x = expected[i].size();
    Tensor::CreateFromVector(expected[i], TensorShape({x}), &expected_tensor);
    EXPECT_EQ(*ind, *expected_tensor);
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
  std::shared_ptr<TensorOperation> unicodescript_tokenizer = text::UnicodeScriptTokenizer(true);
  EXPECT_NE(unicodescript_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({unicodescript_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  std::vector<std::vector<std::string>> expected = {
    {"Welcome", " ", "to", " ", "Beijing", "!"}, {"北京欢迎您", "！"}, {"我喜欢", "English", "!"}, {"  "}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> expected_tensor;
    int x = expected[i].size();
    Tensor::CreateFromVector(expected[i], TensorShape({x}), &expected_tensor);
    EXPECT_EQ(*ind, *expected_tensor);
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
  std::shared_ptr<TensorOperation> unicodescript_tokenizer = text::UnicodeScriptTokenizer(false, true);
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
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  std::vector<std::vector<std::string>> expected = {
    {"Welcome", "to", "Beijing", "!"}, {"北京欢迎您", "！"}, {"我喜欢", "English", "!"}, {""}};

  std::vector<std::vector<uint32_t>> expected_offsets_start = {{0, 8, 11, 18}, {0, 15}, {0, 9, 16}, {0}};
  std::vector<std::vector<uint32_t>> expected_offsets_limit = {{7, 10, 18, 19}, {15, 18}, {9, 16, 17}, {0}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["offsets_start"];
    auto ind1 = row["offsets_limit"];
    auto token = row["token"];
    std::shared_ptr<Tensor> expected_tensor;
    std::shared_ptr<Tensor> expected_tensor_offsets_start;
    std::shared_ptr<Tensor> expected_tensor_offsets_limit;
    int x = expected[i].size();
    Tensor::CreateFromVector(expected[i], TensorShape({x}), &expected_tensor);
    Tensor::CreateFromVector(expected_offsets_start[i], TensorShape({x}), &expected_tensor_offsets_start);
    Tensor::CreateFromVector(expected_offsets_limit[i], TensorShape({x}), &expected_tensor_offsets_limit);
    EXPECT_EQ(*ind, *expected_tensor_offsets_start);
    EXPECT_EQ(*ind1, *expected_tensor_offsets_limit);
    EXPECT_EQ(*token, *expected_tensor);

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
  std::shared_ptr<TensorOperation> unicodescript_tokenizer = text::UnicodeScriptTokenizer(true, true);
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
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  std::vector<std::vector<std::string>> expected = {
    {"Welcome", " ", "to", " ", "Beijing", "!"}, {"北京欢迎您", "！"}, {"我喜欢", "English", "!"}, {"  "}};

  std::vector<std::vector<uint32_t>> expected_offsets_start = {{0, 7, 8, 10, 11, 18}, {0, 15}, {0, 9, 16}, {0}};
  std::vector<std::vector<uint32_t>> expected_offsets_limit = {{7, 8, 10, 11, 18, 19}, {15, 18}, {9, 16, 17}, {2}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["offsets_start"];
    auto ind1 = row["offsets_limit"];
    auto token = row["token"];
    std::shared_ptr<Tensor> expected_tensor;
    std::shared_ptr<Tensor> expected_tensor_offsets_start;
    std::shared_ptr<Tensor> expected_tensor_offsets_limit;
    int x = expected[i].size();
    Tensor::CreateFromVector(expected[i], TensorShape({x}), &expected_tensor);
    Tensor::CreateFromVector(expected_offsets_start[i], TensorShape({x}), &expected_tensor_offsets_start);
    Tensor::CreateFromVector(expected_offsets_limit[i], TensorShape({x}), &expected_tensor_offsets_limit);
    EXPECT_EQ(*ind, *expected_tensor_offsets_start);
    EXPECT_EQ(*ind1, *expected_tensor_offsets_limit);
    EXPECT_EQ(*token, *expected_tensor);

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
  std::shared_ptr<TensorOperation> white_tokenizer = text::WhitespaceTokenizer();
  EXPECT_NE(white_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({white_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  std::vector<std::vector<std::string>> expected = {
    {"This", "is", "a", "text", "file."}, {"Be", "happy", "every", "day."}, {"Good", "luck", "to", "everyone."}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> expected_tensor;
    int x = expected[i].size();
    Tensor::CreateFromVector(expected[i], TensorShape({x}), &expected_tensor);
    EXPECT_EQ(*ind, *expected_tensor);
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
  std::shared_ptr<TensorOperation> white_tokenizer = text::WhitespaceTokenizer(true);
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
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  std::vector<std::vector<std::string>> expected = {
    {"Welcome", "to", "Beijing!"}, {"北京欢迎您！"}, {"我喜欢English!"}, {""}};

  std::vector<std::vector<uint32_t>> expected_offsets_start = {{0, 8, 11}, {0}, {0}, {0}};
  std::vector<std::vector<uint32_t>> expected_offsets_limit = {{7, 10, 19}, {18}, {17}, {0}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["offsets_start"];
    auto ind1 = row["offsets_limit"];
    auto token = row["token"];
    std::shared_ptr<Tensor> expected_tensor;
    std::shared_ptr<Tensor> expected_tensor_offsets_start;
    std::shared_ptr<Tensor> expected_tensor_offsets_limit;
    int x = expected[i].size();
    Tensor::CreateFromVector(expected[i], TensorShape({x}), &expected_tensor);
    Tensor::CreateFromVector(expected_offsets_start[i], TensorShape({x}), &expected_tensor_offsets_start);
    Tensor::CreateFromVector(expected_offsets_limit[i], TensorShape({x}), &expected_tensor_offsets_limit);
    EXPECT_EQ(*ind, *expected_tensor_offsets_start);
    EXPECT_EQ(*ind1, *expected_tensor_offsets_limit);
    EXPECT_EQ(*token, *expected_tensor);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}
