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
#include "minddata/dataset/include/constants.h"
#include "minddata/dataset/include/datasets.h"
#include "minddata/dataset/include/text.h"
#include "minddata/dataset/include/transforms.h"
#include "minddata/dataset/text/sentence_piece_vocab.h"

using namespace mindspore::dataset;
using mindspore::dataset::SentencePieceModel;
using mindspore::dataset::SentencePieceVocab;
using mindspore::dataset::ShuffleMode;
using mindspore::dataset::Tensor;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestPipeline, TestSentencePieceVocabSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSentencePieceVocabSuccess1 plus sentencepiece tokenizer.";

  // Create a TextFile dataset
  std::string vocab_file = datasets_root_path_ + "/test_sentencepiece/botchan.txt";
  std::shared_ptr<Dataset> ds_vocab = TextFile({vocab_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds_vocab, nullptr);

  // Create vocab from dataset
  std::shared_ptr<SentencePieceVocab> vocab =
    ds_vocab->BuildSentencePieceVocab({}, 5000, 0.9995, SentencePieceModel::kUnigram, {});
  EXPECT_NE(vocab, nullptr);

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/sentencepiece_tokenizer.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);

  // Create SentencePieceTokenizer operation from vocab object
  std::shared_ptr<TensorTransform> sentencepiece_tokenizer =
    std::make_shared<text::SentencePieceTokenizer>(vocab, mindspore::dataset::SPieceTokenizerOutType::kString);
  EXPECT_NE(sentencepiece_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({sentencepiece_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  // Expected result after tokenization
  std::vector<std::string> expected = {"▁I", "▁sa", "w", "▁a", "▁girl", "▁with", "▁a", "▁te", "les", "co", "pe", "."};
  std::shared_ptr<Tensor> de_expected_tensor;
  ASSERT_OK(Tensor::CreateFromVector(expected, &de_expected_tensor));
  mindspore::MSTensor expected_tensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto txt = row["text"];
    TEST_MS_LOG_MSTENSOR(INFO, "txt: ", txt);

    EXPECT_MSTENSOR_EQ(txt, expected_tensor);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 1);
}

TEST_F(MindDataTestPipeline, TestSentencePieceVocabSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSentencePieceVocabSuccess2 plus sentencepiece tokenizer.";

  // Create a TextFile dataset
  std::string vocab_file = datasets_root_path_ + "/test_sentencepiece/botchan.txt";
  std::shared_ptr<Dataset> ds_vocab = TextFile({vocab_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds_vocab, nullptr);

  // Create vocab from dataset
  std::shared_ptr<SentencePieceVocab> vocab =
    ds_vocab->BuildSentencePieceVocab({}, 5000, 0.9995, SentencePieceModel::kUnigram, {});
  EXPECT_NE(vocab, nullptr);

  // Save vocab model to local
  vocab->SaveModel(&vocab, datasets_root_path_ + "/test_sentencepiece", "m.model");

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/sentencepiece_tokenizer.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);

  // Create SentencePieceTokenizer operation from local vocab model
  std::string vocab_model = datasets_root_path_ + "/test_sentencepiece/m.model";
  std::shared_ptr<TensorTransform> sentencepiece_tokenizer =
    std::make_shared<text::SentencePieceTokenizer>(vocab_model, mindspore::dataset::SPieceTokenizerOutType::kString);
  EXPECT_NE(sentencepiece_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({sentencepiece_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  // Expected result after tokenization
  std::vector<std::string> expected = {"▁I", "▁sa", "w", "▁a", "▁girl", "▁with", "▁a", "▁te", "les", "co", "pe", "."};
  std::shared_ptr<Tensor> de_expected_tensor;
  ASSERT_OK(Tensor::CreateFromVector(expected, &de_expected_tensor));
  mindspore::MSTensor expected_tensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto txt = row["text"];
    TEST_MS_LOG_MSTENSOR(INFO, "txt: ", txt);

    EXPECT_MSTENSOR_EQ(txt, expected_tensor);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 1);
}

TEST_F(MindDataTestPipeline, TestSentencePieceVocabFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSentencePieceVocabFail1 with incorrect parameter.";

  // Create a TextFile dataset
  std::string vocab_file = datasets_root_path_ + "/test_sentencepiece/botchan.txt";
  std::shared_ptr<Dataset> ds_vocab = TextFile({vocab_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds_vocab, nullptr);

  // vocab_size can not less than or equal to 0
  std::shared_ptr<SentencePieceVocab> vocab1 =
    ds_vocab->BuildSentencePieceVocab({}, 0, 0.9995, SentencePieceModel::kUnigram, {});
  EXPECT_EQ(vocab1, nullptr);

  // character_coverage should to be between 0.98 and 1.0
  std::shared_ptr<SentencePieceVocab> vocab2 =
    ds_vocab->BuildSentencePieceVocab({}, 1, 0.979, SentencePieceModel::kUnigram, {});
  EXPECT_EQ(vocab2, nullptr);

  // character_coverage should to be between 0.98 and 1.0
  std::shared_ptr<SentencePieceVocab> vocab3 =
    ds_vocab->BuildSentencePieceVocab({}, 1, 1.01, SentencePieceModel::kUnigram, {});
  EXPECT_EQ(vocab3, nullptr);

  // column name does not exist
  std::shared_ptr<SentencePieceVocab> vocab4 =
    ds_vocab->BuildSentencePieceVocab({"image"}, 2, 0.98, SentencePieceModel::kUnigram, {});
  EXPECT_EQ(vocab4, nullptr);
}

TEST_F(MindDataTestPipeline, TestSentencePieceTokenizerFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSentencePieceTokenizerFail with incorrect parameter.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/sentencepiece_tokenizer.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);

  // Create SentencePieceTokenizer operation from local vocab model
  std::string vocab_model = "";
  std::shared_ptr<TensorTransform> sentencepiece_tokenizer =
    std::make_shared<text::SentencePieceTokenizer>(vocab_model, mindspore::dataset::SPieceTokenizerOutType::kString);
  EXPECT_NE(sentencepiece_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({sentencepiece_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid SentencePieceTokenizer input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestSentencePieceTokenizerFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSentencePieceTokenizerFail2 with incorrect parameter.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/sentencepiece_tokenizer.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);

  // Create SentencePieceTokenizer operation from local vocab model
  std::string vocab_model = "m.model";
  std::shared_ptr<TensorTransform> sentencepiece_tokenizer =
    std::make_shared<text::SentencePieceTokenizer>(vocab_model, mindspore::dataset::SPieceTokenizerOutType::kString);
  EXPECT_NE(sentencepiece_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({sentencepiece_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid SentencePieceTokenizer input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestSentencePieceTokenizerFail3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSentencePieceTokenizerFail3 with incorrect parameter.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/sentencepiece_tokenizer.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);

  // Create SentencePieceTokenizer operation from vocab object
  std::shared_ptr<SentencePieceVocab> vocab_model = nullptr;
  std::shared_ptr<TensorTransform> sentencepiece_tokenizer =
    std::make_shared<text::SentencePieceTokenizer>(vocab_model, mindspore::dataset::SPieceTokenizerOutType::kString);
  EXPECT_NE(sentencepiece_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({sentencepiece_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid SentencePieceTokenizer input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestSentencePieceTokenizerFail4) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSentencePieceTokenizerFail with invalid SentencePieceVocab object.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/sentencepiece_tokenizer.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);

  // Create SentencePieceTokenizer operation from vocab object
  std::shared_ptr<SentencePieceVocab> vocab_model4 = std::make_shared<SentencePieceVocab>();
  std::shared_ptr<TensorTransform> sentencepiece_tokenizer4 =
    std::make_shared<text::SentencePieceTokenizer>(vocab_model4, mindspore::dataset::SPieceTokenizerOutType::kString);
  EXPECT_NE(sentencepiece_tokenizer4, nullptr);

  // Create Map operation on ds
  ds = ds->Map({sentencepiece_tokenizer4}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
}
