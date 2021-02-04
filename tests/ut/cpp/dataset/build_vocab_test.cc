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
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <string>

#include "common/common.h"
#include "include/api/status.h"
#include "minddata/dataset/include/datasets.h"
#include "minddata/dataset/text/vocab.h"

using mindspore::dataset::Tensor;
using mindspore::dataset::Vocab;

class MindDataTestVocab : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestVocab, TestVocabFromUnorderedMap) {
  MS_LOG(INFO) << "Doing MindDataTestVocab-TestVocabFromUnorderedMap.";
  // Build a map
  std::unordered_map<std::string, int32_t> dict;
  dict["banana"] = 0;
  dict["apple"] = 1;
  dict["cat"] = 2;
  dict["dog"] = 3;

  // Build vocab from map
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  Status s = Vocab::BuildFromUnorderedMap(dict, &vocab);
  EXPECT_EQ(s, Status::OK());

  // Look up specified words
  std::vector<std::string> words = {"apple", "dog", "egg"};
  std::vector<int64_t> expected = {1, 3, -1};
  for (uint32_t i = 0; i < words.size(); ++i) {
    int32_t x = vocab->Lookup(words[i]);
    EXPECT_EQ(x, expected[i]);
  }
}

TEST_F(MindDataTestVocab, TestVocabFromEmptyMap) {
  MS_LOG(INFO) << "Doing MindDataTestVocab-TestVocabFromEmptyMap.";
  // Build vocab from empty map
  std::unordered_map<std::string, int32_t> dict;
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  Status s = Vocab::BuildFromUnorderedMap(dict, &vocab);
  EXPECT_EQ(s, Status::OK());

  // Look up specified words
  // Expect that we will return -1 when word is not in vocab
  std::vector<std::string> words = {"apple", "dog", "egg"};
  std::vector<int64_t> expected = {-1, -1, -1};
  for (uint32_t i = 0; i < words.size(); ++i) {
    int32_t x = vocab->Lookup(words[i]);
    EXPECT_EQ(x, expected[i]);
  }
}

TEST_F(MindDataTestVocab, TestVocabFromMapFail) {
  MS_LOG(INFO) << "Doing MindDataTestVocab-TestVocabFromMapFail.";
  // Build a map
  std::unordered_map<std::string, int32_t> dict;
  dict["banana"] = 0;
  dict["apple"] = -1;

  // Expected failure: index of word can not be negative
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  Status s = Vocab::BuildFromUnorderedMap(dict, &vocab);
  EXPECT_NE(s, Status::OK());
}

TEST_F(MindDataTestVocab, TestVocabFromVectorPrependSpTokens) {
  MS_LOG(INFO) << "Doing MindDataTestVocab-TestVocabFromVectorPrependSpTokens.";
  // Build vocab from a vector of words, special tokens are prepended to vocab
  std::vector<std::string> list = {"apple", "banana", "cat", "dog", "egg"};
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  Status s = Vocab::BuildFromVector(list, {"<unk>"}, true, &vocab);
  EXPECT_EQ(s, Status::OK());

  // Look up specified words
  // Expect that we will return -1 when word is not in vocab
  std::vector<std::string> words = {"apple", "banana", "fox"};
  std::vector<int64_t> expected = {1, 2, -1};
  for (uint32_t i = 0; i < words.size(); ++i) {
    int32_t x = vocab->Lookup(words[i]);
    EXPECT_EQ(x, expected[i]);
  }
}

TEST_F(MindDataTestVocab, TestVocabFromVectorAppendSpTokens) {
  MS_LOG(INFO) << "Doing MindDataTestVocab-TestVocabFromVectorAppendSpTokens.";
  // Build vocab from a vector of words, special tokens are appended to vocab
  std::vector<std::string> list = {"apple", "banana", "cat", "dog", "egg"};
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  Status s = Vocab::BuildFromVector(list, {"<unk>"}, false, &vocab);
  EXPECT_EQ(s, Status::OK());

  // Look up specified words
  std::vector<std::string> words = {"apple", "<unk>", "fox"};
  std::vector<int64_t> expected = {0, 5, -1};
  for (uint32_t i = 0; i < words.size(); ++i) {
    int32_t x = vocab->Lookup(words[i]);
    EXPECT_EQ(x, expected[i]);
  }
}

TEST_F(MindDataTestVocab, TestVocabFromVectorWithNoSpTokens) {
  MS_LOG(INFO) << "Doing MindDataTestVocab-TestVocabFromVectorWithNoSpTokens.";
  // Build vocab from a vector of words with no special tokens
  std::vector<std::string> list = {"apple", "banana", "cat", "dog", "egg"};
  std::vector<std::string> sp_tokens = {};
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  Status s = Vocab::BuildFromVector(list, sp_tokens, true, &vocab);
  EXPECT_EQ(s, Status::OK());

  // Look up specified words
  std::vector<std::string> words = {"apple", "banana", "fox", "<pad>"};
  std::vector<int64_t> expected = {0, 1, -1, -1};
  for (uint32_t i = 0; i < words.size(); ++i) {
    int32_t x = vocab->Lookup(words[i]);
    EXPECT_EQ(x, expected[i]);
  }
}

TEST_F(MindDataTestVocab, TestVocabFromEmptyVector) {
  MS_LOG(INFO) << "Doing MindDataTestVocab-TestVocabFromEmptyVector.";
  // Build vocab from empty vector
  std::vector<std::string> list = {};
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  Status s = Vocab::BuildFromVector(list, {}, false, &vocab);
  EXPECT_EQ(s, Status::OK());

  // Look up specified words
  // Expect that we will return -1 when word is not in vocab
  std::vector<std::string> words = {"apple", "banana", "fox"};
  std::vector<int64_t> expected = {-1, -1, -1};
  for (uint32_t i = 0; i < words.size(); ++i) {
    int32_t x = vocab->Lookup(words[i]);
    EXPECT_EQ(x, expected[i]);
  }
}

TEST_F(MindDataTestVocab, TestVocabFromVectorFail1) {
  MS_LOG(INFO) << "Doing MindDataTestVocab-TestVocabFromVectorFail1.";
  // Build vocab from a vector of words
  std::vector<std::string> list = {"apple", "apple", "cat", "cat", "egg"};
  std::vector<std::string> sp_tokens = {};
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();

  // Expected failure: duplicate word apple
  Status s = Vocab::BuildFromVector(list, sp_tokens, true, &vocab);
  EXPECT_NE(s, Status::OK());
}

TEST_F(MindDataTestVocab, TestVocabFromVectorFail2) {
  MS_LOG(INFO) << "Doing MindDataTestVocab-TestVocabFromVectorFail2.";
  // Build vocab from a vector
  std::vector<std::string> list = {"apple", "dog", "egg"};
  std::vector<std::string> sp_tokens = {"<pad>", "<unk>", "<pad>", "<unk>", "<none>"};
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();

  // Expected failure: duplicate special token <pad> <unk>
  Status s = Vocab::BuildFromVector(list, sp_tokens, true, &vocab);
  EXPECT_NE(s, Status::OK());
}

TEST_F(MindDataTestVocab, TestVocabFromVectorFail3) {
  MS_LOG(INFO) << "Doing MindDataTestVocab-TestVocabFromVectorFail3.";
  // Build vocab from a vector
  std::vector<std::string> list = {"apple", "dog", "egg", "<unk>", ""};
  std::vector<std::string> sp_tokens = {"", "<unk>"};
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();

  // Expected failure: special tokens are already existed in word_list
  Status s = Vocab::BuildFromVector(list, sp_tokens, true, &vocab);
  EXPECT_NE(s, Status::OK());
}

TEST_F(MindDataTestVocab, TestVocabFromFile) {
  MS_LOG(INFO) << "Doing MindDataTestVocab-TestVocabFromFile.";
  // Build vocab from local file
  std::string vocab_dir = datasets_root_path_ + "/testVocab/vocab_list.txt";
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  Status s = Vocab::BuildFromFileCpp(vocab_dir, ",", -1, {"<pad>", "<unk>"}, true, &vocab);
  EXPECT_EQ(s, Status::OK());

  // Look up specified words
  std::vector<std::string> words = {"not", "all"};
  std::vector<int64_t> expected = {2, 3};
  for (uint32_t i = 0; i < words.size(); ++i) {
    int32_t x = vocab->Lookup(words[i]);
    EXPECT_EQ(x, expected[i]);
  }
}

TEST_F(MindDataTestVocab, TestVocabFromFileFail1) {
  MS_LOG(INFO) << "Doing MindDataTestVocab-TestVocabFromFileFail1.";
  // Build vocab from local file which is not exist
  std::string vocab_dir = datasets_root_path_ + "/testVocab/not_exist.txt";
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  Status s = Vocab::BuildFromFileCpp(vocab_dir, ",", -1, {}, true, &vocab);
  EXPECT_NE(s, Status::OK());
}

TEST_F(MindDataTestVocab, TestVocabFromFileFail2) {
  MS_LOG(INFO) << "Doing MindDataTestVocab-TestVocabFromFileFail2.";
  // Build vocab from local file
  std::string vocab_dir = datasets_root_path_ + "/testVocab/vocab_list.txt";
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();

  // Expected failure: vocab_size should be either -1 or positive integer
  Status s = Vocab::BuildFromFileCpp(vocab_dir, ",", -2, {}, true, &vocab);
  EXPECT_NE(s, Status::OK());
}

TEST_F(MindDataTestVocab, TestVocabFromFileFail3) {
  MS_LOG(INFO) << "Doing MindDataTestVocab-TestVocabFromFileFail3.";
  // Build vocab from local file
  std::string vocab_dir = datasets_root_path_ + "/testVocab/vocab_list.txt";
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();

  // Expected failure: duplicate special token <unk>
  Status s = Vocab::BuildFromFileCpp(vocab_dir, ",", -1, {"<unk>", "<unk>"}, true, &vocab);
  EXPECT_NE(s, Status::OK());
}

TEST_F(MindDataTestVocab, TestVocabFromFileFail4) {
  MS_LOG(INFO) << "Doing MindDataTestVocab-TestVocabFromFileFail4.";
  // Build vocab from local file
  std::string vocab_dir = datasets_root_path_ + "/testVocab/vocab_list.txt";
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();

  // Expected failure: special_tokens and word_list contain duplicate word
  Status s = Vocab::BuildFromFileCpp(vocab_dir, ",", -1, {"home"}, true, &vocab);
  EXPECT_NE(s, Status::OK());
}
