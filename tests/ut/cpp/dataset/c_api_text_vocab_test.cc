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
#include <vector>
#include <string>

#include "common/common.h"
#include "include/api/status.h"
#include "minddata/dataset/include/dataset/datasets.h"
#include "minddata/dataset/include/dataset/text.h"
#include "minddata/dataset/include/dataset/transforms.h"

using namespace mindspore::dataset;
using mindspore::Status;
using mindspore::dataset::DataType;
using mindspore::dataset::ShuffleMode;
using mindspore::dataset::Tensor;
using mindspore::dataset::Vocab;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

// Macro to compare 2 MSTensors as not equal; compare datasize only
#define EXPECT_MSTENSOR_DATA_NE(_mstensor1, _mstensor2)      \
  do {                                                       \
    EXPECT_NE(_mstensor1.DataSize(), _mstensor2.DataSize()); \
  } while (false)

/// Feature: C++ text.Vocab class.
/// Description: Test TokensToIds() IdsToTokens() methods of text::Vocab.
/// Expectation: Success.
TEST_F(MindDataTestPipeline, TestVocabLookupAndReverseLookup) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVocabLookupAndReverseLookup.";
  // Create a vocab from vector
  std::vector<std::string> list = {"home", "IS", "behind", "the", "world", "ahead", "!"};
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  Status s = Vocab::BuildFromVector(list, {"<pad>", "<unk>"}, true, &vocab);
  EXPECT_EQ(s, Status::OK());

  // lookup, convert token to id
  auto single_index = vocab->TokensToIds("home");
  EXPECT_EQ(single_index, 2);
  single_index = vocab->TokensToIds("hello");
  EXPECT_EQ(single_index, -1);

  // lookup multiple tokens
  auto multi_indexs = vocab->TokensToIds(std::vector<std::string>{"<pad>", "behind"});
  std::vector<int32_t> expected_multi_indexs = {0, 4};
  EXPECT_EQ(multi_indexs, expected_multi_indexs);
  multi_indexs = vocab->TokensToIds(std::vector<std::string>{"<pad>", "apple"});
  expected_multi_indexs = {0, -1};
  EXPECT_EQ(multi_indexs, expected_multi_indexs);

  // reverse lookup, convert id to token
  auto single_word = vocab->IdsToTokens(2);
  EXPECT_EQ(single_word, "home");
  single_word = vocab->IdsToTokens(-1);
  EXPECT_EQ(single_word, "");

  // reverse lookup multiple ids
  auto multi_words = vocab->IdsToTokens(std::vector<int32_t>{0, 4});
  std::vector<std::string> expected_multi_words = {"<pad>", "behind"};
  EXPECT_EQ(multi_words, expected_multi_words);
  multi_words = vocab->IdsToTokens(std::vector<int32_t>{0, 99});
  expected_multi_words = {"<pad>", ""};
  EXPECT_EQ(multi_words, expected_multi_words);
}

/// Feature: C++ text.Vocab class
/// Description: Test Lookup op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestVocabLookupOp) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVocabLookupOp.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testVocab/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create a vocab from vector
  std::vector<std::string> list = {"home", "IS", "behind", "the", "world", "ahead", "!"};
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  Status s = Vocab::BuildFromVector(list, {"<pad>", "<unk>"}, true, &vocab);
  EXPECT_EQ(s, Status::OK());

  // Create Lookup operation on ds
  std::shared_ptr<TensorTransform> lookup =
    std::make_shared<text::Lookup>(vocab, "<unk>", mindspore::DataType::kNumberTypeInt32);
  EXPECT_NE(lookup, nullptr);

  // Create Map operation on ds
  ds = ds->Map({lookup}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  std::vector<int32_t> expected = {2, 1, 4, 5, 6, 7};
  while (row.size() != 0) {
    auto ind = row["text"];
    MS_LOG(INFO) << ind.Shape();
    TEST_MS_LOG_MSTENSOR(INFO, "ind: ", ind);
    std::shared_ptr<Tensor> de_expected_item;
    ASSERT_OK(Tensor::CreateScalar(expected[i], &de_expected_item));
    mindspore::MSTensor ms_expected_item =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_item));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_item);

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: C++ text.Vocab class
/// Description: Test Lookup op using an empty string as the unknown_token
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestVocabLookupOpEmptyString) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVocabLookupOpEmptyString.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testVocab/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create a vocab from vector
  std::vector<std::string> list = {"home", "IS", "behind", "the", "world", "ahead", "!"};
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  Status s = Vocab::BuildFromVector(list, {"<pad>", ""}, true, &vocab);
  EXPECT_EQ(s, Status::OK());

  // Create Lookup operation on ds
  std::shared_ptr<TensorTransform> lookup =
    std::make_shared<text::Lookup>(vocab, "", mindspore::DataType::kNumberTypeInt32);
  EXPECT_NE(lookup, nullptr);

  // Create Map operation on ds
  ds = ds->Map({lookup}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  std::vector<int32_t> expected = {2, 1, 4, 5, 6, 7};
  while (row.size() != 0) {
    auto ind = row["text"];
    MS_LOG(INFO) << ind.Shape();
    TEST_MS_LOG_MSTENSOR(INFO, "ind: ", ind);
    std::shared_ptr<Tensor> de_expected_item;
    ASSERT_OK(Tensor::CreateScalar(expected[i], &de_expected_item));
    mindspore::MSTensor ms_expected_item =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_item));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_item);

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: C++ text.Vocab class
/// Description: Test Lookup op with mindspore::DataType::kNumberTypeBool
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestVocabLookupBool) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVocabLookupBool.";
  // Invoke Lookup with Bool data_type

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testVocab/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create a vocab from vector
  std::vector<std::string> list = {"home", "IS", "behind", "the", "world", "ahead", "!"};
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  Status s = Vocab::BuildFromVector(list, {"<pad>", "<unk>"}, true, &vocab);
  EXPECT_EQ(s, Status::OK());

  // Create Lookup operation on ds
  std::shared_ptr<TensorTransform> lookup =
    std::make_shared<text::Lookup>(vocab, "<unk>", mindspore::DataType::kNumberTypeBool);
  EXPECT_NE(lookup, nullptr);

  // Create Map operation on ds
  ds = ds->Map({lookup}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    MS_LOG(INFO) << ind.Shape();
    TEST_MS_LOG_MSTENSOR(INFO, "ind: ", ind);

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: C++ text.Vocab class
/// Description: Test Lookup op with an unknown token that is not a word of vocab
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestVocabLookupOpFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVocabLookupOpFail1.";
  // Create a TextFile Dataset
  std::string data_file = datasets_root_path_ + "/testVocab/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Build vocab from vector
  std::vector<std::string> list = {"home", "IS", "behind", "the", "world", "ahead", "!"};
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  Status s = Vocab::BuildFromVector(list, {}, true, &vocab);
  EXPECT_EQ(s, Status::OK());

  // Create lookup op for ds
  std::shared_ptr<TensorTransform> lookup =
    std::make_shared<text::Lookup>(vocab, "<unk>", mindspore::DataType::kNumberTypeInt32);
  EXPECT_NE(lookup, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({lookup});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Lookup input ("<unk>" is not a word of vocab)
  EXPECT_EQ(iter, nullptr);
}

/// Feature: C++ text.Vocab class
/// Description: Test Lookup op with null vocab
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestVocabLookupOpFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVocabLookupOpFail2.";
  // Create a TextFile Dataset
  std::string data_file = datasets_root_path_ + "/testVocab/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Vocab has nothing
  std::shared_ptr<Vocab> vocab;

  // Create lookup op
  std::shared_ptr<TensorTransform> lookup =
    std::make_shared<text::Lookup>(vocab, "", mindspore::DataType::kNumberTypeInt32);
  EXPECT_NE(lookup, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({lookup});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Lookup input (vocab is null)
  EXPECT_EQ(iter, nullptr);
}

/// Feature: C++ text.Vocab class
/// Description: Test Lookup op with mindspore::DataType::kObjectTypeString
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestVocabLookupOpFail3DataType) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVocabLookupOpFail3DataType.";
  // Create a TextFile Dataset
  std::string data_file = datasets_root_path_ + "/testVocab/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Build vocab from vector
  std::vector<std::string> list = {"home", "IS", "behind", "the", "world", "ahead", "!"};
  std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  Status s = Vocab::BuildFromVector(list, {"<pad>", "<unk>"}, true, &vocab);
  EXPECT_EQ(s, Status::OK());

  // Create lookup op for ds
  std::shared_ptr<TensorTransform> lookup =
    std::make_shared<text::Lookup>(vocab, "", mindspore::DataType::kObjectTypeString);
  EXPECT_NE(lookup, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({lookup});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Lookup input (String is not valid for data_type)
  EXPECT_EQ(iter, nullptr);
}

/// Feature: C++ text.Vocab class
/// Description: Test BuildVocab using a dataset
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestVocabFromDataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVocabFromDataset.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testVocab/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create vocab from dataset
  std::shared_ptr<Vocab> vocab = ds->BuildVocab({"text"}, {0, std::numeric_limits<int64_t>::max()},
                                                std::numeric_limits<int64_t>::max(), {"<pad>", "<unk>"}, true);
  EXPECT_NE(vocab, nullptr);

  // Check if vocab has words or not
  int32_t home_index = vocab->TokensToIds("home");
  EXPECT_EQ(home_index, 4);

  // Create Lookup operation on ds
  std::shared_ptr<TensorTransform> lookup =
    std::make_shared<text::Lookup>(vocab, "<unk>", mindspore::DataType::kNumberTypeInt32);
  EXPECT_NE(lookup, nullptr);

  // Create Map operation on ds
  ds = ds->Map({lookup}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  std::vector<int32_t> expected = {4, 5, 3, 6, 7, 2};
  while (row.size() != 0) {
    auto ind = row["text"];
    MS_LOG(INFO) << ind.Shape();
    TEST_MS_LOG_MSTENSOR(INFO, "ind: ", ind);
    std::shared_ptr<Tensor> de_expected_item;
    ASSERT_OK(Tensor::CreateScalar(expected[i], &de_expected_item));
    mindspore::MSTensor ms_expected_item =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_item));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_item);

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: C++ text.Vocab class
/// Description: Test BuildVocab with default parameters
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestVocabFromDatasetDefault) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVocabFromDatasetDefault.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testVocab/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create vocab from dataset
  std::shared_ptr<Vocab> vocab = ds->BuildVocab();
  EXPECT_NE(vocab, nullptr);

  // Check if vocab has words or not
  int32_t home_index = vocab->TokensToIds("home");
  EXPECT_EQ(home_index, 2);

  // Create Lookup operation on ds
  // Use default data_type parameter
  std::shared_ptr<TensorTransform> lookup = std::make_shared<text::Lookup>(vocab, "home");
  EXPECT_NE(lookup, nullptr);

  // Create Map operation on ds
  ds = ds->Map({lookup});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  std::vector<int32_t> expected = {2, 3, 1, 4, 5, 0};
  std::vector<int64_t> not_expected = {2, 3, 1, 4, 5, 0};
  while (row.size() != 0) {
    auto ind = row["text"];
    MS_LOG(INFO) << ind.Shape();
    TEST_MS_LOG_MSTENSOR(INFO, "ind: ", ind);

    std::shared_ptr<Tensor> de_expected_item;
    ASSERT_OK(Tensor::CreateScalar(expected[i], &de_expected_item));
    mindspore::MSTensor ms_expected_item =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_item));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_item);

    std::shared_ptr<Tensor> de_not_expected_item;
    ASSERT_OK(Tensor::CreateScalar(not_expected[i], &de_not_expected_item));
    mindspore::MSTensor ms_not_expected_item =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_not_expected_item));
    EXPECT_MSTENSOR_DATA_NE(ind, ms_not_expected_item);

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: C++ text.Vocab class
/// Description: Test BuildVocab where top_k is negative
/// Expectation: Throw correct error and message
TEST_F(MindDataTestPipeline, TestVocabFromDatasetFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVocabFromDatasetFail1.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testVocab/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create vocab from dataset
  // Expected failure: top_k can not be negative
  std::shared_ptr<Vocab> vocab =
    ds->BuildVocab({"text"}, {0, std::numeric_limits<int64_t>::max()}, -2, {"<pad>", "<unk>"}, true);
  EXPECT_EQ(vocab, nullptr);
}

/// Feature: C++ text.Vocab class
/// Description: Test BuildVocab where frequency_range [a, b] is a > b
/// Expectation: Throw correct error and message
TEST_F(MindDataTestPipeline, TestVocabFromDatasetFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVocabFromDatasetFail2.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testVocab/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create vocab from dataset
  // Expected failure: frequency_range [a,b] should be 0 <= a <= b
  std::shared_ptr<Vocab> vocab =
    ds->BuildVocab({"text"}, {4, 1}, std::numeric_limits<int64_t>::max(), {"<pad>", "<unk>"}, true);
  EXPECT_EQ(vocab, nullptr);
}

/// Feature: C++ text.Vocab class
/// Description: Test BuildVocab where column name does not exist in dataset
/// Expectation: Throw correct error and message
TEST_F(MindDataTestPipeline, TestVocabFromDatasetFail3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVocabFromDatasetFail3.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testVocab/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create vocab from dataset
  // Expected failure: column name does not exist in ds
  std::shared_ptr<Vocab> vocab = ds->BuildVocab({"ColumnNotExist"});
  EXPECT_EQ(vocab, nullptr);
}

/// Feature: C++ text.Vocab class
/// Description: Test BuildVocab where special tokens are already in the dataset
/// Expectation: Throw correct error and message
TEST_F(MindDataTestPipeline, TestVocabFromDatasetFail4) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVocabFromDatasetFail4.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testVocab/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create vocab from dataset
  // Expected failure: special tokens are already in the dataset
  std::shared_ptr<Vocab> vocab =
    ds->BuildVocab({"text"}, {0, std::numeric_limits<int64_t>::max()}, std::numeric_limits<int64_t>::max(), {"world"});
  EXPECT_EQ(vocab, nullptr);
}

/// Feature: C++ text.Vocab class
/// Description: Test BuildVocab from dataset and Lookup op with mindspore::DataType::kNumberTypeInt64
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestVocabFromDatasetInt64) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVocabFromDatasetInt64.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testVocab/words.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create vocab from dataset
  std::shared_ptr<Vocab> vocab = ds->BuildVocab();
  EXPECT_NE(vocab, nullptr);

  // Check if vocab has words or not
  int32_t home_index = vocab->TokensToIds("home");
  EXPECT_EQ(home_index, 2);

  // Create Lookup operation on ds
  std::shared_ptr<TensorTransform> lookup =
    std::make_shared<text::Lookup>(vocab, "home", mindspore::DataType::kNumberTypeInt64);
  EXPECT_NE(lookup, nullptr);

  // Create Map operation on ds
  ds = ds->Map({lookup});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  std::vector<int64_t> expected = {2, 3, 1, 4, 5, 0};
  std::vector<int8_t> not_expected = {2, 3, 1, 4, 5, 0};
  while (row.size() != 0) {
    auto ind = row["text"];
    MS_LOG(INFO) << ind.Shape();
    TEST_MS_LOG_MSTENSOR(INFO, "ind: ", ind);

    std::shared_ptr<Tensor> de_expected_item;
    ASSERT_OK(Tensor::CreateScalar(expected[i], &de_expected_item));
    mindspore::MSTensor ms_expected_item =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_item));
    EXPECT_MSTENSOR_EQ(ind, ms_expected_item);

    std::shared_ptr<Tensor> de_not_expected_item;
    ASSERT_OK(Tensor::CreateScalar(not_expected[i], &de_not_expected_item));
    mindspore::MSTensor ms_not_expected_item =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_not_expected_item));
    EXPECT_MSTENSOR_DATA_NE(ind, ms_not_expected_item);

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}
