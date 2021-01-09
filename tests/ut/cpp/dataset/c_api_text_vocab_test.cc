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
#include "minddata/dataset/include/text.h"
#include "minddata/dataset/include/transforms.h"
#include "minddata/dataset/text/vocab.h"

using namespace mindspore::dataset;
using mindspore::dataset::DataType;
using mindspore::dataset::ShuffleMode;
using mindspore::dataset::Status;
using mindspore::dataset::Tensor;
using mindspore::dataset::Vocab;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

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
  std::shared_ptr<TensorOperation> lookup = text::Lookup(vocab, "<unk>", "int32");
  EXPECT_NE(lookup, nullptr);

  // Create Map operation on ds
  ds = ds->Map({lookup}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  std::vector<int32_t> expected = {2, 1, 4, 5, 6, 7};
  while (row.size() != 0) {
    auto ind = row["text"];
    MS_LOG(INFO) << ind->shape() << " " << *ind;
    std::shared_ptr<Tensor> expected_item;
    Tensor::CreateScalar(expected[i], &expected_item);
    EXPECT_EQ(*ind, *expected_item);
    iter->GetNextRow(&row);
    i++;
  }
}

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
  std::shared_ptr<TensorOperation> lookup = text::Lookup(vocab, "", "int32");
  EXPECT_NE(lookup, nullptr);

  // Create Map operation on ds
  ds = ds->Map({lookup}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  std::vector<int32_t> expected = {2, 1, 4, 5, 6, 7};
  while (row.size() != 0) {
    auto ind = row["text"];
    MS_LOG(INFO) << ind->shape() << " " << *ind;
    std::shared_ptr<Tensor> expected_item;
    Tensor::CreateScalar(expected[i], &expected_item);
    EXPECT_EQ(*ind, *expected_item);
    iter->GetNextRow(&row);
    i++;
  }
}

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
  // Expected failure: "<unk>" is not a word of vocab
  std::shared_ptr<TensorOperation> lookup = text::Lookup(vocab, "<unk>", "int32");
  EXPECT_EQ(lookup, nullptr);
}

TEST_F(MindDataTestPipeline, TestVocabLookupOpFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVocabLookupOpFail2.";
  // Vocab has nothing
  std::shared_ptr<Vocab> vocab;

  // Create lookup op
  // Expected failure: vocab is null
  std::shared_ptr<TensorOperation> lookup = text::Lookup(vocab, "", "int32");
  EXPECT_EQ(lookup, nullptr);
}

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
  int32_t home_index = vocab->Lookup("home");
  EXPECT_EQ(home_index, 4);

  // Create Lookup operation on ds
  std::shared_ptr<TensorOperation> lookup = text::Lookup(vocab, "<unk>", "int32");
  EXPECT_NE(lookup, nullptr);

  // Create Map operation on ds
  ds = ds->Map({lookup}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  std::vector<int32_t> expected = {4, 5, 3, 6, 7, 2};
  while (row.size() != 0) {
    auto ind = row["text"];
    MS_LOG(INFO) << ind->shape() << " " << *ind;
    std::shared_ptr<Tensor> expected_item;
    Tensor::CreateScalar(expected[i], &expected_item);
    EXPECT_EQ(*ind, *expected_item);
    iter->GetNextRow(&row);
    i++;
  }
}

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
  int32_t home_index = vocab->Lookup("home");
  EXPECT_EQ(home_index, 2);

  // Create Lookup operation on ds
  std::shared_ptr<TensorOperation> lookup = text::Lookup(vocab, "home");
  EXPECT_NE(lookup, nullptr);

  // Create Map operation on ds
  ds = ds->Map({lookup});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  std::vector<int32_t> expected = {2, 3, 1, 4, 5, 0};
  std::vector<int64_t> not_expected = {2, 3, 1, 4, 5, 0};
  while (row.size() != 0) {
    auto ind = row["text"];
    MS_LOG(INFO) << ind->shape() << " " << *ind;
    std::shared_ptr<Tensor> expected_item, not_expected_item;
    Tensor::CreateScalar(expected[i], &expected_item);
    Tensor::CreateScalar(not_expected[i], &not_expected_item);
    EXPECT_EQ(*ind, *expected_item);
    EXPECT_NE(*ind, *not_expected_item);
    iter->GetNextRow(&row);
    i++;
  }
}

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
  int32_t home_index = vocab->Lookup("home");
  EXPECT_EQ(home_index, 2);

  // Create Lookup operation on ds
  std::shared_ptr<TensorOperation> lookup = text::Lookup(vocab, "home", "int64");
  EXPECT_NE(lookup, nullptr);

  // Create Map operation on ds
  ds = ds->Map({lookup});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  std::vector<int64_t> expected = {2, 3, 1, 4, 5, 0};
  std::vector<int8_t> not_expected = {2, 3, 1, 4, 5, 0};
  while (row.size() != 0) {
    auto ind = row["text"];
    MS_LOG(INFO) << ind->shape() << " " << *ind;
    std::shared_ptr<Tensor> expected_item, not_expected_item;
    Tensor::CreateScalar(expected[i], &expected_item);
    Tensor::CreateScalar(not_expected[i], &not_expected_item);
    EXPECT_EQ(*ind, *expected_item);
    EXPECT_NE(*ind, *not_expected_item);
    iter->GetNextRow(&row);
    i++;
  }
}
