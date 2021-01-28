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

#include <string>
#include <string_view>
#include "common/common.h"
#include "minddata/dataset/engine/datasetops/build_sentence_piece_vocab_op.h"
#include "minddata/dataset/text/kernels/sentence_piece_tokenizer_op.h"
#include "minddata/dataset/text/sentence_piece_vocab.h"
#include "minddata/dataset/engine/datasetops/source/text_file_op.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include "minddata/dataset/util/status.h"

using namespace mindspore::dataset;

class MindDataTestSentencePieceVocabOp : public UT::DatasetOpTesting {
 public:
  void CheckEqual(const std::shared_ptr<Tensor> &o, const std::vector<dsize_t> &index, const std::string &expect) {
    std::string_view str;
    Status s = o->GetItemAt(&str, index);
    EXPECT_TRUE(s.IsOk());
    EXPECT_EQ(str, expect);
  }
};

TEST_F(MindDataTestSentencePieceVocabOp, TestSentencePieceFromDatasetFuntions) {
  MS_LOG(INFO) << "Doing MindDataTestSentencePieceVocabOp  TestSentencePieceFromDatasetFuntions.";

  std::string dataset_path;
  dataset_path = datasets_root_path_ + "/test_sentencepiece/botchan.txt";
  auto tree = std::make_shared<ExecutionTree>();

  std::shared_ptr<TextFileOp> file_op;
  TextFileOp::Builder builder_file;
  builder_file.SetTextFilesList({dataset_path}).SetRowsPerBuffer(1).SetNumWorkers(1).SetOpConnectorSize(2);

  Status rc = builder_file.Build(&file_op);
  ASSERT_TRUE(rc.IsOk());

  rc = tree->AssociateNode(file_op);
  ASSERT_TRUE(rc.IsOk());

  std::shared_ptr<SentencePieceVocab> spm = std::make_unique<SentencePieceVocab>();
  std::shared_ptr<BuildSentencePieceVocabOp> spv_op;
  BuildSentencePieceVocabOp::Builder builder_spv;
  std::vector<std::string> cols;
  std::unordered_map<std::string, std::string> m_params;
  builder_spv.SetVocab(spm)
    .SetVocabSize(5000)
    .SetColumnNames(cols)
    .SetCharacterCoverage(0.9995)
    .SetModelType(SentencePieceModel::kUnigram)
    .SetParams(m_params)
    .SetOpConnectorSize(2);

  rc = builder_spv.Build(&spv_op);
  ASSERT_TRUE(rc.IsOk());
  rc = tree->AssociateNode(spv_op);
  ASSERT_TRUE(rc.IsOk());

  rc = spv_op->AddChild(file_op);
  ASSERT_TRUE(rc.IsOk());

  file_op->set_total_repeats(1);
  file_op->set_num_repeats_per_epoch(1);
  rc = tree->AssignRoot(spv_op);
  ASSERT_TRUE(rc.IsOk());
  rc = tree->Prepare();
  ASSERT_TRUE(rc.IsOk());

  rc = tree->Launch();
  ASSERT_TRUE(rc.IsOk());

  // Start the loop of reading tensors from our pipeline
  DatasetIterator di(tree);
  TensorRow tensor_list;
  rc = di.FetchNextTensorRow(&tensor_list);
  ASSERT_TRUE(rc.IsOk());

  while (!tensor_list.empty()) {
    rc = di.FetchNextTensorRow(&tensor_list);
  }
  ASSERT_TRUE(rc.IsOk());
}

TEST_F(MindDataTestSentencePieceVocabOp, TestSentencePieceFromFileFuntions) {
  MS_LOG(INFO) << "Doing MindDataTestSentencePieceVocabOp  TestSentencePieceFromFileFuntions.";

  std::string dataset_path;
  dataset_path = datasets_root_path_ + "/test_sentencepiece/botchan.txt";
  std::vector<std::string> path_list;
  path_list.emplace_back(dataset_path);
  std::unordered_map<std::string, std::string> param_map;
  std::shared_ptr<SentencePieceVocab> spm = std::make_unique<SentencePieceVocab>();
  Status rc = SentencePieceVocab::BuildFromFile(path_list, 5000, 0.9995, SentencePieceModel::kUnigram, param_map, &spm);
  ASSERT_TRUE(rc.IsOk());
}

TEST_F(MindDataTestSentencePieceVocabOp, TestSentencePieceTokenizerFuntions) {
  MS_LOG(INFO) << "Doing MindDataTestSentencePieceVocabOp  TestSentencePieceTokenizerFuntions.";

  std::string dataset_path;
  dataset_path = datasets_root_path_ + "/test_sentencepiece/botchan.txt";
  auto tree = std::make_shared<ExecutionTree>();

  std::shared_ptr<TextFileOp> file_op;
  TextFileOp::Builder builder_file;
  builder_file.SetTextFilesList({dataset_path}).SetRowsPerBuffer(1).SetNumWorkers(1).SetOpConnectorSize(2);

  Status rc = builder_file.Build(&file_op);
  ASSERT_TRUE(rc.IsOk());

  rc = tree->AssociateNode(file_op);
  ASSERT_TRUE(rc.IsOk());

  std::shared_ptr<SentencePieceVocab> spm = std::make_unique<SentencePieceVocab>();
  std::shared_ptr<BuildSentencePieceVocabOp> spv_op;
  BuildSentencePieceVocabOp::Builder builder_spv;
  std::vector<std::string> cols;
  std::unordered_map<std::string, std::string> m_params;

  builder_spv.SetVocab(spm)
    .SetVocabSize(5000)
    .SetColumnNames(cols)
    .SetCharacterCoverage(0.9995)
    .SetModelType(SentencePieceModel::kUnigram)
    .SetParams(m_params)
    .SetOpConnectorSize(2);

  rc = builder_spv.Build(&spv_op);
  ASSERT_TRUE(rc.IsOk());
  rc = tree->AssociateNode(spv_op);
  ASSERT_TRUE(rc.IsOk());

  rc = spv_op->AddChild(file_op);
  ASSERT_TRUE(rc.IsOk());

  file_op->set_total_repeats(1);
  file_op->set_num_repeats_per_epoch(1);
  rc = tree->AssignRoot(spv_op);
  ASSERT_TRUE(rc.IsOk());
  rc = tree->Prepare();
  ASSERT_TRUE(rc.IsOk());

  rc = tree->Launch();
  ASSERT_TRUE(rc.IsOk());

  // Start the loop of reading tensors from our pipeline
  DatasetIterator di(tree);
  TensorRow tensor_list;
  rc = di.FetchNextTensorRow(&tensor_list);
  ASSERT_TRUE(rc.IsOk());

  while (!tensor_list.empty()) {
    rc = di.FetchNextTensorRow(&tensor_list);
  }
  std::shared_ptr<Tensor> output_tensor;
  std::unique_ptr<SentencePieceTokenizerOp> op(
    new SentencePieceTokenizerOp(spm, SPieceTokenizerLoadType::kModel, SPieceTokenizerOutType::kString));
  std::shared_ptr<Tensor> input_tensor;
  Tensor::CreateScalar<std::string>("I saw a girl with a telescope.", &input_tensor);
  Status s = op->Compute(input_tensor, &output_tensor);

  std::vector<std::string> expect;
  expect.push_back("▁I");
  expect.push_back("▁sa");
  expect.push_back("w");
  expect.push_back("▁a");
  expect.push_back("▁girl");
  expect.push_back("▁with");
  expect.push_back("▁a");
  expect.push_back("▁te");
  expect.push_back("les");
  expect.push_back("co");
  expect.push_back("pe");
  expect.push_back(".");
  ASSERT_TRUE(output_tensor->Size() == expect.size());
  for (int i = 0; i < output_tensor->Size(); i++) {
    std::string_view str;
    output_tensor->GetItemAt(&str, {i});
    std::string sentence{str};
    ASSERT_TRUE(sentence == expect[i]);
  }
}