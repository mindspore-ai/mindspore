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

#include <string>
#include <string_view>

#include "common/common.h"
#include "minddata/dataset/engine/datasetops/build_sentence_piece_vocab_op.h"
#include "minddata/dataset/text/kernels/sentence_piece_tokenizer_op.h"
#include "minddata/dataset/include/dataset/text.h"
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

// Testing helper to create TextFileOp
std::shared_ptr<TextFileOp> TextFile(std::vector<std::string> text_files_list, int32_t num_workers,
                                     int32_t op_connector_size) {
  std::unique_ptr<DataSchema> schema = std::make_unique<DataSchema>();
  (void)schema->AddColumn(ColDescriptor("text", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1));
  std::shared_ptr<ConfigManager> config_manager = GlobalContext::config_manager();
  auto worker_connector_size = config_manager->worker_connector_size();
  int32_t device_id = 0;
  int32_t num_devices = 1;
  int32_t num_rows = 0;
  bool shuffle = false;

  std::shared_ptr<TextFileOp> text_file_op =
    std::make_shared<TextFileOp>(num_workers, num_rows, worker_connector_size, std::move(schema), text_files_list,
                                 op_connector_size, shuffle, num_devices, device_id);
  (void)text_file_op->Init();
  return text_file_op;
}

/// Feature: SentencePieceVocab
/// Description: Test SentencePieceVocab basic usage
/// Expectation: Runs successfully
TEST_F(MindDataTestSentencePieceVocabOp, TestSentencePieceFromFileFuntions) {
  MS_LOG(INFO) << "Doing MindDataTestSentencePieceVocabOp  TestSentencePieceFromFileFuntions.";

  std::string dataset_path;
  dataset_path = datasets_root_path_ + "/test_sentencepiece/vocab.txt";
  std::vector<std::string> path_list;
  path_list.emplace_back(dataset_path);
  std::unordered_map<std::string, std::string> param_map;
  std::shared_ptr<SentencePieceVocab> spm = std::make_unique<SentencePieceVocab>();
  Status rc = SentencePieceVocab::BuildFromFile(path_list, 100, 0.9995, SentencePieceModel::kUnigram, param_map, &spm);
  ASSERT_TRUE(rc.IsOk());
}
