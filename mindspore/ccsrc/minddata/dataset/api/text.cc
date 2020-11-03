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

#include <unistd.h>
#include "minddata/dataset/include/text.h"
#include "minddata/dataset/text/kernels/lookup_op.h"
#include "minddata/dataset/text/kernels/sentence_piece_tokenizer_op.h"
#include "minddata/dataset/util/path.h"

namespace mindspore {
namespace dataset {

// Transform operations for text.
namespace text {

// FUNCTIONS TO CREATE TEXT OPERATIONS
// (In alphabetical order)

std::shared_ptr<LookupOperation> Lookup(const std::shared_ptr<Vocab> &vocab, const std::string &unknown_token,
                                        const DataType &data_type) {
  auto op = std::make_shared<LookupOperation>(vocab, unknown_token, data_type);

  return op->ValidateParams() ? op : nullptr;
}

std::shared_ptr<SentencePieceTokenizerOperation> SentencePieceTokenizer(
  const std::shared_ptr<SentencePieceVocab> &vocab, SPieceTokenizerOutType out_type) {
  auto op = std::make_shared<SentencePieceTokenizerOperation>(vocab, out_type);

  return op->ValidateParams() ? op : nullptr;
}

std::shared_ptr<SentencePieceTokenizerOperation> SentencePieceTokenizer(const std::string &vocab_path,
                                                                        SPieceTokenizerOutType out_type) {
  auto op = std::make_shared<SentencePieceTokenizerOperation>(vocab_path, out_type);

  return op->ValidateParams() ? op : nullptr;
}

/* ####################################### Validator Functions ############################################ */

/* ####################################### Derived TensorOperation classes ################################# */

// (In alphabetical order)

// LookupOperation
LookupOperation::LookupOperation(const std::shared_ptr<Vocab> &vocab, const std::string &unknown_token,
                                 const DataType &data_type)
    : vocab_(vocab), unknown_token_(unknown_token), default_id_(Vocab::kNoTokenExists), data_type_(data_type) {}

Status LookupOperation::ValidateParams() {
  if (vocab_ == nullptr) {
    std::string err_msg = "Lookup: vocab object type is incorrect or null.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  default_id_ = vocab_->Lookup(unknown_token_);
  if (default_id_ == Vocab::kNoTokenExists) {
    std::string err_msg = "Lookup: " + unknown_token_ + " doesn't exist in vocab.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

std::shared_ptr<TensorOp> LookupOperation::Build() {
  std::shared_ptr<LookupOp> tensor_op = std::make_shared<LookupOp>(vocab_, default_id_, data_type_);
  return tensor_op;
}

// SentencePieceTokenizerOperation
SentencePieceTokenizerOperation::SentencePieceTokenizerOperation(const std::shared_ptr<SentencePieceVocab> &vocab,
                                                                 SPieceTokenizerOutType out_type)
    : vocab_(vocab), vocab_path_(std::string()), load_type_(SPieceTokenizerLoadType::kModel), out_type_(out_type) {}

SentencePieceTokenizerOperation::SentencePieceTokenizerOperation(const std::string &vocab_path,
                                                                 SPieceTokenizerOutType out_type)
    : vocab_(nullptr), vocab_path_(vocab_path), load_type_(SPieceTokenizerLoadType::kFile), out_type_(out_type) {}

Status SentencePieceTokenizerOperation::ValidateParams() {
  if (load_type_ == SPieceTokenizerLoadType::kModel) {
    if (vocab_ == nullptr) {
      std::string err_msg = "SentencePieceTokenizer: vocab object type is incorrect or null.";
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  } else {
    Path vocab_file(vocab_path_);
    if (!vocab_file.Exists() || vocab_file.IsDirectory()) {
      std::string err_msg = "SentencePieceTokenizer : vocab file: [" + vocab_path_ + "] is invalid or does not exist.";
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
    if (access(vocab_file.toString().c_str(), R_OK) == -1) {
      std::string err_msg = "SentencePieceTokenizer : no access to specified dataset file: " + vocab_path_;
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> SentencePieceTokenizerOperation::Build() {
  std::shared_ptr<SentencePieceTokenizerOp> tensor_op;
  if (load_type_ == SPieceTokenizerLoadType::kModel) {
    tensor_op = std::make_shared<SentencePieceTokenizerOp>(vocab_, load_type_, out_type_);
  } else {
    Path vocab_file(vocab_path_);
    std::string model_path = vocab_file.ParentPath();
    std::string model_filename = vocab_file.Basename();
    tensor_op = std::make_shared<SentencePieceTokenizerOp>(model_path, model_filename, load_type_, out_type_);
  }
  return tensor_op;
}

}  // namespace text
}  // namespace dataset
}  // namespace mindspore
