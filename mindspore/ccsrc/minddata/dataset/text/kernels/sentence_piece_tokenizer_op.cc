/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/text/kernels/sentence_piece_tokenizer_op.h"

#include "minddata/dataset/util/path.h"
#include "utils/file_utils.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace dataset {
SentencePieceTokenizerOp::SentencePieceTokenizerOp(const std::shared_ptr<SentencePieceVocab> vocab,
                                                   const SPieceTokenizerLoadType load_type,
                                                   const SPieceTokenizerOutType out_type)
    : vocab_(vocab), load_type_(load_type), out_type_(out_type) {
  auto status = processor_.LoadFromSerializedProto(vocab_.get()->model_proto());
  if (!status.ok()) {
    model_status_ = STATUS_ERROR(StatusCode::kMDUnexpectedError, "SentencePieceTokenizer: parser vocab model filed.");
  } else {
    model_status_ = Status::OK();
  }
}

SentencePieceTokenizerOp::SentencePieceTokenizerOp(const std::string &model_path, const std::string &model_filename,
                                                   const SPieceTokenizerLoadType load_type,
                                                   const SPieceTokenizerOutType out_type)
    : load_type_(load_type), out_type_(out_type) {
  file_path_ = (Path(model_path) / Path(model_filename)).ToString();
  auto status = processor_.Load(file_path_);
  if (!status.ok()) {
    std::string err_msg = "SentencePieceTokenizer: ";
    err_msg += "load vocab model file: " + file_path_ + " failed.";
    model_status_ = STATUS_ERROR(StatusCode::kMDUnexpectedError, err_msg);
  } else {
    model_status_ = Status::OK();
  }
}

Status SentencePieceTokenizerOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  if (!model_status_.IsOk()) {
    RETURN_STATUS_UNEXPECTED(model_status_.GetErrDescription());
  }

  if (input->Rank() != 0 || input->type() != DataType::DE_STRING) {
    RETURN_STATUS_UNEXPECTED(
      "SentencePieceTokenizer: the input shape should be scalar and the input datatype should be string.");
  }

  std::string_view sentence_v;
  RETURN_IF_NOT_OK(input->GetItemAt(&sentence_v, {}));
  std::string sentence{sentence_v};

  if (out_type_ == SPieceTokenizerOutType::kString) {
    std::vector<std::string> pieces;
    auto status = processor_.Encode(sentence, &pieces);
    if (!status.ok()) {
      RETURN_STATUS_UNEXPECTED("SentencePieceTokenizer: Encode sentence failed.");
    }
    RETURN_IF_NOT_OK(Tensor::CreateFromVector(pieces, output));
  } else {
    std::vector<int> ids;
    auto status = processor_.Encode(sentence, &ids);
    if (!status.ok()) {
      RETURN_STATUS_UNEXPECTED("SentencePieceTokenizer: Encode sentence failed.");
    }
    RETURN_IF_NOT_OK(Tensor::CreateFromVector(ids, output));
  }
  return Status::OK();
}

Status SentencePieceTokenizerOp::GetModelRealPath(const std::string &model_path, const std::string &filename) {
  auto realpath = FileUtils::GetRealPath(model_path.c_str());
  if (!realpath.has_value()) {
    RETURN_STATUS_UNEXPECTED(
      "SentencePieceTokenizer: Sentence piece model path is not existed or permission denied. Model path: " +
      model_path);
  }

  file_path_ = (Path(realpath.value()) / Path(filename)).ToString();
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
