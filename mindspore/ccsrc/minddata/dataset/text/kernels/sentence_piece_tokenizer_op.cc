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
#include "minddata/dataset/text/kernels/sentence_piece_tokenizer_op.h"

#include <memory>
#include <vector>

#include "utils/ms_utils.h"
#include "minddata/dataset/util/path.h"

namespace mindspore {
namespace dataset {

SentencePieceTokenizerOp::SentencePieceTokenizerOp(const std::shared_ptr<SentencePieceVocab> vocab,
                                                   const SPieceTokenizerLoadType load_type,
                                                   const SPieceTokenizerOutType out_type)
    : vocab_(vocab), load_type_(load_type), out_type_(out_type) {
  auto status = processor_.LoadFromSerializedProto(vocab_.get()->model_proto());
  if (!status.ok()) {
    model_status_ = Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, "parser vocab model filed.");
  } else {
    model_status_ = Status::OK();
  }
}

SentencePieceTokenizerOp::SentencePieceTokenizerOp(const std::string &model_path, const std::string &model_filename,
                                                   const SPieceTokenizerLoadType load_type,
                                                   const SPieceTokenizerOutType out_type)
    : load_type_(load_type), out_type_(out_type) {
  (void)GetModelRealPath(model_path, model_filename);
  auto status = processor_.Load(file_path_);
  if (!status.ok()) {
    model_status_ = Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, "load vocab model filed.");
  } else {
    model_status_ = Status::OK();
  }
}

Status SentencePieceTokenizerOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  if (!model_status_.IsOk()) {
    return model_status_;
  }

  if (input->Rank() != 0 || input->type() != DataType::DE_STRING) {
    RETURN_STATUS_UNEXPECTED("the input tensor should be scalar string tensor");
  }

  std::string_view sentence_v;
  RETURN_IF_NOT_OK(input->GetItemAt(&sentence_v, {}));
  std::string sentence{sentence_v};

  if (out_type_ == SPieceTokenizerOutType::kString) {
    std::vector<std::string> pieces;
    auto status = processor_.Encode(sentence, &pieces);
    if (!status.ok()) {
      RETURN_STATUS_UNEXPECTED("sentence piece tokenizer error");
    }
    RETURN_IF_NOT_OK(Tensor::CreateFromVector(pieces, output));
  } else {
    std::vector<int> ids;
    auto status = processor_.Encode(sentence, &ids);
    if (!status.ok()) {
      RETURN_STATUS_UNEXPECTED("sentence piece tokenizer error");
    }
    RETURN_IF_NOT_OK(Tensor::CreateFromVector(ids, output));
  }
  return Status::OK();
}

Status SentencePieceTokenizerOp::GetModelRealPath(const std::string &model_path, const std::string &filename) {
  char real_path[PATH_MAX] = {0};
  if (file_path_.size() >= PATH_MAX) {
    RETURN_STATUS_UNEXPECTED("sentence piece model path is invalid.");
  }
#if defined(_WIN32) || defined(_WIN64)
  if (_fullpath(real_path, common::SafeCStr(model_path), PATH_MAX) == nullptr) {
    RETURN_STATUS_UNEXPECTED("sentence piece model path is invalid.");
  }
#else
  if (realpath(common::SafeCStr(model_path), real_path) == nullptr) {
    RETURN_STATUS_UNEXPECTED("sentence piece model path  is invalid.");
  }
#endif
  std::string abs_path = real_path;
  file_path_ = (Path(abs_path) / Path(filename)).toString();
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
