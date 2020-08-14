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

#include "minddata/dataset/include/text.h"
#include "minddata/dataset/text/kernels/lookup_op.h"

namespace mindspore {
namespace dataset {
namespace api {
namespace text {

std::shared_ptr<LookupOperation> Lookup(const std::shared_ptr<Vocab> &vocab, const std::string &unknown_token) {
  auto op = std::make_shared<LookupOperation>(vocab, unknown_token);

  if (!op->ValidateParams()) {
    return nullptr;
  }
  return op;
}

// LookupOperation
LookupOperation::LookupOperation(const std::shared_ptr<Vocab> &vocab, const std::string &unknown_token)
    : vocab_(vocab), unknown_token_(unknown_token), default_id_(Vocab::kNoTokenExists) {}

bool LookupOperation::ValidateParams() {
  if (vocab_ == nullptr) {
    LOG(ERROR) << "Lookup: vocab object type is incorrect or null.";
    return false;
  }
  if (unknown_token_.empty()) {
    LOG(ERROR) << "Lookup: no unknown token is specified.";
    return false;
  } else {
    default_id_ = vocab_->Lookup(unknown_token_);
    if (default_id_ == Vocab::kNoTokenExists) {
      LOG(ERROR) << "Lookup: unknown_token: [" + unknown_token_ + "], does not exist in vocab.";
      return false;
    }
  }
  return true;
}

std::shared_ptr<TensorOp> LookupOperation::Build() {
  std::shared_ptr<LookupOp> tensor_op = std::make_shared<LookupOp>(vocab_, default_id_);
  return tensor_op;
}

}  // namespace text
}  // namespace api
}  // namespace dataset
}  // namespace mindspore
