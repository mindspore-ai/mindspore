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

#include "minddata/dataset/kernels/data/data_utils.h"
#include "minddata/dataset/text/kernels/lookup_op.h"

namespace mindspore {
namespace dataset {

LookupOp::LookupOp(std::shared_ptr<Vocab> vocab, WordIdType default_id, const DataType &data_type)
    : vocab_(vocab), default_id_(default_id), type_(data_type) {}

Status LookupOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  RETURN_UNEXPECTED_IF_NULL(vocab_);
  CHECK_FAIL_RETURN_UNEXPECTED(input->type() == DataType::DE_STRING, "Lookup: input is not string datatype.");

  std::vector<WordIdType> word_ids;
  word_ids.reserve(input->Size());
  for (auto itr = input->begin<std::string_view>(); itr != input->end<std::string_view>(); itr++) {
    WordIdType word_id = vocab_->Lookup(std::string(*itr));
    word_ids.emplace_back(word_id == Vocab::kNoTokenExists ? default_id_ : word_id);
    CHECK_FAIL_RETURN_UNEXPECTED(word_ids.back() != Vocab::kNoTokenExists,
                                 "Lookup: invalid data, token: \"" + std::string(*itr) +
                                   "\" doesn't exist in vocab and no unknown token is specified.");
  }
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(word_ids, input->shape(), output));

  // type cast to user's requirements if what user wants isn't int32_t
  if ((*output)->type() != type_) {
    CHECK_FAIL_RETURN_UNEXPECTED(type_.IsNumeric(),
                                 "Lookup: Lookup doesn't support string to string lookup. "
                                 "data_type needs to be numeric");
    std::shared_ptr<Tensor> cast_to;
    RETURN_IF_NOT_OK(TypeCast(*output, &cast_to, type_));
    *output = cast_to;
  }

  return Status::OK();
}
Status LookupOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  CHECK_FAIL_RETURN_UNEXPECTED(inputs.size() == NumInput() && outputs.size() == NumOutput(), "size doesn't match.");
  CHECK_FAIL_RETURN_UNEXPECTED(inputs[0] == DataType::DE_STRING, "None String tensor type.");
  outputs[0] = type_;
  return Status::OK();
}

void LookupOp::Print(std::ostream &out) const {
  out << "LookupOp: "
      << "type: " << type_ << "\n default lookup id: " << default_id_ << "\n";
}

}  // namespace dataset
}  // namespace mindspore
