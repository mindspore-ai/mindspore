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

#include "minddata/dataset/text/kernels/ngram_op.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace mindspore {
namespace dataset {

NgramOp::NgramOp(const std::vector<int32_t> &ngrams, int32_t l_len, int32_t r_len, const std::string &l_pad,
                 const std::string &r_pad, const std::string &separator)
    : ngrams_(ngrams),
      l_len_(l_len),
      r_len_(r_len),
      l_pad_with_sp_(l_pad + separator),
      r_pad_with_sp_(r_pad + separator),
      separator_(separator) {}

Status NgramOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  CHECK_FAIL_RETURN_UNEXPECTED(input->type() == DataType::DE_STRING && input->Rank() == 1,
                               "Ngram: input is not a 1D data with string datatype.");
  std::vector<int32_t> offsets;                 // offsets for each str
  std::vector<std::string> res;                 // holds the result of ngrams
  std::string str_buffer;                       // concat all pad tokens with string interleaved with separators
  res.reserve(input->shape().NumOfElements());  // this should be more than enough
  offsets.reserve(1 + l_len_ + r_len_ + input->shape().NumOfElements());
  str_buffer.reserve(l_pad_with_sp_.size() * l_len_ + r_pad_with_sp_.size() * r_len_ + input->SizeInBytes());
  offsets.push_back(str_buffer.size());  // insert 0 as the starting pos
  for (int i = 0; i < l_len_; i++) offsets.push_back((str_buffer += l_pad_with_sp_).size());

  for (auto itr = input->begin<std::string_view>(); itr != input->end<std::string_view>(); itr++) {
    str_buffer += (*itr);
    str_buffer += separator_;
    offsets.push_back(str_buffer.size());
  }

  for (int i = 0; i < r_len_; i++) offsets.push_back((str_buffer += r_pad_with_sp_).size());

  for (auto n : ngrams_) {
    CHECK_FAIL_RETURN_UNEXPECTED(n > 0, "Ngram: ngrams needs to be a positive number.\n");
    int32_t start_ind = l_len_ - std::min(l_len_, n - 1);
    int32_t end_ind = offsets.size() - r_len_ + std::min(r_len_, n - 1);
    if (end_ind - start_ind <= n) {
      res.emplace_back(std::string());  // push back empty string
    } else {
      CHECK_FAIL_RETURN_UNEXPECTED(end_ind - n >= 0, "Ngram: get offsets failed.");

      for (int i = start_ind; i < end_ind - n; i++) {
        res.emplace_back(str_buffer.substr(offsets[i], offsets[i + n] - offsets[i] - separator_.size()));
      }
    }
  }
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(res, TensorShape({static_cast<dsize_t>(res.size())}), output));
  return Status::OK();
}

void NgramOp::Print(std::ostream &out) const {
  out << "NgramOp: "
      << "left pad width: " << l_len_ << " left pad token with separator: " << l_pad_with_sp_ << "\n"
      << "right pad width: " << r_len_ << " right pad token with separator: " << r_pad_with_sp_ << "\n"
      << "separator: " << separator_ << "\n";
}

Status NgramOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  CHECK_FAIL_RETURN_UNEXPECTED(inputs.size() == NumInput(), "Ngram: incorrect num of inputs\n");
  CHECK_FAIL_RETURN_UNEXPECTED(inputs[0].Rank() == 1, "Ngram: ngram only works with 1-dim data\n");
  dsize_t num_elements = ngrams_.size();
  for (int32_t n : ngrams_) {
    // here since rank == 1, NumOfElements == shape[0]. add padding length to string
    int32_t len_with_padding = inputs[0].NumOfElements() + std::min(n - 1, l_len_) + std::min(n - 1, r_len_);
    // if len_with_padding - n < 0, this would return an empty string
    num_elements += std::max(len_with_padding - n, 0);
  }
  outputs.emplace_back(TensorShape({num_elements}));
  CHECK_FAIL_RETURN_UNEXPECTED(outputs.size() == NumOutput(), "Ngram: incorrect num of outputs\n");
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
