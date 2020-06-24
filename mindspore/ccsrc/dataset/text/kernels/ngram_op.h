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

#ifndef DATASET_TEXT_KERNELS_NGRAM_OP_H_
#define DATASET_TEXT_KERNELS_NGRAM_OP_H_

#include <string>
#include <memory>
#include <vector>

#include "dataset/core/tensor.h"
#include "dataset/kernels/tensor_op.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
namespace py = pybind11;

class NgramOp : public TensorOp {
 public:
  // Constructor of Ngram model
  // @param const std::vector<int32_t> &ngrams
  // @param int32_tl_len - padding length on the left
  // @param int32_t r_len - padding length on the right
  // @param const std::string &l_pad - padding token on the left
  // @param const std::string &r_pad - padding token on the right
  // @param const std::string &separator - use to join strings
  NgramOp(const std::vector<int32_t> &ngrams, int32_t l_len, int32_t r_len, const std::string &l_pad,
          const std::string &r_pad, const std::string &separator);

  // perform ngram model on each tensor
  // @param const std::shared_ptr<Tensor> &input
  // @param std::shared_ptr<Tensor> *output
  // @return error code
  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  // destructor
  ~NgramOp() override = default;

  // @param std::vector<TensorShape> &inputs - shape of input tensors
  // @param std::vector<TensorShape> &outputs - shape of output tensors
  // @return error code
  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  // print arg for debugging
  // @param std::ostream &out
  void Print(std::ostream &out) const override;

 private:
  std::vector<int32_t> ngrams_;  // list of n grams
  int32_t l_len_;                // left padding length
  int32_t r_len_;                // right padding length
  std::string l_pad_with_sp_;    // left padding appended with separator
  std::string r_pad_with_sp_;    // right padding appended with separator
  std::string separator_;        // separator
};

}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_TEXT_KERNELS_NGRAM_OP_H_
