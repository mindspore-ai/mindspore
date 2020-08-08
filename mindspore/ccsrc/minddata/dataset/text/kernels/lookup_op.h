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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_KERNELS_LOOKUP_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_KERNELS_LOOKUP_OP_H_

#include <memory>
#include <vector>
#include <utility>
#include <string>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/text/vocab.h"

namespace mindspore {
namespace dataset {
class LookupOp : public TensorOp {
 public:
  // constructor for lookup, takes in a vocab object
  // @param std::shared_ptr<Vocab> vocab -
  // @param WordIdType default_id, id to lookup if a word is not in vocab
  explicit LookupOp(std::shared_ptr<Vocab> vocab, WordIdType default_id = 1);

  ~LookupOp() = default;

  // perform actual lookup on each tensor
  // @param const std::shared_ptr<Tensor> &input
  // @param std::shared_ptr<Tensor> *output
  // @return error code
  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  // print method
  // @param std::ostream out
  void Print(std::ostream &out) const override;

  // @param std::vector<DataType> &inputs -
  // @param std::vector<DataType> &outputs -
  // @return error code
  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

  std::string Name() const override { return kLookupOp; }

 private:
  std::shared_ptr<Vocab> vocab_;
  WordIdType default_id_;
  DataType type_;  // type of tensor after lookup
};

}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_KERNELS_LOOKUP_OP_H_
