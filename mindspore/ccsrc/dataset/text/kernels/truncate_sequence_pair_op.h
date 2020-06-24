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
#ifndef DATASET_KERNELS_DATA_TRUNCATE_SEQUENCE_PAIR_OP_H_
#define DATASET_KERNELS_DATA_TRUNCATE_SEQUENCE_PAIR_OP_H_

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dataset/core/tensor.h"
#include "dataset/kernels/tensor_op.h"
#include "dataset/kernels/data/type_cast_op.h"
#include "dataset/kernels/data/data_utils.h"

namespace mindspore {
namespace dataset {

class TruncateSequencePairOp : public TensorOp {
 public:
  explicit TruncateSequencePairOp(dsize_t length) : max_length_(length) {}

  ~TruncateSequencePairOp() override = default;

  void Print(std::ostream &out) const override { out << "TruncateSequencePairOp"; }

  Status Compute(const TensorRow &input, TensorRow *output) override;

 private:
  dsize_t max_length_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // DATASET_KERNELS_DATA_TRUNCATE_SEQUENCE_PAIR_OP_H_
