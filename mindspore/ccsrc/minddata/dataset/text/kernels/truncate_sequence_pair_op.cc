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

#include "minddata/dataset/text/kernels/truncate_sequence_pair_op.h"

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/kernels/data/slice_op.h"

namespace mindspore {
namespace dataset {

Status TruncateSequencePairOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  CHECK_FAIL_RETURN_UNEXPECTED(input.size() == 2, "TruncateSequencePair: Expected two inputs.");
  std::shared_ptr<Tensor> seq1 = input[0];
  std::shared_ptr<Tensor> seq2 = input[1];
  CHECK_FAIL_RETURN_UNEXPECTED(seq1->shape().Rank() == 1 && seq2->shape().Rank() == 1,
                               "TruncateSequencePair: both data column should be of rank 1.");
  dsize_t length1 = seq1->shape()[0];
  dsize_t length2 = seq2->shape()[0];
  dsize_t outLength1 = length1;
  dsize_t outLength2 = length2;

  dsize_t total = length1 + length2;
  while (total > max_length_) {
    if (outLength1 > outLength2)
      outLength1--;
    else
      outLength2--;
    total--;
  }
  std::shared_ptr<Tensor> outSeq1;
  if (length1 != outLength1) {
    std::unique_ptr<SliceOp> slice1(new SliceOp(Slice(outLength1 - length1)));
    RETURN_IF_NOT_OK(slice1->Compute(seq1, &outSeq1));
  } else {
    outSeq1 = std::move(seq1);
  }

  std::shared_ptr<Tensor> outSeq2;
  if (length2 != outLength2) {
    std::unique_ptr<SliceOp> slice2(new SliceOp(Slice(outLength2 - length2)));
    RETURN_IF_NOT_OK(slice2->Compute(seq2, &outSeq2));
  } else {
    outSeq2 = std::move(seq2);
  }
  output->push_back(outSeq1);
  output->push_back(outSeq2);
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
