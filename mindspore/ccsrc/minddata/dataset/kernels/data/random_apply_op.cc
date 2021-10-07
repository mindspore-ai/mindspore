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
#include "minddata/dataset/kernels/data/random_apply_op.h"

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

uint32_t RandomApplyOp::NumOutput() {
  if (compose_->NumOutput() != NumInput()) {
    MS_LOG(WARNING) << "NumOutput!=NumInput (randomApply would randomly affect number of outputs).";
    return 0;
  }
  return compose_->NumOutput();
}

Status RandomApplyOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(compose_->OutputShape(inputs, outputs));
  // randomApply either runs all ops or do nothing. If the two methods don't give the same result. return unknown shape.
  if (inputs != outputs) {  // when RandomApply is not applied, input should be the same as output
    outputs.clear();
    outputs.resize(NumOutput(), TensorShape::CreateUnknownRankShape());
  }
  return Status::OK();
}
Status RandomApplyOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(compose_->OutputType(inputs, outputs));
  if (inputs != outputs) {  // when RandomApply is not applied, input should be the same as output
    outputs.clear();
    outputs.resize(NumOutput(), DataType(DataType::DE_UNKNOWN));
  }
  return Status::OK();
}
Status RandomApplyOp::Compute(const TensorRow &input, TensorRow *output) {
  if (rand_double_(gen_) <= prob_) {
    RETURN_IF_NOT_OK(compose_->Compute(input, output));
  } else {
    IO_CHECK_VECTOR(input, output);
    *output = input;  // copy over the tensors
  }
  return Status::OK();
}
RandomApplyOp::RandomApplyOp(const std::vector<std::shared_ptr<TensorOp>> &ops, double prob)
    : prob_(prob), gen_(GetSeed()), rand_double_(0, 1) {
  compose_ = std::make_unique<ComposeOp>(ops);
  is_deterministic_ = false;
}

}  // namespace dataset
}  // namespace mindspore
