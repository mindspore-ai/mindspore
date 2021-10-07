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
#include "minddata/dataset/kernels/data/random_choice_op.h"

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

uint32_t RandomChoiceOp::NumInput() {
  uint32_t num_input = ops_.front()->NumInput();
  for (auto &op : ops_) {
    uint32_t cur_num = op->NumInput();
    if (num_input != cur_num && cur_num > 0) {
      MS_LOG(WARNING) << "Unable to determine Num of Input, ops in RandomChoice don't take the same number of input.";
      return 0;
    }
  }
  return num_input;
}

uint32_t RandomChoiceOp::NumOutput() {
  uint32_t num_output = ops_.front()->NumOutput();
  for (auto &op : ops_) {
    uint32_t cur_num = op->NumOutput();
    if (num_output != cur_num) {
      MS_LOG(WARNING) << "Unable to determine NumOutput, ops in RandomChoice don't have the same number of output.";
      return 0;
    }
  }
  return num_output;
}

Status RandomChoiceOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(ops_.front()->OutputShape(inputs, outputs));
  for (auto &op : ops_) {
    std::vector<TensorShape> out_shapes;
    RETURN_IF_NOT_OK(op->OutputShape(inputs, out_shapes));
    if (outputs != out_shapes) {
      MS_LOG(WARNING) << "TensorOp in RandomChoice don't return the same tensorShape.";
      outputs.clear();
      outputs.resize(NumOutput(), TensorShape::CreateUnknownRankShape());
      return Status::OK();
    }
  }
  return Status::OK();
}

Status RandomChoiceOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(ops_.front()->OutputType(inputs, outputs));
  for (auto &op : ops_) {
    std::vector<DataType> out_types;
    RETURN_IF_NOT_OK(op->OutputType(inputs, out_types));
    if (outputs != out_types) {
      MS_LOG(WARNING) << "TensorOp in RandomChoice don't return the same tensorType.";
      outputs.clear();
      outputs.resize(NumOutput(), DataType(DataType::DE_UNKNOWN));
      return Status::OK();
    }
  }
  return Status::OK();
}

Status RandomChoiceOp::Compute(const TensorRow &input, TensorRow *output) {
  size_t rand_num = rand_int_(gen_);
  CHECK_FAIL_RETURN_UNEXPECTED(rand_num < ops_.size(), "invalid rand_num:" + std::to_string(rand_num));
  RETURN_IF_NOT_OK(ops_[rand_num]->Compute(input, output));
  return Status::OK();
}
RandomChoiceOp::RandomChoiceOp(const std::vector<std::shared_ptr<TensorOp>> &ops)
    : ops_(ops), gen_(GetSeed()), rand_int_(0, ops.size() - 1) {
  if (ops_.empty()) {
    MS_LOG(ERROR) << "op_list in RandomChoiceOp is empty.";
  }
  if (ops_.size() == 1) {
    MS_LOG(WARNING) << "op_list has only 1 op, this op would be picked every time.";
  }
  is_deterministic_ = false;
}
}  // namespace dataset
}  // namespace mindspore
