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
#include "minddata/dataset/kernels/image/random_select_subpolicy_op.h"

#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

Status RandomSelectSubpolicyOp::Compute(const TensorRow &input, TensorRow *output) {
  TensorRow in_row = input;
  size_t rand_num = rand_int_(gen_);
  CHECK_FAIL_RETURN_UNEXPECTED(rand_num < policy_.size(),
                               "RandomSelectSubpolicy: "
                               "get rand number failed:" +
                                 std::to_string(rand_num));
  for (auto &sub : policy_[rand_num]) {
    if (rand_double_(gen_) <= sub.second) {
      RETURN_IF_NOT_OK(sub.first->Compute(in_row, output));
      in_row = std::move(*output);
    }
  }
  *output = std::move(in_row);
  return Status::OK();
}

uint32_t RandomSelectSubpolicyOp::NumInput() {
  uint32_t num_in = policy_.front().front().first->NumInput();
  for (auto &sub : policy_) {
    for (auto p : sub) {
      if (num_in != p.first->NumInput()) {
        MS_LOG(WARNING) << "Unable to determine numInput.";
        return 0;
      }
    }
  }
  return num_in;
}

uint32_t RandomSelectSubpolicyOp::NumOutput() {
  uint32_t num_out = policy_.front().front().first->NumOutput();
  for (auto &sub : policy_) {
    for (auto p : sub) {
      if (num_out != p.first->NumOutput()) {
        MS_LOG(WARNING) << "Unable to determine numInput.";
        return 0;
      }
    }
  }
  return num_out;
}

Status RandomSelectSubpolicyOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  outputs.clear();
  outputs.resize(NumOutput(), TensorShape::CreateUnknownRankShape());
  return Status::OK();
}

Status RandomSelectSubpolicyOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(policy_.front().front().first->OutputType(inputs, outputs));
  for (auto &sub : policy_) {
    for (auto p : sub) {
      std::vector<DataType> tmp_types;
      RETURN_IF_NOT_OK(p.first->OutputType(inputs, tmp_types));
      if (outputs != tmp_types) {
        outputs.clear();
        outputs.resize(NumOutput(), DataType(DataType::DE_UNKNOWN));
        return Status::OK();
      }
    }
  }
  return Status::OK();
}
RandomSelectSubpolicyOp::RandomSelectSubpolicyOp(const std::vector<Subpolicy> &policy)
    : gen_(GetSeed()), policy_(policy), rand_int_(0, policy.size() - 1), rand_double_(0, 1) {
  if (policy_.empty()) {
    MS_LOG(ERROR) << "RandomSelectSubpolicy: policy in RandomSelectSubpolicyOp is empty.";
  }
  is_deterministic_ = false;
}

}  // namespace dataset
}  // namespace mindspore
