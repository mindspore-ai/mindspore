/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/data/one_hot_op.h"

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/data/data_utils.h"
#include "minddata/dataset/kernels/tensor_op.h"

namespace mindspore {
namespace dataset {
Status OneHotOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  Status s = OneHotEncoding(input, output, num_classes_);
  return s;
}

Status OneHotOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  std::vector<TensorShape> inputs_copy;
  inputs_copy.push_back(inputs[0].Squeeze());
  if (inputs_copy[0].Rank() == 0) outputs.emplace_back(std::vector<dsize_t>{num_classes_});
  if (inputs_copy[0].Rank() == 1) outputs.emplace_back(std::vector<dsize_t>{inputs_copy[0][0], num_classes_});
  if (!outputs.empty()) return Status::OK();
  return Status(StatusCode::kMDUnexpectedError, "OneHot: invalid input shape.");
}

Status OneHotOp::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["num_classes"] = num_classes_;
  *out_json = args;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
