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

#include "minddata/dataset/include/execute.h"
#include "minddata/dataset/include/de_tensor.h"
#include "minddata/dataset/include/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"

namespace mindspore {
namespace dataset {
namespace api {

Execute::Execute(std::shared_ptr<TensorOperation> op) : op_(std::move(op)) {}

std::shared_ptr<tensor::MSTensor> Execute::operator()(std::shared_ptr<tensor::MSTensor> input) {
  // Build the op
  if (op_ == nullptr) {
    MS_LOG(ERROR) << "Input TensorOperation is not valid";
    return nullptr;
  }

  std::shared_ptr<Tensor> de_input = std::dynamic_pointer_cast<tensor::DETensor>(input)->tensor();
  if (de_input == nullptr) {
    MS_LOG(ERROR) << "Input Tensor is not valid";
    return nullptr;
  }
  std::shared_ptr<TensorOp> transform = op_->Build();
  std::shared_ptr<Tensor> de_output;
  Status rc = transform->Compute(de_input, &de_output);

  if (rc.IsError()) {
    // execution failed
    MS_LOG(ERROR) << "Operation execution failed : " << rc.ToString();
    return nullptr;
  }
  return std::make_shared<tensor::DETensor>(std::move(de_output));
}

}  // namespace api
}  // namespace dataset
}  // namespace mindspore
