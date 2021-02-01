/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/core/tensor_row.h"
#ifdef ENABLE_ANDROID
#include "minddata/dataset/include/de_tensor.h"
#endif
#include "minddata/dataset/include/execute.h"
#include "minddata/dataset/include/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#ifndef ENABLE_ANDROID
#include "utils/log_adapter.h"
#else
#include "mindspore/lite/src/common/log_adapter.h"
#endif

namespace mindspore {
namespace dataset {

Execute::Execute(std::shared_ptr<TensorOperation> op) : op_(std::move(op)) {}

/// \brief Destructor
Execute::~Execute() = default;

#ifdef ENABLE_ANDROID
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
#endif

std::shared_ptr<dataset::Tensor> Execute::operator()(std::shared_ptr<dataset::Tensor> input) {
  // Build the op
  if (op_ == nullptr) {
    MS_LOG(ERROR) << "Input TensorOperation is not valid";
    return nullptr;
  }

  if (input == nullptr) {
    MS_LOG(ERROR) << "Input Tensor is not valid";
    return nullptr;
  }
  // will add validate params once API is set
  std::shared_ptr<TensorOp> transform = op_->Build();
  std::shared_ptr<Tensor> de_output;
  Status rc = transform->Compute(input, &de_output);

  if (rc.IsError()) {
    // execution failed
    MS_LOG(ERROR) << "Operation execution failed : " << rc.ToString();
    return nullptr;
  }
  return de_output;
}

Status Execute::operator()(const std::vector<std::shared_ptr<Tensor>> &input_tensor_list,
                           std::vector<std::shared_ptr<Tensor>> *output_tensor_list) {
  CHECK_FAIL_RETURN_UNEXPECTED(op_ != nullptr, "Input TensorOperation is not valid");
  CHECK_FAIL_RETURN_UNEXPECTED(!input_tensor_list.empty(), "Input Tensor is not valid");

  TensorRow input, output;
  std::copy(input_tensor_list.begin(), input_tensor_list.end(), std::back_inserter(input));
  CHECK_FAIL_RETURN_UNEXPECTED(!input.empty(), "Input Tensor is not valid");

  std::shared_ptr<TensorOp> transform = op_->Build();
  Status rc = transform->Compute(input, &output);
  if (rc.IsError()) {
    // execution failed
    RETURN_STATUS_UNEXPECTED("Operation execution failed : " + rc.ToString());
  }

  std::copy(output.begin(), output.end(), std::back_inserter(*output_tensor_list));
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
