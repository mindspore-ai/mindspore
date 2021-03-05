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
#include "minddata/dataset/kernels/tensor_op.h"
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

namespace mindspore {
namespace dataset {
// Name: Compute()
// Description: This Compute() take 1 Tensor and produce 1 Tensor.
//              The derived class should override this function otherwise error.
Status TensorOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  if (!OneToOne()) {
    return Status(StatusCode::kMDUnexpectedError, "Wrong Compute() function is called. This is not 1-1 TensorOp.");
  } else {
    return Status(StatusCode::kMDUnexpectedError,
                  "Is this TensorOp 1-1? If yes, please implement this Compute() in the derived class.");
  }
}

// Name: Compute()
// Description: This Compute() take multiple Tensors from different columns and produce multiple Tensors too.
//              The derived class should override this function otherwise error.
Status TensorOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  if (OneToOne()) {
    CHECK_FAIL_RETURN_UNEXPECTED(input.size() == 1, "The op is OneToOne, can only accept one tensor as input.");
    output->resize(1);
    return Compute(input[0], &(*output)[0]);
  }

  return Status(StatusCode::kMDUnexpectedError,
                "Is this TensorOp oneToOne? If no, please implement this Compute() in the derived class.");
}

Status TensorOp::Compute(const std::shared_ptr<DeviceTensor> &input, std::shared_ptr<DeviceTensor> *output) {
  IO_CHECK(input, output);
  return Status(StatusCode::kMDUnexpectedError,
                "Wrong Compute() function is called. This is a function for operators which can be executed by"
                "different device. If so, please implement it in the derived class.");
}

Status TensorOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  if (inputs.size() != NumInput())
    return Status(StatusCode::kMDUnexpectedError,
                  "The size of the input argument vector does not match the number of inputs");
  outputs = inputs;
  return Status::OK();
}

Status TensorOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  if (inputs.size() != NumInput())
    return Status(StatusCode::kMDUnexpectedError,
                  "The size of the input argument vector does not match the number of inputs");
  outputs = inputs;
  return Status::OK();
}

Status TensorOp::SetAscendResource(const std::shared_ptr<DeviceResource> &resource) {
  return Status(StatusCode::kMDUnexpectedError,
                "This is a CPU operator which doesn't have Ascend Resource. Please verify your context");
}

}  // namespace dataset
}  // namespace mindspore
