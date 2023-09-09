/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include <memory>
#include <vector>

namespace mindspore {
namespace dataset {
// Name: Compute()
// Description: This Compute() take 1 Tensor and produce 1 Tensor.
//              The derived class should override this function otherwise error.
Status TensorOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  if (!OneToOne()) {
    RETURN_STATUS_UNEXPECTED("Wrong Compute() function is called. This is not 1-1 TensorOp.");
  } else {
    RETURN_STATUS_UNEXPECTED("Is this TensorOp 1-1? If yes, please implement this Compute() in the derived class.");
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

  RETURN_STATUS_UNEXPECTED("Is this TensorOp oneToOne? If no, please implement this Compute() in the derived class.");
}

Status TensorOp::Compute(const std::shared_ptr<DeviceTensor> &input, std::shared_ptr<DeviceTensor> *output) {
  IO_CHECK(input, output);
  RETURN_STATUS_UNEXPECTED(
    "Wrong Compute() function is called. This is a function for operators which can be executed by"
    " Ascend310 device. If so, please implement it in the derived class.");
}

#if !defined(BUILD_LITE) && defined(ENABLE_D)
Status TensorOp::Compute(const std::vector<std::shared_ptr<DeviceTensorAscend910B>> &input,
                         std::vector<std::shared_ptr<DeviceTensorAscend910B>> *output) {
  IO_CHECK_VECTOR(input, output);
  if (OneToOne()) {
    CHECK_FAIL_RETURN_UNEXPECTED(input.size() == 1, "The op is OneToOne, can only accept one tensor as input.");
    output->resize(1);
    return Compute(input[0], &(*output)[0]);
  }

  RETURN_STATUS_UNEXPECTED("Is this TensorOp oneToOne? If no, please implement this Compute() in the derived class.");
}

Status TensorOp::Compute(const std::shared_ptr<DeviceTensorAscend910B> &input,
                         std::shared_ptr<DeviceTensorAscend910B> *output) {
  IO_CHECK(input, output);
  RETURN_STATUS_UNEXPECTED(
    "Wrong Compute() function is called. This is a function for operators which can be executed by"
    " Ascend910B device. If so, please implement it in the derived class.");
}
#endif

Status TensorOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  if (inputs.size() != NumInput()) {
    RETURN_STATUS_UNEXPECTED("The size of the input argument vector does not match the number of inputs");
  }
  outputs = inputs;
  return Status::OK();
}

Status TensorOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  if (inputs.size() != NumInput()) {
    RETURN_STATUS_UNEXPECTED("The size of the input argument vector does not match the number of inputs");
  }
  outputs = inputs;
  return Status::OK();
}

Status TensorOp::SetAscendResource(const std::shared_ptr<DeviceResource> &resource) {
  RETURN_STATUS_UNEXPECTED("This is a CPU operator which doesn't have Ascend Resource. Please verify your context");
}
}  // namespace dataset
}  // namespace mindspore
