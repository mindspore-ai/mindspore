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

#include "minddata/dataset/kernels/data/mask_op.h"

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"

namespace mindspore {
namespace dataset {

Status MaskOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  std::shared_ptr<Tensor> temp_output;
  CHECK_FAIL_RETURN_UNEXPECTED(type_.IsNumeric(), "Mask: only support numeric datatype of input.");

  RETURN_IF_NOT_OK(Mask(input, &temp_output, value_, op_));

  // cast the output to the the required type. Skip casting if type_ is bool.
  if (type_ != DataType::DE_BOOL) {
    RETURN_IF_NOT_OK(cast_->Compute(temp_output, output));
  } else {
    *output = std::move(temp_output);
  }

  return Status::OK();
}

Status MaskOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  outputs[0] = type_;
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
