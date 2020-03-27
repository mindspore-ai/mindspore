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
#include "dataset/kernels/data/to_float16_op.h"
#include "dataset/core/tensor.h"
#include "dataset/kernels/data/data_utils.h"
#include "dataset/kernels/tensor_op.h"

namespace mindspore {
namespace dataset {
Status ToFloat16Op::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  return ToFloat16(input, output);
}
Status ToFloat16Op::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  outputs[0] = DataType(DataType::DE_FLOAT16);
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
