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
#include "minddata/dataset/kernels/data/type_cast_op.h"

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/data/data_utils.h"
#include "minddata/dataset/kernels/tensor_op.h"

namespace mindspore {
namespace dataset {
TypeCastOp::TypeCastOp(const DataType &new_type) : type_(new_type) {}

TypeCastOp::TypeCastOp(const std::string &data_type) { type_ = DataType(data_type); }

Status TypeCastOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  return TypeCast(input, output, type_);
}
Status TypeCastOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputType(inputs, outputs));
  outputs[0] = type_;
  return Status::OK();
}

Status TypeCastOp::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["data_type"] = type_.ToString();
  *out_json = args;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
