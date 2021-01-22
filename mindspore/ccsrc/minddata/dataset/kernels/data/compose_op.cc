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
#include "minddata/dataset/kernels/data/compose_op.h"

#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

Status ComposeOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  std::vector<TensorShape> in_shapes = inputs;
  for (auto &op : ops_) {
    RETURN_IF_NOT_OK(op->OutputShape(in_shapes, outputs));
    in_shapes = std::move(outputs);  // outputs become empty after move
  }
  outputs = std::move(in_shapes);
  return Status::OK();
}
Status ComposeOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  std::vector<DataType> in_types = inputs;
  for (auto &op : ops_) {
    RETURN_IF_NOT_OK(op->OutputType(in_types, outputs));
    in_types = std::move(outputs);  // outputs become empty after move
  }
  outputs = std::move(in_types);
  return Status::OK();
}
Status ComposeOp::Compute(const TensorRow &inputs, TensorRow *outputs) {
  IO_CHECK_VECTOR(inputs, outputs);
  TensorRow in_rows = inputs;
  for (auto &op : ops_) {
    RETURN_IF_NOT_OK(op->Compute(in_rows, outputs));
    in_rows = std::move(*outputs);  // after move, *outputs become empty
  }
  (*outputs) = std::move(in_rows);
  return Status::OK();
}

ComposeOp::ComposeOp(const std::vector<std::shared_ptr<TensorOp>> &ops) : ops_(ops) {
  if (ops_.empty()) {
    MS_LOG(ERROR) << "Compose: op_list is empty, this might lead to Segmentation Fault.";
  } else if (ops_.size() == 1) {
    MS_LOG(WARNING) << "Compose: op_list has only 1 op. Compose is probably not needed.";
  }
}

}  // namespace dataset
}  // namespace mindspore
