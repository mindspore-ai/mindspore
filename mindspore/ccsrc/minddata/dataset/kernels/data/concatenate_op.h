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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_DATA_CONCATENATE_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_DATA_CONCATENATE_OP_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"

namespace mindspore {
namespace dataset {
class ConcatenateOp : public TensorOp {
 public:
  /// Constructor to ConcatenateOp.
  /// @param int8_t axis - axis to concatenate tensors along.
  /// @param std::shared_ptr<Tensor> prepend - prepend tensor.
  /// @param std::shared_ptr<Tensor> append -append tensor.
  ConcatenateOp(int8_t axis, std::shared_ptr<Tensor> prepend, std::shared_ptr<Tensor> append)
      : axis_(axis), prepend_(std::move(prepend)), append_(std::move(append)) {}

  ~ConcatenateOp() override = default;

  /// Compute method allowing multiple tensors as inputs
  /// @param TensorRow &input - input tensor rows
  /// @param TensorRow *output - output tensor rows
  Status Compute(const TensorRow &input, TensorRow *output) override;

  /// Compute tensor output shape
  /// @param std::vector<TensorShape> &inputs - vector of input tensor shapes
  /// @param std::vector<TensorShape< &outputs - vector of output tensor shapes
  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  /// Number of inputs the tensor operation accepts
  uint32_t NumInput() override { return 0; }

  std::string Name() const override { return kConcatenateOp; }

 private:
  int8_t axis_;
  std::shared_ptr<Tensor> prepend_;
  std::shared_ptr<Tensor> append_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CONCATENATE_OP_H
