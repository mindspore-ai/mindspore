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

#ifndef MINDDATA_TOFLOAT16OP_H
#define MINDDATA_TOFLOAT16OP_H

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "dataset/core/tensor.h"
#include "dataset/kernels/tensor_op.h"

namespace mindspore {
namespace dataset {
class ToFloat16Op : public TensorOp {
 public:
  ToFloat16Op() = default;

  ~ToFloat16Op() override = default;

  // Overrides the base class compute function
  // Calls the ToFloat16 function in ImageUtils, this function takes an input tensor
  // and transforms its data to float16, the output memory is manipulated to contain the result
  // @return Status - The error code return
  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  void Print(std::ostream &out) const override { out << "ToFloat16Op"; }

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDDATA_TOFLOAT16OP_H
