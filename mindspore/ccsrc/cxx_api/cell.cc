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
#include "include/api/cell.h"

namespace mindspore::api {
std::vector<Output> CellBase::operator()(const std::vector<Input> &inputs) const { return Clone()->Construct(inputs); }

ParameterCell::ParameterCell(const ParameterCell &cell) : tensor_(cell.tensor_.Clone()) {}
ParameterCell &ParameterCell::operator=(const ParameterCell &cell) {
  if (&cell == this) {
    return *this;
  }
  tensor_ = cell.tensor_.Clone();
  return *this;
}

ParameterCell::ParameterCell(ParameterCell &&cell) : tensor_(cell.tensor_) {}

ParameterCell &ParameterCell::operator=(ParameterCell &&cell) {
  if (&cell == this) {
    return *this;
  }
  tensor_ = cell.tensor_;
  return *this;
}

ParameterCell::ParameterCell(const Tensor &tensor) : tensor_(tensor.Clone()) {}

ParameterCell &ParameterCell::operator=(const Tensor &tensor) {
  tensor_ = tensor.Clone();
  return *this;
}

ParameterCell::ParameterCell(Tensor &&tensor) : tensor_(tensor) {}

ParameterCell &ParameterCell::operator=(Tensor &&tensor) {
  tensor_ = tensor;
  return *this;
}

InputAndOutput::InputAndOutput() : cell_(nullptr), prev_(), index_(-1) {}

InputAndOutput::InputAndOutput(const Tensor &tensor)
    : cell_(std::make_shared<ParameterCell>(tensor.Clone())), prev_(), index_(-1) {}
InputAndOutput::InputAndOutput(Tensor &&tensor) : cell_(std::make_shared<ParameterCell>(tensor)), prev_(), index_(-1) {}

InputAndOutput::InputAndOutput(const std::shared_ptr<CellBase> &cell, const std::vector<InputAndOutput> &prev,
                               int32_t index)
    : cell_(cell), prev_(prev), index_(index) {}
}  // namespace mindspore::api
