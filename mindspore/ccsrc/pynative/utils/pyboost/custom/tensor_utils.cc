/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "include/pynative/utils/pyboost/custom/tensor_utils.h"
#include <memory>
#include "ir/tensor.h"
#include "ir/dtype.h"
#include "ir/tensor_new.h"
#include "include/utils/visible.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"

namespace ms {
namespace {
inline mindspore::Int64ImmPtr MakeI64Value(int64_t v) { return std::make_shared<mindspore::Int64Imm>(v); }
inline mindspore::ValueTuplePtr MakeI64Tuple(const std::vector<int64_t> &v) {
  return mindspore::MakeValue(v)->cast<mindspore::ValueTuplePtr>();
}
}  // namespace

Tensor tensor(int64_t value, TypeId dtype) {
  return Tensor(mindspore::tensor::from_scalar(value, TypeIdToType(dtype)));
}

Tensor tensor(const std::vector<int64_t> &value, TypeId dtype) {
  return Tensor(mindspore::tensor::from_vector(value, TypeIdToType(dtype)));
}

Tensor tensor(double value, TypeId dtype) { return Tensor(mindspore::tensor::from_scalar(value, TypeIdToType(dtype))); }

Tensor tensor(const std::vector<double> &value, TypeId dtype) {
  return Tensor(mindspore::tensor::from_vector(value, TypeIdToType(dtype)));
}

Tensor ones(const ShapeVector &shape, TypeId dtype) {
  return Tensor(mindspore::kernel::pyboost::ones(MakeI64Tuple(shape), MakeI64Value(dtype)));
}

Tensor zeros(const ShapeVector &shape, TypeId dtype) {
  return Tensor(mindspore::kernel::pyboost::zeros(MakeI64Tuple(shape), MakeI64Value(dtype)));
}
}  // namespace ms
