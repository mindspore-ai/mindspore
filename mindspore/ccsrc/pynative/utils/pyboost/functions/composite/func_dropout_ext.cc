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

#include "include/pynative/utils/pyboost/functions/composite/func_dropout_ext.h"

#include <algorithm>
#include <vector>

#include "include/pynative/utils/pyboost/functions/auto_generate/functions.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr func_dropout_ext(const mindspore::tensor::TensorPtr &input, const mindspore::FP32ImmPtr &p,
                                   const mindspore::BoolImmPtr &training, const mindspore::BoolImmPtr &inplace,
                                   const mindspore::tensor::TensorPtr &seed,
                                   const mindspore::tensor::TensorPtr &offset) {
  auto p_value = static_cast<double>(p->value());
  if (p_value < 0 || p_value > 1) {
    MS_EXCEPTION(ValueError) << "For FuncDropoutExt, 'p' must be in [0, 1], but got " << p_value;
  }

  auto training_ = training->value();
  auto inplace_ = inplace->value();
  auto input_shape = input->shape();
  if (p_value == 0 || !training_ ||
      std::any_of(input_shape.begin(), input_shape.end(), [](const auto &dim) { return dim == 0; })) {
    return input;
  }

  if (p_value == 1) {
    auto input_typeid = input->Dtype()->type_id();
    auto zeros_shape = MakeValue<std::vector<int64_t>>(input_shape)->cast<ValueTuplePtr>();
    auto zeros_tensor = zeros(zeros_shape, std::make_shared<Int64Imm>(static_cast<int64_t>(input_typeid)));
    if (inplace_) {
      return inplace_mul(input, zeros_tensor);
    }
    return mul(input, zeros_tensor);
  }

  auto results = dropout_ext(input, p, seed, offset);
  auto output = std::get<0>(results);
  if (inplace_) {
    return inplace_copy(input, output, std::make_shared<BoolImm>(false));
  }
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
