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

#include "include/pynative/utils/pyboost/functions/composite/randint.h"
#include "include/pynative/utils/pyboost/functions/composite/empty.h"
#include "include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "mindspore/core/include/mindapi/base/type_id.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr randint(const Int64ImmPtr low, const Int64ImmPtr high, const ValueTuplePtr &shape,
                          const tensor::TensorPtr &seed, const tensor::TensorPtr &offset,
                          const std::optional<Int64ImmPtr> &dtype, const std::optional<Int64ImmPtr> &device) {
  tensor::TensorPtr output_empty = nullptr;
  if (dtype.has_value()) {
    output_empty = empty(shape, dtype, device);
  } else {
    output_empty = empty(shape, std::make_shared<Int64Imm>(TypeId::kNumberTypeInt64), device);
  }
  tensor::TensorPtr output = inplace_random(output_empty, low, high, seed, offset);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
