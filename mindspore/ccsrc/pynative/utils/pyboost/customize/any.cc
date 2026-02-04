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

#include "include/pynative/utils/pyboost/customize/any.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <vector>

#include "mindspore/ops/ops_utils/op_constants.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr AnyCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                               const std::optional<Int64ImmPtr> &dim, const std::optional<BoolImmPtr> &keepdim) {
  ValueTuplePtr axis{nullptr};
  if (dim.has_value()) {
    axis = std::make_shared<ValueTuple>(std::vector<ValuePtr>{dim.value()});
  } else {
    static const auto static_axis = std::make_shared<ValueTuple>(std::vector<ValuePtr>{});
    axis = static_axis;
  }

  static const auto static_false = std::make_shared<BoolImm>(false);
  auto keep_dims = keepdim.value_or(static_false);

  auto out = reduce_any(input_tensor, axis, keep_dims);
  op->set_outputs({out});
  return out;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
