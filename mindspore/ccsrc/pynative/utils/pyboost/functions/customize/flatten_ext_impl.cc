
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

#include "mindspore/ccsrc/pynative/utils/pyboost/functions/customize/view_impl.h"
#include "view/view_strides_calculator.h"

namespace mindspore::kernel::pyboost {
mindspore::tensor::TensorPtr flatten_ext_impl(const mindspore::tensor::TensorPtr &input, const int64_t &start_dim,
                                              const int64_t &end_dim) {
  const auto &input_shape = input->shape();
  const int64_t ndim = static_cast<int64_t>(input_shape.size());
  auto start = ops::DynamicDimWrap(start_dim, ndim, true);
  auto end = ops::DynamicDimWrap(end_dim, ndim, true);
  if (MS_UNLIKELY(start > end)) {
    MS_EXCEPTION(ValueError) << "For 'flatten', 'start_dim' cannot come after 'end_dim'.";
  }

  if (ndim == 0) {
    return reshape_impl(input, {1});
  }
  if (start == end) {
    return input;
  }

  int64_t slice_numel =
    std::accumulate(input_shape.begin() + start, input_shape.begin() + end + 1, int64_t(1), std::multiplies<int64_t>());
  std::vector<int64_t> out_shape;
  out_shape.reserve(ndim - end + start);
  for (int64_t i = 0; i < start; i++) {
    out_shape.push_back(input_shape[i]);
  }
  out_shape.push_back(slice_numel);
  for (int64_t i = end + 1; i < ndim; i++) {
    out_shape.push_back(input_shape[i]);
  }
  return reshape_impl(input, out_shape);
}
}  // namespace mindspore::kernel::pyboost
