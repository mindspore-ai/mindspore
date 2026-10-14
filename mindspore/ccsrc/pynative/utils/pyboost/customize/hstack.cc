/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "include/pynative/utils/pyboost/customize/hstack.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <vector>
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "include/pynative/utils/pyboost/functions/auto_generate/functions.h"

namespace mindspore::kernel::pyboost {
tensor::TensorPtr HstackCustomize(const ValueTuplePtr &tensors) {
  auto inputs = ConvertValueTupleToVector<tensor::TensorPtr>(tensors);
  if (inputs.empty()) {
    MS_EXCEPTION(ValueError) << "For 'hstack', tensors must not be empty.";
  }
  std::vector<ValuePtr> promoted;
  promoted.reserve(inputs.size());
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(promoted), [](const auto &input) {
    return input->shape().empty() ? reshape(input, std::vector<int64_t>{1}) : input;
  });
  const auto first = promoted.front()->cast<tensor::TensorPtr>();
  const int64_t axis = first->shape().size() == 1 ? 0 : 1;
  return concat(std::make_shared<ValueTuple>(promoted), MakeValue<int64_t>(axis)->cast<Int64ImmPtr>());
}
}  // namespace mindspore::kernel::pyboost
