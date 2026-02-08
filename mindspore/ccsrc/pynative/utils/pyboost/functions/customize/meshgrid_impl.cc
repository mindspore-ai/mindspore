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

#include <algorithm>
#include <iterator>
#include "mindspore/ccsrc/pynative/utils/pyboost/functions/customize/view_impl.h"
#include "include/op_enum.h"
#include "mindspore/core/include/ops/op_def_utils.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"

namespace mindspore::kernel::pyboost {
std::vector<tensor::TensorPtr> meshgrid_impl(const ValueTuplePtr &tensors_list, const int64_t &indexing_imm) {
  MS_LOG(DEBUG) << "Meshgrid call start";

  std::vector<TensorPtr> tensors_list_vector = ConvertValueTupleToVector<tensor::TensorPtr>(tensors_list);
  MS_CHECK_VALUE(tensors_list_vector.size() > 0, "For [Meshgrid], the size of input tensors must be greater than 0.");

  for (size_t i = 0; i < tensors_list_vector.size() - 1; ++i) {
    MS_CHECK_VALUE(tensors_list_vector[i]->data_type() == tensors_list_vector[i + 1]->data_type(),
                   "For Primitive [Meshgrid], all tensors should have the same type.");
  }

  bool swap_tensors = false;
  const size_t MIN_SWAP_SIZE = 2;
  if (indexing_imm == ops::Indexing::XY && tensors_list_vector.size() >= MIN_SWAP_SIZE) {
    swap_tensors = true;
    std::swap(tensors_list_vector[kIndex0], tensors_list_vector[kIndex1]);
  }

  std::vector<tensor::TensorPtr> view_outputs;
  auto view_shape_list = std::vector<int64_t>(tensors_list_vector.size(), 1);
  for (size_t i = 0; i < tensors_list_vector.size(); ++i) {
    view_shape_list[i] = -1;
    view_outputs.push_back(view(tensors_list_vector[i], view_shape_list));
    view_shape_list[i] = 1;
  }

  //   ShapeVector outputs_shape;
  std::vector<int64_t> broadcasted_shape;
  constexpr int64_t SCALAR_TO_TENSOR_SIZE = 1;
  for (const auto &tensor : tensors_list_vector) {
    const auto &input_shape = tensor->shape();
    if (input_shape.empty()) {
      broadcasted_shape.push_back(SCALAR_TO_TENSOR_SIZE);
    } else {
      broadcasted_shape.push_back(input_shape[kIndex0]);
    }
  }

  std::vector<tensor::TensorPtr> broadcast_to_outputs;
  broadcast_to_outputs.reserve(view_outputs.size());
  (void)std::transform(view_outputs.begin(), view_outputs.end(), std::back_inserter(broadcast_to_outputs),
                       [&broadcasted_shape](const tensor::TensorPtr &view_tensor) {
                         return broadcast_to(view_tensor, broadcasted_shape);
                       });

  if (swap_tensors) {
    std::swap(broadcast_to_outputs[kIndex0], broadcast_to_outputs[kIndex1]);
  }

  return broadcast_to_outputs;
}
}  // namespace mindspore::kernel::pyboost
