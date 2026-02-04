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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PYBOOST_FUNCTIONS_CUSTOMIZE_VIEW_IMPL_H_
#define MINDSPORE_MINDSPORE_CCSRC_PYBOOST_FUNCTIONS_CUSTOMIZE_VIEW_IMPL_H_

#include <vector>
#include "ir/tensor.h"

namespace mindspore::kernel::pyboost {
mindspore::tensor::TensorPtr reshape_impl(const mindspore::tensor::TensorPtr &input, const std::vector<int64_t> &shape);
mindspore::tensor::TensorPtr flatten_ext_impl(const mindspore::tensor::TensorPtr &input, const int64_t &start_dim,
                                              const int64_t &end_dim);
mindspore::tensor::TensorPtr t_ext_impl(const mindspore::tensor::TensorPtr &input);
mindspore::tensor::TensorPtr view_as_impl(const mindspore::tensor::TensorPtr &input,
                                          const mindspore::tensor::TensorPtr &other);
mindspore::tensor::TensorPtr expand_as_impl(const mindspore::tensor::TensorPtr &input,
                                            const mindspore::tensor::TensorPtr &other);
std::vector<tensor::TensorPtr> meshgrid_impl(const ValueTuplePtr &tensors_list, const int64_t &indexing_imm);
}  // namespace mindspore::kernel::pyboost
#endif  // MINDSPORE_MINDSPORE_CCSRC_PYBOOST_FUNCTIONS_CUSTOMIZE_VIEW_IMPL_H_
