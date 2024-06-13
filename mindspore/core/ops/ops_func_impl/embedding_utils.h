/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_FUNC_IMPL_EMBEDDING_UTILS_H_
#define MINDSPORE_CORE_OPS_FUNC_IMPL_EMBEDDING_UTILS_H_

#include <vector>
#include <set>
#include <tuple>
#include <memory>
#include "ir/primitive.h"
#include "abstract/abstract_value.h"

namespace mindspore {
namespace ops {
// for EmbeddingOptimizer, check embedding_dim, keys' num, grad's num and max_grad_norm's num
int32_t CheckEmbeddingOptimizerArgsValidation(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args,
                                              const std::tuple<size_t, size_t, size_t> &indexes,
                                              bool check_max_grad_norm = false,
                                              const std::tuple<size_t, size_t> &other_indexs = std::make_tuple(0, 0));

// for EmbeddingOps, check embedding_dim, _embedding_dim, _max_key_num and keys' shape
int32_t CheckEmbeddingOpsExtraArgs(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_FUNC_IMPL_EMBEDDING_UTILS_H_
