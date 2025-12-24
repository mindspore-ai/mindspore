/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_OP_FUNC_IMPL_TILE_H
#define MINDSPORE_CORE_OPS_OP_FUNC_IMPL_TILE_H

#include <set>
#include <utility>
#include <vector>
#include "ops/ops_func_impl/op_func_impl.h"

namespace mindspore::ops {
class OPS_API TileFuncImpl : public OpFuncImpl {
 public:
  TileFuncImpl() = default;
  ~TileFuncImpl() = default;

  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override;
  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override;
  bool GeneralInferRegistered() const override { return true; }

  std::set<int64_t> GetValueDependArgIndices() const override { return {1}; }
};

/**
 * @brief Expanding one with heading 1 if it is less than the other. Cases:
 *        1. input_shape:       [3], dims: [2, 2, 2] ==> <[1, 1, 3], [2, 2, 2]>.
 *        2. input_shape:    [3, 4], dims:    [2, 2] ==> <   [3, 4],    [2, 2]>.
 *        3. input_shape: [3, 4, 5], dims:       [2] ==> <[3, 4, 5], [1, 1, 2]>.
 *
 * @param input_shape Pointer for input input_shape, must be static rank. Will be expanded if need.
 * @param dims Pointer for dims. Will be expanded if need.
 */
OPS_API void AdaptShapeAndMultipies(ShapeVector *input_shape, ShapeVector *dims);
}  // namespace mindspore::ops
#endif  // MINDSPORE_CORE_OPS_OP_FUNC_IMPL_TILE_H
