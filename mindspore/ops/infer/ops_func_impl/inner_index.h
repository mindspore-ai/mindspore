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

#ifndef MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_INNER_INDEX_H_
#define MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_INNER_INDEX_H_

#include <vector>
#include <tuple>
#include "mindapi/base/types.h"
#include "ops/ops_func_impl/op_func_impl.h"
#include "primitive/op_name.h"

namespace mindspore {
namespace ops {
class OPS_API InnerIndexFuncImpl : public OpFuncImpl {
 public:
  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override;
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override;
  bool GeneralInferRegistered() const override { return true; };
  ShapeVector CheckAndCalOutputShapeInTupleCase(const ShapeVector &x_shape, const ShapeArray &indices_shapes) const;

  std::tuple<ShapeVector, ShapeArray> TransposeToFront(const ShapeVector &x_shape,
                                                       const ShapeArray &index_shapes) const;
  bool IndexContiguous(const ShapeArray &index_shape) const;
  ShapeArray ExpandIndexShape(const ShapeArray &to_expand) const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_INNER_INDEX_H_
