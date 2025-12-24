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

#ifndef MINDSPORE_CORE_OPS_BATCH_MATMUL_H_
#define MINDSPORE_CORE_OPS_BATCH_MATMUL_H_

#include <vector>
#include <string>
#include "mindapi/base/macros.h"
#include "infer/ops_func_impl/binary_op.h"

namespace mindspore {
namespace ops {
class OPS_API BatchMatMulFuncImpl : public OpFuncImpl {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;
  ShapeArray InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const override;
  TypePtrList InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const override;
};

OPS_API void BatchMatMulMakeShape(ShapeVector *output, const ShapeVector xshp, const ShapeVector yshp, bool transpose_a,
                                  bool transpose_b);
OPS_API void CheckBatchMatmulInputWhetherCanBeMul(const std::string &name, const ShapeVector &x_shape,
                                                  const ShapeVector &y_shape, bool transpose_a, bool transpose_b);
OPS_API void CheckBatchMatmulInputWhetherCanBeBroadcast(const std::string &name, const ShapeVector &x_shape,
                                                        const ShapeVector &y_shape);
OPS_API void CheckBatchMatmulInputSize(const std::string &op_name, const std::string &input_name,
                                       const ShapeVector &shape);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_BATCH_MATMUL_H_
