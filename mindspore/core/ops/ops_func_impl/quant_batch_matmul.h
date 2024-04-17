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

#ifndef MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_QUANT_BATCH_MATMUL_H_
#define MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_QUANT_BATCH_MATMUL_H_

#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/ops_func_impl/op_func_impl.h"

namespace mindspore {
namespace ops {
class MIND_API QuantBatchMatmulFuncImpl : public OpFuncImpl {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;

 private:
  void CheckBatchMatmulInputSize(const std::string &op_name, const std::string &input_name,
                                 const ShapeVector &shape) const;
  void CheckBatchMatmulInputWhetherCanBeMul(const std::string &name, const ShapeVector &x1_shape,
                                            const ShapeVector &x2_shape, bool transpose_a, bool transpose_b) const;
  void CheckBatchMatmulInputWhetherCanBeBroadcast(const std::string &name, const ShapeVector &x1_shape,
                                                  const ShapeVector &x2_shape) const;
  void BatchMatMulMakeShape(ShapeVector *output, const ShapeVector xshp, const ShapeVector yshp, bool transpose_x1,
                            bool transpose_x2, size_t offset) const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_QUANT_BATCH_MATMUL_H_
