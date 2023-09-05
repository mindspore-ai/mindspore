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

#include "ops/remake_tuple_index.h"

#include <vector>
#include <string>
#include <memory>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "include/common/utils/utils.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/structure_ops.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(RemakeTupleIndex, BaseOperator);
AbstractBasePtr RemakeTupleIndexInferInner(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string op_name = primitive->name();
  const size_t inputs_size = 9;
  CheckArgsSize(op_name, input_args, inputs_size);
  const AbstractBasePtr &data_abs = input_args[kIndex0];
  const AbstractBasePtr &index_tensor_abs = input_args[kIndex1];
  if (index_tensor_abs->BuildShape()->IsDynamic()) {
    auto output = std::make_shared<abstract::AbstractTensor>(kInt64, ShapeVector{abstract::Shape::kShapeRankAny});
    return output;
  }
  ShapeVector data_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(data_abs->BuildShape())[kShape];
  ShapeVector final_data_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(index_tensor_abs->BuildShape())[kShape];
  (void)final_data_shape.emplace_back(data_shape.size());
  auto output = std::make_shared<abstract::AbstractTensor>(kInt64, final_data_shape);
  return output;
}

class MIND_API RemakeTupleIndexInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return RemakeTupleIndexInferInner(primitive, input_args)->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    return RemakeTupleIndexInferInner(prim, input_args)->BuildType();
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(RemakeTupleIndex, prim::kPrimRemakeTupleIndex, RemakeTupleIndexInfer, false);
}  // namespace ops
}  // namespace mindspore
