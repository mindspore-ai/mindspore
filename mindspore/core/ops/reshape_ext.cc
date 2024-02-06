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

#include "ops/reshape_ext.h"

#include <string>

#include "abstract/ops/primitive_infer_map.h"
#include "ops/array_ops.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/primitive_c.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kReshapeExtInputsNum = 3;
abstract::ShapePtr ReshapeExtInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto ordinary_input_num = CheckAndConvertUtils::GetRemoveUMonadAbsNum(input_args);
  (void)CheckAndConvertUtils::CheckInteger("inputs num", SizeToLong(ordinary_input_num), kEqual, kReshapeExtInputsNum,
                                           op_name);
  auto key_shape_ptr = input_args[0]->GetShape()->GetShapeVector();
  int64_t var = GetScalarValue<int64_t>(input_args[2]->GetValue()).value();
  key_shape_ptr[2] = var;
  auto key_shape = std::make_shared<abstract::Shape>(key_shape_ptr);
  return key_shape;  // output shape
}

TypePtr ReshapeExtInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto input_type = input_args[0]->GetType();
  return input_type;  // output type
}
}  // namespace

AbstractBasePtr ReshapeExtInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto ordinary_input_num = CheckAndConvertUtils::GetRemoveUMonadAbsNum(input_args);
  (void)CheckAndConvertUtils::CheckInteger("inputs num", SizeToLong(ordinary_input_num), kEqual, kReshapeExtInputsNum,
                                           prim_name);
  auto infer_type = ReshapeExtInferType(primitive, input_args);
  auto infer_shape = ReshapeExtInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(ReshapeExt, BaseOperator);

// AG means auto generated
class MIND_API AGReshapeExtInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ReshapeExtInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ReshapeExtInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ReshapeExtInfer(engine, primitive, input_args);
  }
  std::set<int64_t> GetValueDependArgIndices() const override { return {1}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ReshapeExt, prim::kPrimReshapeExt, AGReshapeExtInfer, false);
}  // namespace ops
}  // namespace mindspore
