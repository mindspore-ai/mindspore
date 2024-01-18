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

#include "ops/ops_func_impl/apply_rotary_pos_emb.h"

#include <string>

#include "abstract/ops/primitive_infer_map.h"
#include "ops/nn_ops.h"
#include "utils/check_convert_utils.h"
#include "ops/primitive_c.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {

abstract::TupleShapePtr ApplyRotaryPosEmbInferShape(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kApplyRotaryPosEmbInputsNum, op_name);
  auto query_shape1 =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kApplyRotaryPosEmbQueryIndex]->BuildShape())[kShape];
  auto key_shape1 =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kApplyRotaryPosEmbKeyIndex]->BuildShape())[kShape];
  auto query_shape2 = std::make_shared<abstract::Shape>(query_shape1);
  auto key_shape2 = std::make_shared<abstract::Shape>(key_shape1);
  return std::make_shared<abstract::TupleShape>(abstract::BaseShapePtrList{query_shape2, key_shape2});
}

TuplePtr ApplyRotaryPosEmbInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set valid_types = {kFloat16, kBFloat16};
  auto op_name = prim->name();
  std::map<std::string, TypePtr> types;

  (void)types.emplace("query", input_args[kApplyRotaryPosEmbQueryIndex]->BuildType());
  (void)types.emplace("key", input_args[kApplyRotaryPosEmbKeyIndex]->BuildType());
  auto type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);

  TypePtrList output_type_ptr_list(kFApplyRotaryPosEmbOutputsNum);
  output_type_ptr_list[kApplyRotaryPosEmbQueryEmbedIndex] = type;
  output_type_ptr_list[kApplyRotaryPosEmbKeyEmbedIndex] = type;
  return std::make_shared<Tuple>(output_type_ptr_list);
}
}  // namespace

AbstractBasePtr ApplyRotaryPosEmbInferShapeAndType(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kApplyRotaryPosEmbInputsNum, primitive->name());
  auto infer_type = ApplyRotaryPosEmbInferType(primitive, input_args);
  auto infer_shape = ApplyRotaryPosEmbInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(ApplyRotaryPosEmb, BaseOperator);

// AG means auto generated
class MIND_API AGApplyRotaryPosEmbInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ApplyRotaryPosEmbInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ApplyRotaryPosEmbInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ApplyRotaryPosEmbInferShapeAndType(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ApplyRotaryPosEmb, prim::kPrimApplyRotaryPosEmb, AGApplyRotaryPosEmbInfer, false);
}  // namespace ops
}  // namespace mindspore
