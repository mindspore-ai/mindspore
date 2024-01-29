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

#include "ops/reshape_and_cache.h"

#include <string>

#include "abstract/ops/primitive_infer_map.h"
#include "ops/nn_ops.h"
#include "utils/check_convert_utils.h"
#include "ops/primitive_c.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr ReshapeAndCacheInferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto ordinary_input_num = CheckAndConvertUtils::GetRemoveUMonadAbsNum(input_args);
  (void)CheckAndConvertUtils::CheckInteger("inputs num", SizeToLong(ordinary_input_num), kEqual,
                                           kReshapeAndCacheInputsNum, op_name);
  auto key_shape_ptr = input_args[kReshapeAndCacheInputKeyIndex]->GetShape()->GetShapeVector();
  auto key_shape = std::make_shared<abstract::Shape>(key_shape_ptr);
  return key_shape;  // output shape
}

TypePtr ReshapeAndCacheInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set valid_types = {kFloat16, kBFloat16};
  auto op_name = prim->name();
  std::map<std::string, TypePtr> types;

  (void)types.emplace("key", input_args[kReshapeAndCacheInputKeyIndex]->GetType());
  (void)types.emplace("value", input_args[kReshapeAndCacheInputValueIndex]->GetType());
  (void)types.emplace("key_cache", input_args[kReshapeAndCacheInputKeyCacheIndex]->GetType());
  (void)types.emplace("value_cache", input_args[kReshapeAndCacheInputValueCacheIndex]->GetType());
  auto type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);

  auto slot_mapping_type = input_args[kReshapeAndCacheInputSlotMappingIndex]->GetType();
  CheckAndConvertUtils::CheckTensorTypeValid("slot_mapping", slot_mapping_type, {kInt32}, op_name);

  return type;  // output type
}
}  // namespace

AbstractBasePtr ReshapeAndCacheInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto ordinary_input_num = CheckAndConvertUtils::GetRemoveUMonadAbsNum(input_args);
  (void)CheckAndConvertUtils::CheckInteger("inputs num", SizeToLong(ordinary_input_num), kEqual,
                                           kReshapeAndCacheInputsNum, prim_name);
  auto infer_type = ReshapeAndCacheInferType(primitive, input_args);
  auto infer_shape = ReshapeAndCacheInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(ReshapeAndCache, BaseOperator);

// AG means auto generated
class MIND_API AGReshapeAndCacheInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ReshapeAndCacheInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ReshapeAndCacheInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ReshapeAndCacheInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ReshapeAndCache, prim::kPrimReshapeAndCache, AGReshapeAndCacheInfer, false);
}  // namespace ops
}  // namespace mindspore
