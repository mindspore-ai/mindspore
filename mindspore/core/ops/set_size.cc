/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "ops/set_size.h"

#include <set>

#include "abstract/abstract_value.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr SetSizeInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto set_indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto set_values_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  auto set_shape_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[2]->BuildShape())[kShape];
  // support dynamic rank
  if (IsDynamicRank(set_indices_shape) || IsDynamicRank(set_values_shape) || IsDynamicRank(set_shape_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  auto set_indices_shape_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("dimension of SetSize input set_indices",
                                           SizeToLong(set_indices_shape.size()), kEqual, set_indices_shape_num,
                                           op_name);
  (void)CheckAndConvertUtils::CheckInteger("dimension of SetSize input set_values", SizeToLong(set_values_shape.size()),
                                           kEqual, 1, op_name);
  (void)CheckAndConvertUtils::CheckInteger("dimension of SetSize input set_shape", SizeToLong(set_shape_shape.size()),
                                           kEqual, 1, op_name);

  if (IsDynamic(set_shape_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  if (!IsDynamic(set_indices_shape) && !IsDynamic(set_values_shape)) {
    (void)CheckAndConvertUtils::CheckInteger("dimension of SetSize input set_indices or set_shape",
                                             set_indices_shape[1], kEqual, set_shape_shape[0], op_name);
    (void)CheckAndConvertUtils::CheckInteger("dimension of SetSize input set_indices or set_values",
                                             set_indices_shape[0], kEqual, set_values_shape[0], op_name);
  }

  MS_EXCEPTION_IF_NULL(primitive->GetAttr("validate_indices"));
  auto shape_size_dim = set_shape_shape[0];
  bool gen_value_succ = false;
  std::vector<int64_t> set_shape_value_vec(shape_size_dim);
  auto set_shape_tensor = input_args[2];
  MS_EXCEPTION_IF_NULL(set_shape_tensor);
  if (set_shape_tensor->isa<abstract::AbstractTensor>()) {
    const std::set<TypePtr> output_size_valid_types = {kInt64};
    (void)CheckAndConvertUtils::CheckTensorTypeValid("set_shape", set_shape_tensor->BuildType(),
                                                     output_size_valid_types, op_name);
    auto set_shape_value = set_shape_tensor->BuildValue();
    MS_EXCEPTION_IF_NULL(set_shape_value);
    if (!set_shape_value->isa<None>() && !set_shape_value->isa<ValueAny>()) {
      auto set_shape_value_tensor = set_shape_value->cast<tensor::TensorPtr>();
      auto value = static_cast<int64_t *>(set_shape_value_tensor->data_c());
      MS_EXCEPTION_IF_NULL(value);
      for (size_t i = 0; i < LongToSize(shape_size_dim); ++i) {
        set_shape_value_vec[i] = value[i];
      }
      gen_value_succ = true;
    }
  }
  if (!gen_value_succ) {
    auto dense_size = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[2]->BuildShape())[kShape];
    ShapeVector dynamic_shape(dense_size[0] - 1), min_shape(dense_size[0] - 1), max_shape(dense_size[0] - 1);
    auto max_length_ptr = primitive->GetAttr("max_length");
    MS_EXCEPTION_IF_NULL(max_length_ptr);
    int64_t max_length = GetValue<int64_t>(max_length_ptr);
    for (int64_t i = 1; i <= dense_size[0] - 1; ++i) {
      dynamic_shape.end()[-i] = abstract::Shape::kShapeDimAny;
      max_shape.end()[-i] = max_length;
    }
    return std::make_shared<abstract::Shape>(dynamic_shape, max_shape);
  } else {
    ShapeVector output_shape;
    auto set_values_index = 2;
    auto dense_size = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[2]->BuildShape())[kShape];
    if (dense_size.size() == 1 && dense_size[0] < set_values_index) {
      output_shape.push_back(1);
    } else {
      for (unsigned int i = 0; i < dense_size[0] - 1; ++i) {
        output_shape.push_back(set_shape_value_vec[i]);
      }
    }
    return std::make_shared<abstract::Shape>(output_shape);
  }
}

TypePtr SetSizeInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  const std::set<TypePtr> valid_types = {kInt64};
  const std::set<TypePtr> set_values_valid_types = {kInt8, kInt16, kInt32, kInt64, kUInt8, kUInt16};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("set_indices", input_args[kInputIndex0]->BuildType(), valid_types,
                                                   prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("set_values", input_args[kInputIndex1]->BuildType(),
                                                   set_values_valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("set_shape", input_args[kInputIndex2]->BuildType(), valid_types,
                                                   prim_name);
  return std::make_shared<TensorType>(kInt32);
}
}  // namespace

MIND_API_OPERATOR_IMPL(SetSize, BaseOperator);

void SetSize::Init(const bool validate_indices) { set_validate_indices(validate_indices); }

void SetSize::set_validate_indices(const bool &validate_indices) {
  (void)AddAttr(kValidateIndices, api::MakeValue(validate_indices));
}

bool SetSize::get_validate_indices() const {
  auto value_ptr = GetAttr(kValidateIndices);
  return GetValue<bool>(value_ptr);
}

AbstractBasePtr SetSizeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 3;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto infer_type = SetSizeInferType(primitive, input_args);
  auto infer_shape = SetSizeInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGSetSizeInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SetSizeInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SetSizeInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SetSizeInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SetSize, prim::kPrimSetSize, AGSetSizeInfer, false);
}  // namespace ops
}  // namespace mindspore
