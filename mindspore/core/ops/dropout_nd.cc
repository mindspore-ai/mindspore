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
#include "ops/dropout_nd.h"
#include <string>
#include <algorithm>
#include <set>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr Dropout2DInferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  const int64_t input_num = 1;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, op_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto input_shape_ptr = input_args[kInputIndex0]->BuildShape();
  auto input_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape());
  auto input_shape = input_shape_map[kShape];
  if (IsDynamicRank(input_shape)) {
    abstract::ShapePtr out_shape = std::make_shared<abstract::Shape>(std::vector<int64_t>{UNKNOWN_RANK});
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{out_shape, out_shape});
  }
  // Check Dropout2d input shape whether equal to 4D.
  const int64_t input_rank = 4;
  (void)CheckAndConvertUtils::CheckValue<int64_t>("rank of input ", SizeToLong(input_shape.size()), kEqual, input_rank,
                                                  primitive->name());
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{input_shape_ptr, input_shape_ptr});
}

abstract::TupleShapePtr Dropout3DInferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  const int64_t input_num = 1;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, op_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto input_shape_ptr = input_args[kInputIndex0]->BuildShape();
  auto input_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape());
  auto input_shape = input_shape_map[kShape];
  if (IsDynamicRank(input_shape)) {
    auto unknow_shape_p = std::make_shared<abstract::Shape>(ShapeVector{UNKNOWN_RANK});
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{unknow_shape_p, unknow_shape_p});
  }
  // Check Dropout3d input shape whether equal to 5D.
  const int64_t input_rank = 5;
  (void)CheckAndConvertUtils::CheckValue<int64_t>("rank of input ", SizeToLong(input_shape.size()), kEqual, input_rank,
                                                  primitive->name());
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{input_shape_ptr, input_shape_ptr});
}

TypePtr DropoutNDInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto op_name = primitive->name();
  const int64_t input_num = 1;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, op_name);
  auto input_type = input_args[0]->BuildType();
  std::set<TypePtr> check_list = {kInt8, kInt16, kInt32, kInt64, kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_type, check_list, op_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{input_type, kBool});
}
}  // namespace

MIND_API_OPERATOR_IMPL(Dropout2D, BaseOperator);
MIND_API_OPERATOR_IMPL(Dropout3D, BaseOperator);

void Dropout2D::Init(float keep_prob) { set_keep_prob(keep_prob); }

void Dropout2D::set_keep_prob(float keep_prob) { (void)AddAttr(kKeepProb, api::MakeValue(keep_prob)); }

float Dropout2D::get_keep_prob() const {
  auto value_ptr = GetAttr(kKeepProb);
  return GetValue<float>(value_ptr);
}

void Dropout3D::Init(float keep_prob) { set_keep_prob(keep_prob); }

void Dropout3D::set_keep_prob(float keep_prob) { (void)AddAttr(kKeepProb, api::MakeValue(keep_prob)); }

float Dropout3D::get_keep_prob() const {
  auto value_ptr = GetAttr(kKeepProb);
  return GetValue<float>(value_ptr);
}

AbstractBasePtr Dropout2DInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  TypePtr output_type = DropoutNDInferType(primitive, input_args);
  abstract::TupleShapePtr output_shape = Dropout2DInferShape(primitive, input_args);
  return abstract::MakeAbstract(output_shape, output_type);
}

AbstractBasePtr Dropout3DInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  TypePtr output_type = DropoutNDInferType(primitive, input_args);
  abstract::TupleShapePtr output_shape = Dropout3DInferShape(primitive, input_args);
  return abstract::MakeAbstract(output_shape, output_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(Dropout2D, prim::kPrimDropout2D, Dropout2DInfer, nullptr, true);
REGISTER_PRIMITIVE_EVAL_IMPL(Dropout3D, prim::kPrimDropout3D, Dropout3DInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
