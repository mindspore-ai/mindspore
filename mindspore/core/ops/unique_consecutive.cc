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

#include "ops/unique_consecutive.h"

#include <functional>
#include <iostream>

#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "abstract/dshape.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kUniqueConsecutiveInputNum = 1;
// For aicpu, if axis is 1000, that represents None.
constexpr int64_t kAxisIsNone = 1000;

bool CheckNullInput(const std::vector<int64_t> &shape) {
  if (shape.size() != 0) {
    if (std::any_of(shape.begin(), shape.end(), [](int64_t i) { return i == 0; })) {
      return true;
    }
  }
  return false;
}

abstract::BaseShapePtr UniqueConsecutiveInferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  auto input_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto input_shape_vec = input_shape_map[kShape];
  if (CheckNullInput(input_shape_vec)) {
    MS_LOG(EXCEPTION) << "For " << op_name << ", the shape of input cannot contain zero.";
  }

  auto axis_ptr = primitive->GetAttr(kAxis);
  MS_EXCEPTION_IF_NULL(axis_ptr);
  abstract::ShapePtr output_shape, idx_shape, counts_shape;
  ShapeVector output_vec, output_max_vec;
  ShapeVector idx_shape_vec;
  ShapeVector counts_shape_vec, counts_max_vec;
  // dynamic shape, the infershape function will be called two times. In the second time, the attribute
  // axis may be deleted so as to axis_ptr is nullptr.
  if (axis_ptr->isa<None>() || GetValue<int64_t>(axis_ptr) == kAxisIsNone) {
    MS_LOG(INFO) << "node:" << op_name << " has no axis attribute or axis id None! Deal as flatten";
    (void)primitive->SetAttrs({{"axis", MakeValue(kAxisIsNone)}});
    output_vec = {abstract::Shape::kShapeDimAny};
    counts_shape_vec = {abstract::Shape::kShapeDimAny};
    idx_shape_vec = input_shape_vec;
    auto input_total = std::accumulate(input_shape_vec.begin(), input_shape_vec.end(), 1, std::multiplies<int64_t>());
    output_max_vec = {input_total};
    counts_max_vec = {input_total};
  } else {
    int64_t axis = GetValue<int64_t>(axis_ptr);
    int64_t ndims = SizeToLong(input_shape_vec.size());
    if (axis >= ndims || axis < -ndims) {
      MS_EXCEPTION(ValueError) << "For " << op_name << ", the axis must be in the range [-" << ndims << "," << ndims
                               << "), but got " << axis << ".";
    }
    if (axis < 0) {
      axis = axis + ndims;
    }
    if (IsDynamicRank(input_shape_vec) || IsDynamicShape(input_shape_vec)) {
      output_vec = {abstract::Shape::kShapeRankAny};
      counts_shape_vec = {abstract::Shape::kShapeRankAny};
      idx_shape_vec = {abstract::Shape::kShapeRankAny};
    } else {
      size_t axis_size = LongToSize(axis);
      output_vec = input_shape_vec;
      output_vec[axis_size] = abstract::Shape::kShapeDimAny;
      output_max_vec = input_shape_vec;

      idx_shape_vec = {input_shape_vec[axis_size]};

      counts_shape_vec = {abstract::Shape::kShapeDimAny};
      counts_max_vec = {input_shape_vec[axis_size]};
    }
  }

  auto idx_ptr = primitive->GetAttr("return_idx");
  MS_EXCEPTION_IF_NULL(idx_ptr);
  auto cnt_ptr = primitive->GetAttr("return_counts");
  MS_EXCEPTION_IF_NULL(cnt_ptr);
  const auto &return_idx = GetValue<bool>(idx_ptr);
  const auto &return_counts = GetValue<bool>(cnt_ptr);
  if (return_idx == false) {
    idx_shape_vec = {0};
  }
  if (return_counts == false) {
    counts_shape_vec = {0};
  }

  if (IsDynamicRank(input_shape_vec) || IsDynamicShape(input_shape_vec)) {
    output_shape = std::make_shared<abstract::Shape>(output_vec);
    counts_shape = std::make_shared<abstract::Shape>(counts_shape_vec);
  } else {
    output_shape = std::make_shared<abstract::Shape>(output_vec, output_max_vec);
    counts_shape = std::make_shared<abstract::Shape>(counts_shape_vec, counts_max_vec);
  }
  idx_shape = std::make_shared<abstract::Shape>(idx_shape_vec);

  auto ret_shape_vec = std::vector<abstract::BaseShapePtr>{output_shape};
  (void)ret_shape_vec.emplace_back(idx_shape);
  (void)ret_shape_vec.emplace_back(counts_shape);
  return std::make_shared<abstract::TupleShape>(ret_shape_vec);
}

TypePtr UniqueConsecutiveInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto name = primitive->name();
  const std::set valid_types = {kComplex64, kComplex128, kFloat16, kFloat,  kFloat64, kInt8,  kInt16,
                                kInt32,     kInt64,      kUInt8,   kUInt16, kUInt32,  kUInt64};
  auto input_type = CheckAndConvertUtils::CheckTypeValid("input", input_args[0]->BuildType(), valid_types, name);
  std::vector<TypePtr> ret_type_vec = {input_type, std::make_shared<TensorType>(kInt32),
                                       std::make_shared<TensorType>(kInt32)};
  return std::make_shared<Tuple>(ret_type_vec);
}
}  // namespace

AbstractBasePtr UniqueConsecutiveInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kUniqueConsecutiveInputNum, primitive->name());
  auto infer_type = UniqueConsecutiveInferType(primitive, input_args);
  auto infer_shape = UniqueConsecutiveInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(UniqueConsecutive, BaseOperator);

// AG means auto generated
class MIND_API AGUniqueConsecutiveInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return UniqueConsecutiveInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return UniqueConsecutiveInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return UniqueConsecutiveInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(UniqueConsecutive, prim::kPrimUniqueConsecutive, AGUniqueConsecutiveInfer, false);
}  // namespace ops
}  // namespace mindspore
