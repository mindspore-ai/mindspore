/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "ops/coalesce.h"

#include <string>
#include <memory>
#include <set>
#include <vector>

#include "abstract/ops/primitive_infer_map.h"
#include "abstract/dshape.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
TuplePtr CoalesceInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x_values", input_args[kInputIndex1]->BuildType(), valid_types,
                                                   prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x_indices", input_args[kInputIndex0]->BuildType(), {kInt64},
                                                   prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x_shape", input_args[kInputIndex2]->BuildType(), {kInt64},
                                                   prim->name());
  std::vector<TypePtr> types_list = {input_args[0]->BuildType(), input_args[1]->BuildType(),
                                     input_args[2]->BuildType()};
  return std::make_shared<Tuple>(types_list);
}

abstract::TupleShapePtr CoalesceInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr int x_indices_shape_size = 2;
  auto x_indices_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto x_values_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto x_shape_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto x_indices_shape_BaseShapePtr = input_args[kInputIndex0]->BuildShape();
  auto x_values_shape_BaseShapePtr = input_args[kInputIndex1]->BuildShape();
  auto x_shape_shape_BaseShapePtr = input_args[kInputIndex2]->BuildShape();
  auto x_indices_shape_ptr = x_indices_shape_BaseShapePtr->cast<abstract::ShapePtr>();
  auto x_values_shape_ptr = x_values_shape_BaseShapePtr->cast<abstract::ShapePtr>();
  auto x_shape_shape_ptr = x_shape_shape_BaseShapePtr->cast<abstract::ShapePtr>();

  if (IsDynamicRank(x_indices_shape)) {
    abstract::ShapePtr x_indices_shape_dyn = std::make_shared<abstract::Shape>(std::vector<int64_t>{-1, -1});
    abstract::ShapePtr x_values_shape_dyn = std::make_shared<abstract::Shape>(std::vector<int64_t>{-1});
    abstract::ShapePtr x_shape_shape_dyn = std::make_shared<abstract::Shape>(std::vector<int64_t>{-1});
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{x_indices_shape_dyn, x_values_shape_dyn, x_shape_shape_dyn});
  }

  if (!x_indices_shape_ptr->IsDynamic() && !x_values_shape_ptr->IsDynamic() && !x_shape_shape_ptr->IsDynamic()) {
    if (x_indices_shape.size() != x_indices_shape_size || x_values_shape.size() != 1 || x_shape_shape.size() != 1) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "' x_indices must be a 2-D tensor"
                               << ", x_values and x_shape must be a 1-D tensor, but got x_indices is a "
                               << x_indices_shape.size() << "-D tensor, got x_values is a " << x_values_shape.size()
                               << "-D tensor, got x_shape is a " << x_shape_shape.size() << "-D tensor.";
    }
    if (x_indices_shape[0] != x_shape_shape[0]) {
      MS_EXCEPTION(ValueError) << "For " << prim_name
                               << ", first dim of x_indices and first dim of x_shape must be the same"
                               << ", but got first dim of x_indices: " << x_indices_shape[0]
                               << ", first dim of x_shape: " << x_shape_shape[0] << ".";
    }
    if (x_indices_shape[1] != x_values_shape[0]) {
      MS_EXCEPTION(ValueError) << "For " << prim_name
                               << ", second dim of x_indices and first dim of x_values must be the same"
                               << ", but got second dim of x_indices: " << x_indices_shape[1]
                               << ", first dim of x_values: " << x_values_shape[0] << ".";
    }
  }
  ShapeVector y_indices_shape = {x_indices_shape[0], -1};
  ShapeVector y_indices_max_shape = {x_indices_shape[0], x_indices_shape[1]};
  ShapeVector y_values_shape = {-1};
  ShapeVector y_values_max_shape = {x_indices_shape[1]};
  if (x_indices_shape_ptr->IsDynamic()) {
    y_indices_max_shape = {1, 1};
    y_values_max_shape = {1};
  }
  auto y_shape = input_args[2]->BuildShape();
  MS_EXCEPTION_IF_NULL(y_shape);
  abstract::ShapePtr y_shape_shape_list = y_shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(y_shape_shape_list);
  abstract::ShapePtr y_indices_shape_list = std::make_shared<abstract::Shape>(y_indices_shape, y_indices_max_shape);
  abstract::ShapePtr y_values_shape_list = std::make_shared<abstract::Shape>(y_values_shape, y_values_max_shape);
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{y_indices_shape_list, y_values_shape_list, y_shape_shape_list});
}
}  // namespace

MIND_API_OPERATOR_IMPL(Coalesce, BaseOperator);
AbstractBasePtr CoalesceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  const int64_t input_num = 3;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = CoalesceInferType(primitive, input_args);
  auto infer_shape = CoalesceInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGCoalesceInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return CoalesceInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return CoalesceInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return CoalesceInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Coalesce, prim::kPrimCoalesce, AGCoalesceInfer, false);
}  // namespace ops
}  // namespace mindspore
