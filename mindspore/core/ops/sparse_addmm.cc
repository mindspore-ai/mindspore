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
#include <string>
#include <set>
#include <vector>
#include <memory>
#include <map>
#include <algorithm>

#include "ops/sparse_addmm.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
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
abstract::ShapePtr SparseAddmmInferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto values_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto shape_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto x2_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  auto x3_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->BuildShape())[kShape];
  auto alpha_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex5]->BuildShape())[kShape];
  auto beta_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex6]->BuildShape())[kShape];
  const int kDimensionTwo = 2;
  const int kDimensionOne = 1;
  std::vector<ShapeVector> all_shapes = {indices_shape, values_shape, shape_shape, x2_shape, x3_shape};
  bool is_dynamic = std::any_of(all_shapes.begin(), all_shapes.end(), IsDynamic);
  bool is_dynamic_rank = std::any_of(all_shapes.begin(), all_shapes.end(), IsDynamicRank);
  if (!is_dynamic && !is_dynamic_rank) {
    if (indices_shape.size() != kDimensionTwo) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the input indices should "
                               << "have rank 2, but got " << indices_shape.size() << ".";
    }
    if (indices_shape[1] != kDimensionTwo) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the 2nd dimension of indices "
                               << "should be 2, but got " << indices_shape[1] << ".";
    }
    if (shape_shape.size() != kDimensionOne) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the input shape should "
                               << "have rank 1, but got " << shape_shape.size() << ".";
    }
    if (shape_shape[0] != kDimensionTwo) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the 1st dimension of input shape "
                               << "should be 2, but got " << shape_shape[0] << ".";
    }
    if (x2_shape.size() != kDimensionTwo) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the shape of input dense "
                               << "should be [2], but got [" << x2_shape.size() << "].";
    }
    if (x3_shape.size() != kDimensionTwo) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the shape of input dense "
                               << "should be [2], but got [" << x3_shape.size() << "].";
    }
  }
  if (values_shape.size() != kDimensionOne) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the input value should "
                             << "have rank 1, but got " << values_shape.size() << ".";
  }
  if (alpha_shape.size() != kDimensionOne) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the input shape should "
                             << "have rank 1, but got " << alpha_shape.size() << ".";
  }
  if (beta_shape.size() != kDimensionOne) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the input shape should "
                             << "have rank 1, but got " << beta_shape.size() << ".";
  }
  return std::make_shared<abstract::Shape>(x3_shape);
}

TypePtr SparseAddmmInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  std::map<std::string, TypePtr> types;
  std::set<TypePtr> valid_types = {kFloat32, kFloat64, kInt32,  kInt64,  kInt16,
                                   kInt8,    kUInt32,  kUInt64, kUInt16, kUInt8};
  TypePtr indices_type = input_args[kInputIndex0]->BuildType();
  TypePtr values_type = input_args[kInputIndex1]->BuildType();
  TypePtr shape_type = input_args[kInputIndex2]->BuildType();
  TypePtr x2_type = input_args[kInputIndex3]->BuildType();
  TypePtr x3_type = input_args[kInputIndex4]->BuildType();
  TypePtr alpha_type = input_args[kInputIndex5]->BuildType();
  TypePtr beta_type = input_args[kInputIndex6]->BuildType();
  auto prim_name = primitive->name();

  (void)types.emplace("x1_values", values_type);
  (void)types.emplace("x2", x2_type);
  (void)types.emplace("x3", x3_type);
  (void)types.emplace("alpha", alpha_type);
  (void)types.emplace("beta", beta_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());

  const std::set<TypePtr> valid_type = {kInt64, kInt32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices", indices_type, valid_type, prim_name);

  (void)CheckAndConvertUtils::CheckTensorTypeValid("sparse_shape", shape_type, valid_type, prim_name);

  auto tensor_type = x3_type->cast<TensorTypePtr>();
  auto tensor_element = tensor_type->element();

  return tensor_element;
}
}  // namespace
AbstractBasePtr SparseAddmmInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 7;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  // infer type
  auto infer_type = SparseAddmmInferType(primitive, input_args);
  // infer shape
  auto infer_shape = SparseAddmmInferShape(primitive, input_args);
  return std::make_shared<abstract::AbstractTensor>(infer_type, infer_shape);
}
MIND_API_OPERATOR_IMPL(SparseAddmm, BaseOperator);

// AG means auto generated
class MIND_API AGSparseAddmmInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseAddmmInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseAddmmInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseAddmmInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SparseAddmm, prim::kPrimSparseAddmm, AGSparseAddmmInfer, false);
}  // namespace ops
}  // namespace mindspore
