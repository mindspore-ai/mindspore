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

#include "ops/sparse_softmax.h"

#include <set>
#include <map>
#include <string>
#include <algorithm>
#include <memory>

#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
namespace {
constexpr int64_t kIndicesSize = 2;
constexpr int64_t kValuesSize = 1;
constexpr int64_t kShapeSize = 1;
constexpr int64_t kShapeMin = 2;

inline bool CheckShapePositive(const std::vector<int64_t> &input_shape) {
  if (input_shape.size() != 0) {
    if (std::all_of(input_shape.begin(), input_shape.end(), [](int64_t i) { return i > 0; })) {
      return true;
    }
  }
  return false;
}
}  // namespace

abstract::ShapePtr SparseSoftmaxInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto indices_shape_ptr = input_args[kInputIndex0]->BuildShape();
  auto values_shape_ptr = input_args[kInputIndex1]->BuildShape();
  auto shape_shape_ptr = input_args[kInputIndex2]->BuildShape();
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(indices_shape_ptr)[kShape];
  auto values_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(values_shape_ptr)[kShape];
  auto shape_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(shape_shape_ptr)[kShape];
  if (IsDynamicRank(values_shape)) {
    return std::make_shared<abstract::Shape>(values_shape);
  }
  if (!IsDynamicRank(indices_shape)) {
    (void)CheckAndConvertUtils::CheckInteger("indices dimension", SizeToLong(indices_shape.size()), kEqual,
                                             kIndicesSize, prim_name);
  }
  (void)CheckAndConvertUtils::CheckInteger("values dimension", SizeToLong(values_shape.size()), kEqual, kValuesSize,
                                           prim_name);
  if (!IsDynamicRank(shape_shape)) {
    (void)CheckAndConvertUtils::CheckInteger("shape dimension", SizeToLong(shape_shape.size()), kEqual, kShapeSize,
                                             prim_name);
  }
  if (CheckShapePositive(indices_shape) && CheckShapePositive(values_shape) && CheckShapePositive(shape_shape)) {
    auto shape_shape_size = LongToSize(shape_shape[kInputIndex0]);
    (void)CheckAndConvertUtils::CheckInteger("shape size", SizeToLong(shape_shape_size), kGreaterEqual, kShapeMin,
                                             prim_name);
    if (indices_shape[kInputIndex0] != values_shape[kInputIndex0]) {
      MS_EXCEPTION(ValueError) << "For " << prim_name << " the indices size[0] must equal to values number "
                               << values_shape[kInputIndex0] << ", but got " << indices_shape[kInputIndex0] << ".";
    }
    if (indices_shape[kInputIndex1] != shape_shape[kInputIndex0]) {
      MS_EXCEPTION(ValueError) << "For " << prim_name << " the indices size[1] must equal to shape number "
                               << shape_shape[kInputIndex0] << ", but got " << indices_shape[kInputIndex1] << ".";
    }
  }
  return std::make_shared<abstract::Shape>(values_shape);
}

TypePtr SparseSoftmaxInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  auto infer_type_indices = input_args[kInputIndex0]->BuildType();
  auto infer_type_values = input_args[kInputIndex1]->BuildType();
  auto infer_type_shape = input_args[kInputIndex2]->BuildType();
  const std::set<TypePtr> valid_types = {kInt64};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("indices", infer_type_indices);
  (void)types.emplace("shape", infer_type_shape);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim_name);
  const std::set<TypePtr> valid_types_values = {kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("values", infer_type_values, valid_types_values, prim_name);
  return infer_type_values;
}
}  // namespace

MIND_API_OPERATOR_IMPL(SparseSoftmax, BaseOperator);
AbstractBasePtr SparseSoftmaxInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infertype = SparseSoftmaxInferType(primitive, input_args);
  auto infershape = SparseSoftmaxInferShape(primitive, input_args);
  return abstract::MakeAbstract(infershape, infertype);
}

// AG means auto generated
class MIND_API AGSparseSoftmaxInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseSoftmaxInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseSoftmaxInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseSoftmaxInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SparseSoftmax, prim::kPrimSparseSoftmax, AGSparseSoftmaxInfer, false);
}  // namespace ops
}  // namespace mindspore
