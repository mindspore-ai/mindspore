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

#include "ops/inplace_index_add.h"

#include <memory>
#include <set>

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
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InplaceIndexAddInferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto var_shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, kInputIndex0);
  auto indices_shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, kInputIndex1);
  auto updates_shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, kInputIndex2);
  auto var_is_dynamic = var_shape_ptr->IsDynamic();
  auto indices_is_dynamic = indices_shape_ptr->IsDynamic();
  auto updates_is_dynamic = updates_shape_ptr->IsDynamic();
  if (var_is_dynamic) {
    return var_shape_ptr;
  }
  auto var_shape = var_shape_ptr->shape();
  auto updates_shape = updates_shape_ptr->shape();
  auto var_rank = SizeToLong(var_shape.size());
  auto updates_rank = SizeToLong(updates_shape.size());
  if (!updates_is_dynamic) {
    CheckAndConvertUtils::Check("var rank", var_rank, kEqual, updates_rank, prim_name);
  }
  auto axis = GetValue<int64_t>(primitive->GetAttr(kAxis));
  CheckAndConvertUtils::CheckInRange("axis", axis, kIncludeNeither, {-var_rank - 1, var_rank}, prim_name);
  auto indices_shape = indices_shape_ptr->shape();
  auto indices_rank = SizeToLong(indices_shape.size());
  (void)CheckAndConvertUtils::CheckInteger("indices size", indices_rank, kEqual, 1, prim_name);
  size_t axis_rank = 0;
  if (axis < 0) {
    axis_rank = axis + var_rank;
  } else {
    axis_rank = LongToSize(axis);
  }
  if (updates_is_dynamic) {
    return var_shape_ptr;
  }
  if (!indices_is_dynamic) {
    (void)CheckAndConvertUtils::Check("size of indices", indices_shape[0], kEqual, updates_shape[axis_rank], prim_name);
  }
  for (size_t dim = 0; dim < LongToSize(var_rank); dim = dim + 1) {
    if (dim != axis_rank) {
      (void)CheckAndConvertUtils::Check("var dim", var_shape[dim], kEqual, updates_shape[dim], prim_name);
    }
  }

  return var_shape_ptr;
}

TypePtr InplaceIndexAddInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> valid_types = {kUInt8, kInt8, kInt16, kInt32, kFloat16, kFloat32, kFloat64};
  const std::set<TypePtr> indices_types = {kInt32};
  auto var_type = input_args[kInputIndex0]->BuildType();
  auto indices_type = input_args[kInputIndex1]->BuildType();
  auto updates_type = input_args[kInputIndex2]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices", indices_type, indices_types, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("updates", updates_type, valid_types, prim->name());
  return CheckAndConvertUtils::CheckTensorTypeValid("var", var_type, valid_types, prim->name());
}
}  // namespace

MIND_API_OPERATOR_IMPL(InplaceIndexAdd, BaseOperator);
AbstractBasePtr InplaceIndexAddInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 3;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = InplaceIndexAddInferType(primitive, input_args);
  auto infer_shape = InplaceIndexAddInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGInplaceIndexAddInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return InplaceIndexAddInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return InplaceIndexAddInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return InplaceIndexAddInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(InplaceIndexAdd, prim::kPrimInplaceIndexAdd, AGInplaceIndexAddInfer, false);
}  // namespace ops
}  // namespace mindspore
