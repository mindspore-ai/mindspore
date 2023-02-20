/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "ops/index_add.h"

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
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
int64_t IndexAdd::get_axis() const {
  auto value_ptr = this->GetAttr(kAxis);
  return GetValue<int64_t>(value_ptr);
}

void IndexAdd::set_axis(const int64_t axis) { (void)this->AddAttr(kAxis, api::MakeValue(axis)); }

namespace {
abstract::ShapePtr IndexAddInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 3;
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kGreaterEqual, input_num,
                                           prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const size_t x_index = 0;
  const size_t idx_index = 1;
  const size_t y_index = 2;
  auto x_shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, x_index);
  auto idx_shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, idx_index);
  auto y_shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, y_index);
  auto x_is_dynamic = x_shape_ptr->IsDynamic();
  auto idx_is_dynamic = idx_shape_ptr->IsDynamic();
  auto y_is_dynamic = y_shape_ptr->IsDynamic();
  if (x_is_dynamic) {
    return x_shape_ptr;
  }

  auto x_shape = x_shape_ptr->shape();
  auto y_shape = y_shape_ptr->shape();
  auto x_rank = SizeToLong(x_shape.size());
  auto y_rank = SizeToLong(y_shape.size());
  if (!y_is_dynamic) {
    CheckAndConvertUtils::Check("x rank", x_rank, kEqual, y_rank, prim_name);
  }
  auto axis = GetValue<int64_t>(primitive->GetAttr(kAxis));
  CheckAndConvertUtils::CheckInRange("axis", axis, kIncludeNeither, {-x_rank - 1, x_rank}, prim_name);
  auto idx_shape = idx_shape_ptr->shape();
  auto idx_rank = SizeToLong(idx_shape.size());
  (void)CheckAndConvertUtils::CheckInteger("idx size", idx_rank, kEqual, 1, prim_name);
  auto axis_rank = axis;
  if (axis < 0) {
    axis_rank = axis + x_rank;
  }
  if (y_is_dynamic) {
    return x_shape_ptr;
  }
  if (!idx_is_dynamic) {
    size_t axis_value = static_cast<size_t>(axis_rank);
    CheckAndConvertUtils::Check("size of indices", idx_shape[0], kEqual, y_shape[axis_value], prim_name);
  }
  for (int dim = 0; dim < x_rank; dim = dim + 1) {
    if (dim != axis_rank) {
      CheckAndConvertUtils::Check("x dim", x_shape[IntToSize(dim)], kEqual, y_shape[IntToSize(dim)], prim_name);
    }
  }
  return x_shape_ptr;
}

TypePtr IndexAddInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = prim->name();
  const int64_t input_num = 3;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, input_num,
                                           op_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypePtr> valid_types = {kInt8, kInt16, kInt32, kUInt8, kFloat16, kFloat32, kFloat64};
  const std::set<TypePtr> indices_types = {kInt32};
  auto var_type = input_args[kInputIndex0]->BuildType();
  auto indices_type = input_args[kInputIndex1]->BuildType();
  auto updates_type = input_args[kInputIndex2]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices type", indices_type, indices_types, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input_y type", updates_type, valid_types, prim->name());
  return CheckAndConvertUtils::CheckTensorTypeValid("input_x type", var_type, valid_types, prim->name());
}
}  // namespace

MIND_API_OPERATOR_IMPL(IndexAdd, BaseOperator);
AbstractBasePtr IndexAddInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  return abstract::MakeAbstract(IndexAddInferShape(primitive, input_args), IndexAddInferType(primitive, input_args));
}

// AG means auto generated
class MIND_API AGIndexAddInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return IndexAddInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return IndexAddInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return IndexAddInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(IndexAdd, prim::kPrimIndexAdd, AGIndexAddInfer, false);
}  // namespace ops
}  // namespace mindspore
