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
#include "ops/ops_func_impl/split.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr SplitFuncImpl::InferShape(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto x_shape_ptr = input_args[0]->GetShape();
  auto x_shape = x_shape_ptr->GetShapeVector();
  auto output_num_ptr = input_args[2]->GetValue();
  auto output_num_opt = GetScalarValue<int64_t>(output_num_ptr);
  auto output_num = output_num_opt.value();

  std::vector<abstract::BaseShapePtr> output_list;
  if (IsDynamicRank(x_shape)) {
    for (int64_t i = 0; i < output_num; ++i) {
      abstract::ShapePtr output =
        std::make_shared<abstract::TensorShape>(std::vector<int64_t>(1, abstract::Shape::kShapeRankAny));
      output_list.push_back(output);
    }
    return std::make_shared<abstract::TupleShape>(output_list);
  }

  auto rank = SizeToLong(x_shape.size());
  MS_CHECK_VALUE(rank > 0,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("input rank", rank, kGreaterEqual, 1, primitive));

  auto axis_ptr = input_args[1]->GetValue();
  auto axis_opt = GetScalarValue<int64_t>(axis_ptr);
  if (MS_UNLIKELY(!axis_opt.has_value())) {
    for (int64_t i = 0; i < output_num; ++i) {
      abstract::ShapePtr output =
        std::make_shared<abstract::TensorShape>(std::vector<int64_t>(rank, abstract::Shape::kShapeDimAny));
      output_list.push_back(output);
    }
    return std::make_shared<abstract::TupleShape>(output_list);
  }
  auto axis = axis_opt.value();
  if (axis < 0) {
    axis += rank;
  }

  size_t pos = LongToSize(axis);
  if ((!x_shape_ptr->IsDynamic()) && (x_shape[pos] % output_num != 0)) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', x_shape[" << pos
                             << "] must be divisible by output_num = " << output_num << ", but got " << x_shape[pos];
  }

  auto output_shape = x_shape;
  if (!x_shape_ptr->IsDynamic() || output_shape[pos] > 0) {
    output_shape[pos] = x_shape[pos] / output_num;
  }

  for (int64_t i = 0; i < output_num; ++i) {
    abstract::ShapePtr output = std::make_shared<abstract::TensorShape>(output_shape);
    output_list.push_back(output);
  }
  return std::make_shared<abstract::TupleShape>(output_list);
}

TypePtr SplitFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto output_num_ptr = input_args[2]->GetValue();
  auto output_num_opt = GetScalarValue<int64_t>(output_num_ptr);
  auto output_num = output_num_opt.value();
  auto infer_type = input_args[0]->GetType();
  MS_EXCEPTION_IF_NULL(infer_type);
  std::vector<TypePtr> type_tuple;
  for (int32_t i = 0; i < output_num; i++) {
    type_tuple.push_back(infer_type);
  }
  return std::make_shared<Tuple>(type_tuple);
}

int32_t SplitFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  auto op_name = primitive->name();
  int32_t check_status = OP_CHECK_SUCCESS;
  // Check output_num valid.
  auto output_num_ptr = input_args[2]->GetValue();
  auto output_num_opt = GetScalarValue<int64_t>(output_num_ptr);

  if (MS_UNLIKELY(!output_num_opt.has_value())) {
    check_status = OP_CHECK_RETRY;
  } else {
    const auto output_num = output_num_opt.value();
    if (output_num <= 0) {
      MS_LOG(EXCEPTION) << "For '" << op_name << "', output_num must be positive, but got " << output_num << ".";
    }
  }
  // Check axis valid.
  auto x_shape_ptr = input_args[0]->GetShape();
  auto x_shape = x_shape_ptr->GetShapeVector();
  // Skip to check axis valid if input is dynamic rank.
  if (IsDynamicRank(x_shape)) {
    return check_status;
  }
  auto rank = SizeToLong(x_shape.size());
  auto axis_ptr = input_args[1]->GetValue();
  auto axis_opt = GetScalarValue<int64_t>(axis_ptr);

  if (MS_UNLIKELY(!axis_opt.has_value())) {
    check_status = OP_CHECK_RETRY;
  } else {
    const auto axis = axis_opt.value();
    if (axis >= rank || axis < -rank) {
      MS_LOG(EXCEPTION) << "For '" << op_name << "', axis must in [" << -rank << " , " << rank << "), but got " << axis
                        << ".";
    }
  }
  return check_status;
}
}  // namespace ops
}  // namespace mindspore
