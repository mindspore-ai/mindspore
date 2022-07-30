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

#include <set>
#include <algorithm>
#include "ops/arg_min_v2.h"
#include "mindapi/ir/type.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
int64_t InferImplReduceFuncCheckAxis(const PrimitivePtr &prim, const int64_t &axis, const size_t &dim) {
  int64_t dim_ = static_cast<int64_t>(dim);
  if (axis < -dim_ || axis >= dim_) {
    MS_LOG(EXCEPTION) << "For '" << prim->name() << "', 'axis' must be in [" << -dim_ << ", " << dim_
                      << "). But got 'axis' = " << axis << ".";
  }
  int64_t ret_axis = axis;
  if (axis >= -dim_ && axis < 0) {
    ret_axis += dim_;
  }
  return ret_axis;
}

void InferImplReduceFuncCalShape(const PrimitivePtr &primitive, ShapeVector *shape, const ShapeVector &x_shape,
                                 const ValuePtr &axis) {
  MS_EXCEPTION_IF_NULL(axis);
  MS_EXCEPTION_IF_NULL(shape);
  MS_EXCEPTION_IF_NULL(primitive);
  if (axis->isa<ValueTuple>() || axis->isa<ValueList>()) {
    ValuePtrList axis_ptr_value_list;
    if (axis->isa<ValueTuple>()) {
      auto axis_ptr_tuple = axis->cast<ValueTuplePtr>();
      MS_EXCEPTION_IF_NULL(axis_ptr_tuple);
      axis_ptr_value_list = axis_ptr_tuple->value();
    } else {
      auto axis_ptr_list = axis->cast<ValueListPtr>();
      MS_EXCEPTION_IF_NULL(axis_ptr_list);
      axis_ptr_value_list = axis_ptr_list->value();
    }
    if (axis_ptr_value_list.size() < 1) {
      MS_LOG(EXCEPTION) << "For '" << primitive->name()
                        << "', element of 'axis' must not be none if it is one of these types: [tuple/list].";
    } else {
      (void)shape->insert(shape->end(), x_shape.begin(), x_shape.end());
      ValuePtrList axis_items = axis_ptr_value_list;
      ValuePtrList::iterator it;
      std::vector<int64_t> axis_value_list;
      for (it = axis_items.begin(); it != axis_items.end(); ++it) {
        auto axis_value = GetValue<int64_t>(*it);
        auto axis_positive_value = InferImplReduceFuncCheckAxis(primitive, axis_value, x_shape.size());
        axis_value_list.push_back(axis_positive_value);
      }
      std::sort(axis_value_list.begin(), axis_value_list.end());
      std::vector<int64_t>::reverse_iterator it_re;
      for (it_re = axis_value_list.rbegin(); it_re != axis_value_list.rend(); ++it_re) {
        (void)shape->erase(shape->begin() + *it_re);
      }
    }
  } else if (axis->isa<Int32Imm>() || axis->isa<Int64Imm>()) {
    (void)shape->insert(shape->end(), x_shape.begin(), x_shape.end());
    int64_t axis_value = GetValue<int64_t>(axis);
    axis_value = InferImplReduceFuncCheckAxis(primitive, axis_value, x_shape.size());
    (void)shape->erase(shape->begin() + axis_value);
  } else {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', 'axis' must be one of these types: [int/tuple/list].";
  }
  return;
}

abstract::ShapePtr ArgminV2InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto shape_ptr = CheckAndConvertUtils::GetTensorInputShape("ArgminV2", input_args, 0);
  MS_EXCEPTION_IF_NULL(shape_ptr);
  auto input_shape = shape_ptr->shape();
  auto input_min_shape = shape_ptr->min_shape();
  auto input_max_shape = shape_ptr->max_shape();
  ShapeVector out_shape = {};
  ValuePtr axis_value;
  ValuePtr axis_ptr = input_args[1]->BuildValue();
  MS_EXCEPTION_IF_NULL(axis_ptr);
  if (axis_ptr->isa<tensor::Tensor>() && input_args[1]) {
    auto axis_type = input_args[1]->BuildType();
    MS_EXCEPTION_IF_NULL(axis_type);
    auto axis_type_id = axis_type->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(axis_type_id);
    auto axis_tensor = axis_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(axis_tensor);
    size_t data_size = axis_tensor->DataSize();
    std::vector<ValuePtr> value_list;
    auto element = axis_type_id->element();
    MS_EXCEPTION_IF_NULL(element);
    if (element->type_id() == kNumberTypeInt32) {
      auto shape_data = reinterpret_cast<int *>(axis_tensor->data_c());
      MS_EXCEPTION_IF_NULL(shape_data);
      for (size_t i = 0; i < data_size; i++) {
        value_list.push_back(MakeValue(static_cast<int64_t>(*shape_data)));
        ++shape_data;
      }
    } else {
      auto shape_data2 = reinterpret_cast<int64_t *>(axis_tensor->data_c());
      for (size_t i = 0; i < data_size; i++) {
        value_list.push_back(MakeValue(static_cast<int64_t>(*shape_data2)));
        ++shape_data2;
      }
    }
    axis_value = std::make_shared<ValueTuple>(value_list);
  } else {
    axis_value = axis_ptr;
  }
  InferImplReduceFuncCalShape(primitive, &out_shape, input_shape, axis_value);
  if (!input_min_shape.empty() && !input_max_shape.empty()) {
    ShapeVector shape_min = {};
    ShapeVector shape_max = {};
    InferImplReduceFuncCalShape(primitive, &shape_min, input_min_shape, axis_value);
    InferImplReduceFuncCalShape(primitive, &shape_max, input_max_shape, axis_value);
    return std::make_shared<abstract::Shape>(out_shape, shape_min, shape_max);
  }
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr ArgminV2InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  if (std::any_of(input_args.begin(), input_args.end(), [](const AbstractBasePtr &a) { return a == nullptr; })) {
    MS_LOG(EXCEPTION) << "For '" << prim->name()
                      << ", the input args used for infer shape and type is necessary, but missing it.";
  }
  // ascend ArgMin supports float16 and float32.
  const std::set<TypePtr> x_valid_types = {kFloat16, kFloat32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[0]->BuildType(), x_valid_types, prim->name());
  const std::set<TypePtr> axis_valid_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTypeValid("axis", input_args[1]->BuildType(), axis_valid_types, prim->name());
  return kInt32;
}

MIND_API_OPERATOR_IMPL(ArgminV2, BaseOperator);
abstract::AbstractBasePtr ArgminV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("input size", SizeToLong(input_args.size()), kEqual, input_num,
                                           primitive->name());
  return abstract::MakeAbstract(ArgminV2InferShape(primitive, input_args), ArgminV2InferType(primitive, input_args));
}

REGISTER_PRIMITIVE_EVAL_IMPL(ArgminV2, prim::kPrimArgminV2, ArgminV2Infer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
