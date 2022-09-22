/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "abstract/ops/infer_functions.h"
#include "abstract/utils.h"
#include "abstract/param_validator.h"
#include "utils/ms_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace abstract {
AbstractBasePtr InferImplSqrt(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const AbstractBasePtrList &args_spec_list) {
  // Inputs: three tensors.
  constexpr auto kSqrtInputNum = 1;
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, kSqrtInputNum);
  auto inp = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  return inp->Clone()->Broaden();
}

AbstractBasePtr InferImplSqrtGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list) {
  // Inputs: two tensors.
  constexpr auto kSqrtGradInputNum = 2;
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, kSqrtGradInputNum);
  auto out = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto dout = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  (void)CheckDtypeSame(op_name, out, dout);
  CheckShapeSame(op_name, out, dout);

  return out->Broaden();
}

int64_t InferImplReduceFuncCheckAxis(const int64_t &axis, const size_t dim) {
  int64_t dim_ = static_cast<int64_t>(dim);
  if (axis < -dim_ || axis >= dim_) {
    MS_LOG(EXCEPTION) << "axis should be in [" << -dim_ << ", " << dim_ << "). But got axis = " << axis;
  }
  int64_t ret_axis = axis;
  if (axis >= -dim_ && axis < 0) {
    ret_axis += dim_;
  }
  return ret_axis;
}

void InferImplReduceFuncCalShape(ShapeVector *shape, const ShapeVector &x_shape, const ValuePtr &axis,
                                 bool keep_dims_value) {
  MS_EXCEPTION_IF_NULL(axis);
  if (axis->isa<ValueTuple>() || axis->isa<ValueList>()) {
    auto axis_ptr_list =
      axis->isa<ValueTuple>() ? axis->cast<ValueTuplePtr>()->value() : axis->cast<ValueListPtr>()->value();
    if (axis_ptr_list.empty()) {
      if (keep_dims_value) {
        (void)shape->insert(shape->end(), x_shape.size(), 1);
      }
    } else {
      if (keep_dims_value) {
        *shape = x_shape;
        for (auto it = axis_ptr_list.begin(); it != axis_ptr_list.end(); ++it) {
          int64_t axis_value = GetValue<int64_t>(*it);
          axis_value = InferImplReduceFuncCheckAxis(axis_value, x_shape.size());
          shape->at(LongToSize(axis_value)) = 1;
        }
      } else {
        std::set<size_t> axis_items;
        for (auto &axis_ptr : axis_ptr_list) {
          auto positive_axis = InferImplReduceFuncCheckAxis(GetValue<int64_t>(axis_ptr), x_shape.size());
          (void)axis_items.insert(LongToSize(positive_axis));
        }
        for (size_t i = 0; i < x_shape.size(); ++i) {
          if (axis_items.count(i) == 0) {
            (void)shape->emplace_back(x_shape[i]);
          }
        }
      }
    }
  } else if (axis->isa<Int32Imm>() || axis->isa<Int64Imm>()) {
    (void)shape->insert(shape->end(), x_shape.begin(), x_shape.end());
    auto axis_value = GetValue<int64_t>(axis);
    axis_value = InferImplReduceFuncCheckAxis(axis_value, x_shape.size());
    if (keep_dims_value) {
      shape->at(LongToSize(axis_value)) = 1;
    } else {
      (void)shape->erase(shape->begin() + axis_value);
    }
  } else {
    MS_LOG(EXCEPTION) << "Axis should be one of types: [int/tuple/list].";
  }
  return;
}

AbstractBasePtr InferImplBinaryBase(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list) {
  constexpr auto kBinaryBaseInputNum = 2;
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, kBinaryBaseInputNum);
  auto input_x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(input_x);
  MS_EXCEPTION_IF_NULL(input_x->shape());

  auto input_y = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  MS_EXCEPTION_IF_NULL(input_y);
  MS_EXCEPTION_IF_NULL(input_y->shape());

  auto x_shape = input_x->shape()->shape();
  auto y_shape = input_y->shape()->shape();
  auto output_shape = BroadcastShape(x_shape, y_shape);

  auto x_type = input_x->BuildType();
  MS_EXCEPTION_IF_NULL(x_type);
  MS_EXCEPTION_IF_NULL(x_type->cast<TensorTypePtr>());
  auto y_type = input_y->BuildType();
  MS_EXCEPTION_IF_NULL(y_type);
  MS_EXCEPTION_IF_NULL(y_type->cast<TensorTypePtr>());

  auto x_element = x_type->cast<TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(x_element);
  auto y_element = y_type->cast<TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(y_element);

  auto x_element_type = x_element->number_type();
  auto y_element_type = y_element->number_type();

  auto x_priority = type_priority_map().find(x_element_type);
  if (x_priority == type_priority_map().cend()) {
    MS_LOG(EXCEPTION) << "input_x type is " << x_element_type << ", it's not number type.";
  }
  auto y_priority = type_priority_map().find(y_element_type);
  if (y_priority == type_priority_map().cend()) {
    MS_LOG(EXCEPTION) << "input_y type is " << y_element_type << ", it's not number type.";
  }

  if (x_priority->second >= y_priority->second) {
    return std::make_shared<AbstractTensor>(input_x->element(), std::make_shared<Shape>(output_shape));
  } else {
    return std::make_shared<AbstractTensor>(input_y->element(), std::make_shared<Shape>(output_shape));
  }
}

AbstractBasePtr InferImplMinimum(const AnalysisEnginePtr &engine_ptr, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list) {
  return InferImplBinaryBase(engine_ptr, primitive, args_spec_list);
}

AbstractBasePtr InferImplDivNoNan(const AnalysisEnginePtr &engine_ptr, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list) {
  return InferImplBinaryBase(engine_ptr, primitive, args_spec_list);
}

AbstractBasePtr InferImplLinSpace(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list) {
  constexpr auto kLinSpaceInputNum = 3;
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, kLinSpaceInputNum);
  auto start = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(start);
  MS_EXCEPTION_IF_NULL(start->shape());
  auto stop = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  MS_EXCEPTION_IF_NULL(stop);
  MS_EXCEPTION_IF_NULL(stop->shape());
  (void)CheckTensorDType(start, {kFloat32}, "Input 0 (start) for LinSpace should be %s");
  (void)CheckTensorDType(stop, {kFloat32}, "Input 1 (stop) for LinSpace should be %s");
  ShapeVector shape;
  ShapeVector max_shape;
  ShapeVector min_shape;
  int64_t num_val = 0;
  // 3rd input is a Tensor when LinSpace is a dynamic shape operator
  const size_t tensor_index = 2;
  auto abs_num = args_spec_list[tensor_index];
  if (abs_num->isa<AbstractTensor>()) {
    auto num = abs_num->cast<AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(num);
    auto num_value_ptr = num->BuildValue();
    MS_EXCEPTION_IF_NULL(num_value_ptr);
    auto num_tensor = num_value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(num_tensor);
    num_val = *static_cast<int64_t *>(num_tensor->data_c());
  } else if (abs_num->isa<AbstractScalar>()) {
    auto num = abs_num->cast<AbstractScalarPtr>();
    num_val = GetValue<int64_t>(num->BuildValue());
  } else {
    MS_LOG(EXCEPTION) << "Invalid abstract type:" << abs_num->type_name();
  }
  shape.emplace_back(num_val);
  if (shape[0] < 0) {
    MS_LOG(EXCEPTION) << "num must be >= 0 in LinSpace";
  }
  max_shape.emplace_back(num_val);
  min_shape.emplace_back(num_val);
  AbstractTensorPtr ret =
    std::make_shared<AbstractTensor>(start->element(), std::make_shared<Shape>(shape, min_shape, max_shape));
  return ret;
}

AbstractBasePtr InferImplMatMul(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const AbstractBasePtrList &args_spec_list) {
  constexpr auto kMatMulInputNum = 2;
  const std::string op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(args_spec_list.size()), kGreaterEqual,
                                           kMatMulInputNum, op_name);
  auto x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  auto y = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  MS_EXCEPTION_IF_NULL(y);
  MS_EXCEPTION_IF_NULL(y->shape());
  auto x_shp = x->shape()->shape();
  auto y_shp = y->shape()->shape();

  TypePtr x_type = x->element()->GetTypeTrack();
  if (x_type->type_id() == TypeId::kNumberTypeInt8) {
    x_type = kInt32;
  }
  if (primitive->HasAttr("cast_type")) {
    auto out_type = primitive->GetAttr("cast_type");
    MS_EXCEPTION_IF_NULL(out_type);
    if (!out_type->isa<Type>()) {
      MS_EXCEPTION(ValueError) << "MatMul cast_type must be a `Type`";
    }
    x_type = out_type->cast<TypePtr>();
  }

  ValuePtr transpose_a_ptr = primitive->GetAttr("transpose_a");
  ValuePtr transpose_b_ptr = primitive->GetAttr("transpose_b");
  bool transpose_a = GetValue<bool>(transpose_a_ptr);
  bool transpose_b = GetValue<bool>(transpose_b_ptr);
  ShapeVector x_min_shape = x->shape()->min_shape();
  ShapeVector x_max_shape = x->shape()->max_shape();
  ShapeVector y_min_shape = y->shape()->min_shape();
  ShapeVector y_max_shape = y->shape()->max_shape();

  if (IsDynamicRank(x_shp) || IsDynamicRank(y_shp)) {
    ShapeVector ret_shape{UNKNOWN_RANK};
    return std::make_shared<AbstractTensor>(x_type, std::make_shared<Shape>(ret_shape));
  }

  const size_t SHAPE_SIZE = 2;
  if (x_shp.size() != SHAPE_SIZE || y_shp.size() != SHAPE_SIZE) {
    MS_LOG(EXCEPTION) << "MatMul inputs should have the same dimension size and equal to 2.";
  }
  auto x_col = x_shp[(transpose_a ? 0 : 1)];
  auto y_row = y_shp[(transpose_b ? 1 : 0)];
  if (x_col != y_row && x_col >= 0 && y_row >= 0) {
    MS_LOG(EXCEPTION) << "MatMul shape error, got x_col: " << x_col << ", y_row: " << y_row
                      << ". In MatMul x_col and y_row should be equal.";
  }

  ShapeVector ret_shape;
  ShapeVector ret_min_shape;
  ShapeVector ret_max_shape;
  auto make_shape = [&transpose_a, &transpose_b](ShapeVector &output, const ShapeVector xshp,
                                                 const ShapeVector yshp) -> void {
    if (!xshp.empty() && !yshp.empty()) {
      output.push_back(xshp[(transpose_a ? 1 : 0)]);
      output.push_back(yshp[(transpose_b ? 0 : 1)]);
    }
    return;
  };
  make_shape(ret_shape, x_shp, y_shp);
  make_shape(ret_min_shape, x_min_shape, y_min_shape);
  make_shape(ret_max_shape, x_max_shape, y_max_shape);
  return std::make_shared<AbstractTensor>(x_type, std::make_shared<Shape>(ret_shape, ret_min_shape, ret_max_shape));
}

AbstractBasePtr InferImplLess(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const AbstractBasePtrList &args_spec_list) {
  constexpr auto kLessInputNum = 2;
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, kLessInputNum);
  auto x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  ShapeVector x_shape = x->shape()->shape();
  ShapeVector x_shape_min = x->shape()->min_shape().empty() ? x_shape : x->shape()->min_shape();
  ShapeVector x_shape_max = x->shape()->max_shape().empty() ? x_shape : x->shape()->max_shape();

  auto y = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  MS_EXCEPTION_IF_NULL(y);
  MS_EXCEPTION_IF_NULL(y->shape());
  ShapeVector y_shape = y->shape()->shape();
  ShapeVector y_shape_min = y->shape()->min_shape().empty() ? y_shape : y->shape()->min_shape();
  ShapeVector y_shape_max = y->shape()->max_shape().empty() ? y_shape : y->shape()->max_shape();

  auto out_shape = BroadcastShape(x_shape, y_shape);
  auto out_shape_min = BroadcastShape(x_shape_min, y_shape_min);
  auto out_shape_max = BroadcastShape(x_shape_max, y_shape_max);
  auto output_type = std::make_shared<Bool>();
  return std::make_shared<AbstractTensor>(output_type,
                                          std::make_shared<Shape>(out_shape, out_shape_min, out_shape_max));
}

AbstractBasePtr InferImplReal(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const AbstractBasePtrList &args_spec_list) {
  // Inputs: one tensors.
  constexpr auto kRealInputNum = 1;
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, kRealInputNum);
  AbstractBasePtr input_abs = args_spec_list[0];
  auto input = dyn_cast<AbstractTensor>(input_abs);
  if (input == nullptr) {
    return input_abs->Clone();
  }
  TypePtr input_type = input->element()->GetTypeTrack();
  TypePtr output_type = nullptr;
  if (input_type->type_id() == TypeId::kNumberTypeComplex64) {
    output_type = kFloat32;
  } else if (input_type->type_id() == TypeId::kNumberTypeComplex128) {
    output_type = kFloat64;
  } else {
    return input_abs->Clone();
  }

  return std::make_shared<AbstractTensor>(output_type, input->shape());
}
}  // namespace abstract
}  // namespace mindspore
