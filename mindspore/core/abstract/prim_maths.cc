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

#include "abstract/infer_functions.h"
#include "abstract/utils.h"
#include "abstract/param_validator.h"
#include "utils/ms_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace abstract {
AbstractBasePtr InferImplMinOrMaxGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list) {
  // Inputs: three tensors.
  constexpr auto kMinMaxGradInputNum = 3;
  const size_t dout_index = 2;
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, kMinMaxGradInputNum);
  auto input_x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto input_y = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  auto dout = CheckArg<AbstractTensor>(op_name, args_spec_list, dout_index);
  (void)CheckTensorsDTypeSame({input_x, input_y, dout}, {kInt, kUInt, kFloat},
                              op_name + "evaluator three inputs should be %s");

  AbstractBasePtr dx = input_x->Broaden();
  AbstractBasePtr dy = input_y->Broaden();

  return std::make_shared<AbstractTuple>(AbstractBasePtrList({dx, dy}));
}

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
  (void)CheckShapeSame(op_name, out, dout);

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
    if (!axis_ptr_list.size()) {
      if (keep_dims_value) (void)shape->insert(shape->end(), x_shape.size(), 1);
    } else {
      (void)shape->insert(shape->end(), x_shape.begin(), x_shape.end());
      ValuePtrList axis_items = axis_ptr_list;
      ValuePtrList::iterator it;
      ValuePtrList::reverse_iterator it_re;
      int64_t axis_value;
      if (keep_dims_value) {
        for (it = axis_items.begin(); it != axis_items.end(); ++it) {
          axis_value = GetValue<int64_t>(*it);
          axis_value = InferImplReduceFuncCheckAxis(axis_value, x_shape.size());
          shape->at(LongToSize(axis_value)) = 1;
        }
      } else {
        std::sort(axis_items.begin(), axis_items.end());
        for (it_re = axis_items.rbegin(); it_re != axis_items.rend(); ++it_re) {
          axis_value = GetValue<int64_t>(*it_re);
          axis_value = InferImplReduceFuncCheckAxis(axis_value, x_shape.size());
          (void)shape->erase(shape->begin() + axis_value);
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

// To reduce code repeat, use InferImplReduceFunc. Currently registered with ReduceMean, ReduceSum,
// ReduceAll, ReduceAny, ReduceMax, ReduceMin.
AbstractBasePtr InferImplReduceFunc(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list) {
  const auto kReduceInputNum = 1;
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, kReduceInputNum);
  auto input_x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(input_x);
  MS_EXCEPTION_IF_NULL(input_x->element());

  ValuePtr keep_dims = primitive->GetAttr("keep_dims");
  MS_EXCEPTION_IF_NULL(keep_dims);
  if (!keep_dims->isa<BoolImm>()) {
    MS_LOG(EXCEPTION) << "Keep_dims should be Bool.";
  }
  bool keep_dims_value = GetValue<bool>(keep_dims);

  ValuePtr axis = primitive->GetAttr("axis");
  MS_EXCEPTION_IF_NULL(axis);

  ShapeVector shape = {};
  ShapeVector x_shape = input_x->shape()->shape();
  InferImplReduceFuncCalShape(&shape, x_shape, axis, keep_dims_value);

  bool x_is_dyn = (!input_x->shape()->min_shape().empty() && !input_x->shape()->max_shape().empty());
  if (x_is_dyn) {
    ShapeVector shape_min = {};
    ShapeVector shape_max = {};
    ShapeVector x_shape_min = input_x->shape()->min_shape();
    ShapeVector x_shape_max = input_x->shape()->max_shape();
    InferImplReduceFuncCalShape(&shape_min, x_shape_min, axis, keep_dims_value);
    InferImplReduceFuncCalShape(&shape_max, x_shape_max, axis, keep_dims_value);
    return std::make_shared<AbstractTensor>(input_x->element(), std::make_shared<Shape>(shape, shape_min, shape_max));
  }
  return std::make_shared<AbstractTensor>(input_x->element(), std::make_shared<Shape>(shape));
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

  auto x_priority = type_priority_map.find(x_element_type);
  if (x_priority == type_priority_map.end()) {
    MS_LOG(EXCEPTION) << "input_x type is " << x_element_type << ", it's not number type.";
  }
  auto y_priority = type_priority_map.find(y_element_type);
  if (y_priority == type_priority_map.end()) {
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
  const size_t SHAPE_SIZE = 2;
  if (x_shp.size() != SHAPE_SIZE || y_shp.size() != SHAPE_SIZE) {
    MS_LOG(EXCEPTION) << "MatMul inputs should have the same dimension size and equal to 2.";
  }
  ValuePtr transpose_a_ptr = primitive->GetAttr("transpose_a");
  ValuePtr transpose_b_ptr = primitive->GetAttr("transpose_b");
  bool transpose_a = GetValue<bool>(transpose_a_ptr);
  bool transpose_b = GetValue<bool>(transpose_b_ptr);
  ShapeVector x_min_shape = x->shape()->min_shape();
  ShapeVector x_max_shape = x->shape()->max_shape();
  ShapeVector y_min_shape = y->shape()->min_shape();
  ShapeVector y_max_shape = y->shape()->max_shape();
  CheckMinMaxShape(x_shp, &x_min_shape, &x_max_shape);
  CheckMinMaxShape(y_shp, &y_min_shape, &y_max_shape);
  // Additional check for dynamic shape
  // Last infer will be real shape values
  bool x_not_dyn = std::all_of(x_shp.begin(), x_shp.end(), [](int64_t value) { return value != Shape::SHP_ANY; });
  bool y_not_dyn = std::all_of(y_shp.begin(), y_shp.end(), [](int64_t value) { return value != Shape::SHP_ANY; });
  if (x_not_dyn && y_not_dyn) {
    auto x_col = x_shp[(transpose_a ? 0 : 1)];
    auto y_row = y_shp[(transpose_b ? 1 : 0)];
    if (x_col != y_row) {
      MS_LOG(EXCEPTION) << "MatMul shape error, got x_col: " << x_col << ", y_row: " << y_row
                        << ". In MatMul x_col and y_row should be equal.";
    }
  }
  ShapeVector ret_shape;
  ShapeVector ret_min_shape;
  ShapeVector ret_max_shape;
  auto make_shape = [&transpose_a, &transpose_b](ShapeVector &output, const ShapeVector xshp,
                                                 const ShapeVector yshp) -> void {
    output.push_back(xshp[(transpose_a ? 1 : 0)]);
    output.push_back(yshp[(transpose_b ? 0 : 1)]);
    return;
  };
  make_shape(ret_shape, x_shp, y_shp);
  make_shape(ret_min_shape, x_min_shape, y_min_shape);
  make_shape(ret_max_shape, x_max_shape, y_max_shape);
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
  return std::make_shared<AbstractTensor>(x_type, std::make_shared<Shape>(ret_shape, ret_min_shape, ret_max_shape));
}

AbstractBasePtr InferImplBatchMatMul(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list) {
  constexpr auto kBatchMatMulInputNum = 2;
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, kBatchMatMulInputNum);
  auto x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  auto y = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  MS_EXCEPTION_IF_NULL(y);
  MS_EXCEPTION_IF_NULL(y->shape());
  auto x_shp = x->shape()->shape();
  auto y_shp = y->shape()->shape();
  constexpr size_t minimum_shape = 3;
  if (x_shp.size() != y_shp.size() || x_shp.size() < minimum_shape) {
    MS_LOG(EXCEPTION)
      << "BatchMatMul input x, y should have the same dimension size and should be greater or equal to 3.";
  }
  ValuePtr transpose_a_ptr = primitive->GetAttr("transpose_a");
  ValuePtr transpose_b_ptr = primitive->GetAttr("transpose_b");
  bool transpose_a = GetValue<bool>(transpose_a_ptr);
  bool transpose_b = GetValue<bool>(transpose_b_ptr);
  ShapeVector x_min_shape = x->shape()->min_shape();
  ShapeVector x_max_shape = x->shape()->max_shape();
  ShapeVector y_min_shape = y->shape()->min_shape();
  ShapeVector y_max_shape = y->shape()->max_shape();
  CheckMinMaxShape(x_shp, &x_min_shape, &x_max_shape);
  CheckMinMaxShape(y_shp, &y_min_shape, &y_max_shape);
  // Additional check for dynamic shape
  // Last infer will be real shape values
  bool x_not_dyn = std::all_of(x_shp.begin(), x_shp.end(), [](int64_t value) { return value != Shape::SHP_ANY; });
  bool y_not_dyn = std::all_of(y_shp.begin(), y_shp.end(), [](int64_t value) { return value != Shape::SHP_ANY; });
  if (x_not_dyn && y_not_dyn) {
    size_t offset = x_shp.size() - 2;
    auto x_col = x_shp[offset + (transpose_a ? 0 : 1)];
    auto y_row = y_shp[offset + (transpose_b ? 1 : 0)];
    if (x_col != y_row) {
      MS_LOG(EXCEPTION) << "BatchMatMul shape error, got x_col: " << x_col << ", y_row: " << y_row
                        << ". In BatchMatMul x_col and y_row should be equal.";
    }
  }
  ShapeVector ret_shape;
  ShapeVector ret_min_shape;
  ShapeVector ret_max_shape;
  auto make_shape = [&transpose_a, &transpose_b](ShapeVector &output, const ShapeVector xshp,
                                                 const ShapeVector yshp) -> void {
    for (size_t i = 0; i < xshp.size() - 2; i++) {
      if (xshp[i] != yshp[i]) {
        if (xshp[i] > 0 && yshp[i] > 0) {
          MS_LOG(EXCEPTION) << "BatchMatMul input x, y are different at index " << i << ".";
        }
        output.push_back(Shape::SHP_ANY);
      } else {
        output.push_back(xshp[i]);
      }
    }
    const size_t bias = 2;
    size_t offset = xshp.size() - bias;
    output.push_back(xshp[offset + (transpose_a ? 1 : 0)]);
    output.push_back(yshp[offset + (transpose_b ? 0 : 1)]);
    return;
  };
  make_shape(ret_shape, x_shp, y_shp);
  make_shape(ret_min_shape, x_min_shape, y_min_shape);
  make_shape(ret_max_shape, x_max_shape, y_max_shape);
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
}  // namespace abstract
}  // namespace mindspore
