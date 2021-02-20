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

namespace mindspore {
namespace abstract {
AbstractBasePtr InferImplMinOrMaxGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list) {
  // Inputs: three tensors.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 3);
  auto input_x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto input_y = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  auto dout = CheckArg<AbstractTensor>(op_name, args_spec_list, 2);
  (void)CheckTensorsDTypeSame({input_x, input_y, dout}, {kInt, kUInt, kFloat},
                              op_name + "evaluator three inputs should be %s");

  AbstractBasePtr dx = input_x->Broaden();
  AbstractBasePtr dy = input_y->Broaden();

  return std::make_shared<AbstractTuple>(AbstractBasePtrList({dx, dy}));
}

AbstractBasePtr InferImplSqrt(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const AbstractBasePtrList &args_spec_list) {
  // Inputs: three tensors.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  auto inp = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  return inp->Clone()->Broaden();
}

AbstractBasePtr InferImplSqrtGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list) {
  // Inputs: two tensors.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  auto out = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto dout = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  (void)CheckDtypeSame(op_name, out, dout);
  (void)CheckShapeSame(op_name, out, dout);

  return out->Broaden();
}

AbstractBasePtr InferImplAdd(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const AbstractBasePtrList &args_spec_list) {
  // Inputs: two tensors.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  ShapePtr shape_x = dyn_cast<Shape>(args_spec_list[0]->GetShapeTrack());
  MS_EXCEPTION_IF_NULL(shape_x);
  std::vector<int64_t> x_dims = shape_x->shape();
  ShapePtr shape_y = dyn_cast<Shape>(args_spec_list[1]->GetShapeTrack());
  MS_EXCEPTION_IF_NULL(shape_y);
  std::vector<int64_t> y_dims = shape_y->shape();
  auto broadcast_shape = BroadcastShape(x_dims, y_dims);
  if (broadcast_shape.empty()) {
    MS_LOG(EXCEPTION) << "BroadcastShape fail: " << args_spec_list[0]->ToString() << ","
                      << args_spec_list[1]->ToString();
  }
  auto out = args_spec_list[0]->Broaden();
  out->set_shape(std::make_shared<Shape>(broadcast_shape));
  return out;
}

AbstractBasePtr InferImplSquare(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const AbstractBasePtrList &args_spec_list) {
  // Inputs: one tensor.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  auto ref = dyn_cast<abstract::AbstractRef>(args_spec_list[0]);
  if (ref != nullptr) {
    return ref->CloneAsTensor();
  }
  return args_spec_list[0]->Broaden();
}

AbstractBasePtr InferImplEqual(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
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
  if (out_shape.empty()) {
    MS_LOG(EXCEPTION) << "BroadcastShape fail: " << args_spec_list[0]->ToString() << ","
                      << args_spec_list[1]->ToString();
  }
  auto out_shape_min = BroadcastShape(x_shape_min, y_shape_min);
  auto out_shape_max = BroadcastShape(x_shape_max, y_shape_max);

  auto output_type = std::make_shared<Bool>();
  auto ret =
    std::make_shared<AbstractTensor>(output_type, std::make_shared<Shape>(out_shape, out_shape_min, out_shape_max));
  return ret;
}

// To reduce code repeat, use InferImplReduceFunc. Currently registered with ReduceMean, ReduceSum,
// ReduceAll, ReduceAny, ReduceMax, ReduceMin.
AbstractBasePtr InferImplReduceFunc(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
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

  auto check_axis = [](int64_t &axis, const size_t dim) -> void {
    int64_t dim_ = static_cast<int64_t>(dim);
    if (axis < -dim_ || axis >= dim_) {
      MS_LOG(EXCEPTION) << "axis should be in [" << -dim_ << ", " << dim_ << "). But got axis = " << axis;
    }
    if (axis >= -dim_ && axis < 0) {
      axis += dim_;
    }
    return;
  };

  auto cal_shape = [axis, keep_dims_value, check_axis](ShapeVector &shape, const ShapeVector &x_shape) -> void {
    if (axis->isa<ValueTuple>() || axis->isa<ValueList>()) {
      auto axis_ptr_list =
        axis->isa<ValueTuple>() ? axis->cast<ValueTuplePtr>()->value() : axis->cast<ValueListPtr>()->value();
      if (!axis_ptr_list.size()) {
        if (keep_dims_value) shape.insert(shape.end(), x_shape.size(), 1);
      } else {
        shape.insert(shape.end(), x_shape.begin(), x_shape.end());
        ValuePtrList axis_items = axis_ptr_list;
        ValuePtrList::iterator it;
        ValuePtrList::reverse_iterator it_re;
        int64_t axis_value;
        if (keep_dims_value) {
          for (it = axis_items.begin(); it != axis_items.end(); ++it) {
            axis_value = GetValue<int64_t>(*it);
            check_axis(axis_value, x_shape.size());
            shape[axis_value] = 1;
          }
        } else {
          std::sort(axis_items.begin(), axis_items.end());
          for (it_re = axis_items.rbegin(); it_re != axis_items.rend(); ++it_re) {
            axis_value = GetValue<int64_t>(*it_re);
            check_axis(axis_value, x_shape.size());
            shape.erase(std::begin(shape) + axis_value);
          }
        }
      }
    } else if (axis->isa<Int32Imm>() || axis->isa<Int64Imm>()) {
      shape.insert(shape.end(), x_shape.begin(), x_shape.end());
      int64_t axis_value = GetValue<int64_t>(axis);
      check_axis(axis_value, x_shape.size());
      if (keep_dims_value) {
        shape[axis_value] = 1;
      } else {
        shape.erase(std::begin(shape) + axis_value);
      }
    } else {
      MS_LOG(EXCEPTION) << "Axis should be one of types: [int/tuple/list].";
    }
    return;
  };

  ShapeVector shape = {};
  ShapeVector x_shape = input_x->shape()->shape();
  cal_shape(shape, x_shape);

  bool x_is_dyn = (!input_x->shape()->min_shape().empty() && !input_x->shape()->max_shape().empty());
  if (x_is_dyn) {
    ShapeVector shape_min = {};
    ShapeVector shape_max = {};
    ShapeVector x_shape_min = input_x->shape()->min_shape();
    ShapeVector x_shape_max = input_x->shape()->max_shape();
    cal_shape(shape_min, x_shape_min);
    cal_shape(shape_max, x_shape_max);
    return std::make_shared<AbstractTensor>(input_x->element(), std::make_shared<Shape>(shape, shape_min, shape_max));
  }
  return std::make_shared<AbstractTensor>(input_x->element(), std::make_shared<Shape>(shape));
}

AbstractBasePtr InferImplBinaryBase(const AnalysisEnginePtr &engine_ptr, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
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

AbstractBasePtr InferImplMul(const AnalysisEnginePtr &engine_ptr, const PrimitivePtr &primitive,
                             const AbstractBasePtrList &args_spec_list) {
  return InferImplBinaryBase(engine_ptr, primitive, args_spec_list);
}

AbstractBasePtr InferImplSub(const AnalysisEnginePtr &engine_ptr, const PrimitivePtr &primitive,
                             const AbstractBasePtrList &args_spec_list) {
  return InferImplBinaryBase(engine_ptr, primitive, args_spec_list);
}

AbstractBasePtr InferImplDivNoNan(const AnalysisEnginePtr &engine_ptr, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list) {
  return InferImplBinaryBase(engine_ptr, primitive, args_spec_list);
}

AbstractBasePtr InferImplLinSpace(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 3);
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
  if (args_spec_list[2]->isa<AbstractTensor>()) {
    auto num = args_spec_list[2]->cast<AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(num);
    auto num_value_ptr = num->BuildValue();
    MS_EXCEPTION_IF_NULL(num_value_ptr);
    auto num_tensor = num_value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(num_tensor);
    num_val = *static_cast<int64_t *>(num_tensor->data_c());
  } else if (args_spec_list[2]->isa<AbstractScalar>()) {
    auto num = args_spec_list[2]->cast<AbstractScalarPtr>();
    num_val = GetValue<int64_t>(num->BuildValue());
  } else {
    MS_LOG(EXCEPTION) << "Invalid abstract type:" << args_spec_list[2]->type_name();
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

AbstractBasePtr InferImplAddN(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  if (args_spec_list.size() < 1) {
    MS_LOG(EXCEPTION) << "AddN operation must have at least one input.";
  }
  auto input = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  return input->Broaden();
}

AbstractBasePtr InferImplMatMul(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  auto x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  auto y = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  MS_EXCEPTION_IF_NULL(y);
  MS_EXCEPTION_IF_NULL(y->shape());
  auto x_shp = x->shape()->shape();
  auto y_shp = y->shape()->shape();
  if (x_shp.size() != 2 || y_shp.size() != 2) {
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
  (void)CheckMinMaxShape(x_shp, &x_min_shape, &x_max_shape);
  (void)CheckMinMaxShape(y_shp, &y_min_shape, &y_max_shape);
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
  return std::make_shared<AbstractTensor>(x_type, std::make_shared<Shape>(ret_shape, ret_min_shape, ret_max_shape));
}

AbstractBasePtr InferImplBatchMatMul(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  auto x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  auto y = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  MS_EXCEPTION_IF_NULL(y);
  MS_EXCEPTION_IF_NULL(y->shape());
  auto x_shp = x->shape()->shape();
  auto y_shp = y->shape()->shape();
  if (x_shp.size() != y_shp.size() || x_shp.size() < 3) {
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
  (void)CheckMinMaxShape(x_shp, &x_min_shape, &x_max_shape);
  (void)CheckMinMaxShape(y_shp, &y_min_shape, &y_max_shape);
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
    size_t offset = xshp.size() - 2;
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
  return std::make_shared<AbstractTensor>(x_type, std::make_shared<Shape>(ret_shape, ret_min_shape, ret_max_shape));
}

AbstractBasePtr InferImplLess(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
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
  if (out_shape.empty()) {
    MS_LOG(EXCEPTION) << "BroadcastShape fail: " << args_spec_list[0]->ToString() << ","
                      << args_spec_list[1]->ToString();
  }
  auto out_shape_min = BroadcastShape(x_shape_min, y_shape_min);
  auto out_shape_max = BroadcastShape(x_shape_max, y_shape_max);

  auto output_type = std::make_shared<Bool>();
  auto ret =
    std::make_shared<AbstractTensor>(output_type, std::make_shared<Shape>(out_shape, out_shape_min, out_shape_max));
  return ret;
}
}  // namespace abstract
}  // namespace mindspore
