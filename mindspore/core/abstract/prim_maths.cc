/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

AbstractBasePtr InferImplTensorAdd(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
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
}  // namespace abstract
}  // namespace mindspore
