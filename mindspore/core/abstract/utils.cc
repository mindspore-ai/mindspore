/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#include "abstract/utils.h"

#include <string>
#include <sstream>
#include <memory>
#include "utils/ms_context.h"
#include "utils/symbolic.h"
#include "abstract/param_validator.h"

namespace mindspore {
namespace abstract {
const std::map<TypeId, size_t> type_map = {
  {kNumberTypeBool, 1},       {kNumberTypeInt, 4},     {kNumberTypeInt8, 1},    {kNumberTypeInt16, 2},
  {kNumberTypeInt32, 4},      {kNumberTypeInt64, 8},   {kNumberTypeUInt, 4},    {kNumberTypeUInt8, 1},
  {kNumberTypeUInt16, 2},     {kNumberTypeUInt32, 4},  {kNumberTypeUInt64, 8},  {kNumberTypeFloat, 4},
  {kNumberTypeFloat16, 2},    {kNumberTypeFloat32, 4}, {kNumberTypeFloat64, 8}, {kNumberTypeComplex64, 8},
  {kNumberTypeComplex128, 16}};

ValuePtr ValueJoin(const ValuePtr &value1, const ValuePtr &value2) {
  MS_EXCEPTION_IF_NULL(value1);
  MS_EXCEPTION_IF_NULL(value2);
  if (*value1 == *value2) {
    return value1;
  }
  return kAnyValue;
}

TypePtr TypeJoin(const TypePtr &type1, const TypePtr &type2) {
  MS_EXCEPTION_IF_NULL(type1);
  MS_EXCEPTION_IF_NULL(type2);
  if (*type1 == *type2) {
    return type1;
  }
  return kAnyType;
}

ShapePtr CalculateDynamicShape(const ShapePtr &shape1, const ShapePtr &shape2, const ShapeVector &dims) {
  // calculate dynamic shape
  ShapeVector min_dims(dims.size());
  ShapeVector max_dims(dims.size());
  MS_EXCEPTION_IF_NULL(shape1);
  MS_EXCEPTION_IF_NULL(shape2);
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] != Shape::SHP_ANY) {
      min_dims[i] = max_dims[i] = dims[i];
      continue;
    }
    if (shape1->shape()[i] != Shape::SHP_ANY && shape2->shape()[i] != Shape::SHP_ANY) {
      min_dims[i] = std::min(shape1->shape()[i], shape2->shape()[i]);
      max_dims[i] = std::max(shape1->shape()[i], shape2->shape()[i]);
      continue;
    }
    if (shape1->shape()[i] == Shape::SHP_ANY && shape2->shape()[i] != Shape::SHP_ANY) {
      if (shape1->min_shape().size() <= i || shape1->max_shape().size() <= i) {
        MS_EXCEPTION(ValueError) << "Shape " << shape1->ToString()
                                 << " has dynamic shape, but does not have min/max shape info.";
      }
      min_dims[i] = std::min(shape1->min_shape()[i], shape2->shape()[i]);
      max_dims[i] = std::max(shape1->max_shape()[i], shape2->shape()[i]);
      continue;
    }
    if (shape1->shape()[i] != Shape::SHP_ANY && shape2->shape()[i] == Shape::SHP_ANY) {
      if (shape2->min_shape().size() <= i || shape2->max_shape().size() <= i) {
        MS_EXCEPTION(ValueError) << "Shape " << shape1->ToString()
                                 << " has dynamic shape, but does not have min/max shape info.";
      }
      min_dims[i] = std::min(shape1->shape()[i], shape2->min_shape()[i]);
      max_dims[i] = std::max(shape1->shape()[i], shape2->max_shape()[i]);
      continue;
    }
    // both shapes contains dynamic shape
    if (shape1->min_shape().size() <= i || shape1->max_shape().size() <= i) {
      MS_EXCEPTION(ValueError) << "Shape " << shape1->ToString()
                               << " has dynamic shape, but does not have min/max shape info.";
    }
    if (shape2->min_shape().size() <= i || shape2->max_shape().size() <= i) {
      MS_EXCEPTION(ValueError) << "Shape " << shape2->ToString()
                               << " has dynamic shape, but does not have min/max shape info.";
    }
    min_dims[i] = std::min(shape1->min_shape()[i], shape2->min_shape()[i]);
    max_dims[i] = std::max(shape1->max_shape()[i], shape2->max_shape()[i]);
  }
  return std::make_shared<Shape>(dims, min_dims, max_dims);
}

ShapePtr ShapeJoin(const ShapePtr &shape1, const ShapePtr &shape2) {
  MS_EXCEPTION_IF_NULL(shape1);
  MS_EXCEPTION_IF_NULL(shape2);
  if (*shape1 == *shape2) {
    return shape1;
  }
  // lengths of two shapes are not same, join failed
  if (shape1->shape().size() != shape2->shape().size()) {
    // special case: shape(1), shape() -> shape(1)
    if (shape1->shape().size() == 1 && shape1->shape()[0] == 1 && shape2->shape().empty()) {
      return shape1;
    }
    if (shape2->shape().size() == 1 && shape2->shape()[0] == 1 && shape1->shape().empty()) {
      return shape2;
    }
    return nullptr;
  }
  ShapeVector dims;
  bool has_dynamic_shape = false;
  dims.resize(shape1->shape().size());
  for (std::size_t i = 0; i < shape1->shape().size(); i++) {
    if (shape1->shape()[i] == shape2->shape()[i]) {
      dims[i] = shape1->shape()[i];
      if (shape1->shape()[i] == Shape::SHP_ANY) {
        has_dynamic_shape = true;
      }
    } else {
      dims[i] = Shape::SHP_ANY;
      has_dynamic_shape = true;
    }
  }
  if (!has_dynamic_shape) {
    return std::make_shared<Shape>(dims);
  }
  return CalculateDynamicShape(shape1, shape2, dims);
}

AbstractBasePtr AbstractJoin(const AbstractBasePtrList &args_spec_list) {
  if (args_spec_list.empty()) {
    MS_LOG(EXCEPTION) << "AbstractJoin requires at least 1 params, while the input size is " << args_spec_list.size()
                      << ".";
  }
  AbstractBasePtr arg_spec_tmp = args_spec_list[0];
  MS_EXCEPTION_IF_NULL(arg_spec_tmp);
  for (const auto &arg_spec : args_spec_list) {
    MS_EXCEPTION_IF_NULL(arg_spec);
    arg_spec_tmp = arg_spec_tmp->Join(arg_spec);
    MS_EXCEPTION_IF_NULL(arg_spec_tmp);
  }
  return arg_spec_tmp;
}

AbstractBasePtrList AbstractJoin(const AbstractBasePtrList &spec1, const AbstractBasePtrList &spec2) {
  if (spec1.size() != spec2.size()) {
    MS_LOG(EXCEPTION) << "Join failed as list don't have the same size. spec1: " << ::mindspore::ToString(spec1)
                      << ", spec2: " << ::mindspore::ToString(spec2);
  }
  AbstractBasePtrList joined_list;
  bool changes = false;
  for (std::size_t i = 0; i < spec1.size(); i++) {
    MS_EXCEPTION_IF_NULL(spec1[i]);
    auto joined_elem = spec1[i]->Join(spec2[i]);
    MS_EXCEPTION_IF_NULL(joined_elem);
    if (joined_elem != spec1[i]) {
      changes = true;
    }
    joined_list.push_back(joined_elem);
  }
  if (!changes) {
    return spec1;
  }
  return joined_list;
}

AbstractBasePtr SensitivityTransform(const AbstractBasePtr &spec) {
  AbstractFunctionPtr f_spec = dyn_cast<AbstractFunction>(spec);
  if (f_spec != nullptr) {
    return std::make_shared<AbstractScalar>(kAnyValue, std::make_shared<EnvType>());
  }
  return spec->Clone();
}

namespace {
// Join all types in args_type_list;
TypePtr TypeJoin(const TypePtrList &args_type_list) {
  if (args_type_list.empty()) {
    MS_LOG(EXCEPTION) << "args_type_list is empty";
  }

  TypePtr type_tmp = args_type_list[0];
  for (std::size_t i = 1; i < args_type_list.size(); i++) {
    type_tmp = abstract::TypeJoin(type_tmp, args_type_list[i]);
  }
  return type_tmp;
}
}  // namespace

bool CheckType(const TypePtr &expected_type, const TypePtr &x) {
  // As x and predicate both are mindspore type statically, here we only to judge whether
  // x is predicate or is a subclass of predicate.
  return IsIdentidityOrSubclass(x, expected_type);
}

TypePtr CheckTypeList(const TypePtr &predicate, const TypePtrList &args_type_list) {
  MS_EXCEPTION_IF_NULL(predicate);
  for (const auto &arg_type : args_type_list) {
    MS_EXCEPTION_IF_NULL(arg_type);
    if (!CheckType(predicate, arg_type)) {
      MS_LOG(EXCEPTION) << "The expected is " << predicate->ToString() << ", not " << arg_type->ToString();
    }
  }
  return TypeJoin(args_type_list);
}

int64_t GetPositiveAxis(int64_t axis_value, size_t increment) {
  if (axis_value < 0) {
    axis_value = axis_value + SizeToLong(increment);
  }

  if (axis_value < 0) {
    MS_LOG(EXCEPTION) << "axis_value should not still <0";
  }

  return axis_value;
}

// Return if two shapes can be broadcast.
// Broadcast shape is placed in broadcast_output_shape.
ShapeVector RealBroadcast(const std::string &op, ShapeVector x_shape, ShapeVector y_shape) {
  std::reverse(x_shape.begin(), x_shape.end());
  std::reverse(y_shape.begin(), y_shape.end());
  // Fill a placeholder value 1 which will be replaced later.
  size_t std_len = x_shape.size() > y_shape.size() ? x_shape.size() : y_shape.size();
  y_shape.resize(std_len, 1);
  x_shape.resize(std_len, 1);

  ShapeVector broadcast_shape;
  for (size_t i = 0; i < std_len; i++) {
    int64_t x_i = x_shape[i];  // i-th dimension of x
    int64_t y_i = y_shape[i];  // i-th dimension of y
    int64_t output_i = 0;      // i-th dimension of the output
    if (x_i == y_i) {
      output_i = x_i;
    } else if (x_i == 1) {
      output_i = y_i;
    } else if (y_i == 1) {
      output_i = x_i;
    } else {
      MS_LOG(EXCEPTION)
        << op
        << " evaluator the shape of first tensor and the shape of second tensor do not meet the broadcasting "
           "requirements";
    }
    broadcast_shape.push_back(output_i);
  }
  std::reverse(broadcast_shape.begin(), broadcast_shape.end());
  return broadcast_shape;
}

ShapeVector BroadcastShape(ShapeVector shpx, ShapeVector shpy) {
  int dlen = SizeToInt(shpx.size()) - SizeToInt(shpy.size());
  if (dlen < 0) {
    for (int i = 0; i < -dlen; ++i) {
      (void)shpx.insert(shpx.begin(), 1);
    }
  } else if (dlen > 0) {
    for (int i = 0; i < dlen; i++) {
      (void)shpy.insert(shpy.begin(), 1);
    }
  }
  if (shpx.size() != shpy.size()) {
    MS_LOG(EXCEPTION) << "Failure: shpx.size() != shpy.size().";
  }
  ShapeVector shp;
  for (size_t i = 0; i < shpx.size(); i++) {
    auto a = shpx[i];
    auto b = shpy[i];
    if (a == 1) {
      shp.push_back(b);
    } else if (b == 1) {
      shp.push_back(a);
    } else if (a == -1) {
      shp.push_back(b);
    } else if (b == -1) {
      shp.push_back(a);
    } else if (a == b) {
      shp.push_back(a);
    } else {
      return ShapeVector();
    }
  }
  return shp;
}

size_t TypeIdSize(const TypeId data_type) {
  const size_t unsupported_type_error = 0;
  auto iter = type_map.find(data_type);
  if (iter != type_map.end()) {
    return iter->second;
  }
  return unsupported_type_error;
}

size_t ShapeSize(const std::vector<size_t> &shape) {
  return std::accumulate(shape.begin(), shape.end(), IntToSize(1), std::multiplies<size_t>());
}

void CheckMinMaxShape(const ShapeVector &shape, ShapeVector *min_shape, ShapeVector *max_shape) {
  *min_shape = (*min_shape).empty() ? shape : *min_shape;
  *max_shape = (*max_shape).empty() ? shape : *max_shape;
}

int64_t GetUnsortedSegmentOpScalarArg(const AbstractBasePtrList &args_spec_list, const std::string &op_name) {
  int64_t num_segments_value = 0;
  constexpr size_t scalar_index = 2;
  if (args_spec_list[scalar_index]->isa<AbstractTensor>()) {  // num_segments is Tensor
    auto num_segments = args_spec_list[scalar_index]->cast<AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(num_segments);
    auto num_segments_value_ptr = num_segments->BuildValue();
    MS_EXCEPTION_IF_NULL(num_segments_value_ptr);
    auto num_segments_tensor = num_segments_value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(num_segments_tensor);
    if (num_segments->element()->GetTypeTrack()->type_id() == TypeId::kNumberTypeInt64) {
      num_segments_value = *static_cast<int64_t *>(num_segments_tensor->data_c());
    } else {
      num_segments_value = *static_cast<int32_t *>(num_segments_tensor->data_c());
    }
  } else if (args_spec_list[scalar_index]->isa<AbstractScalar>()) {  // num_segments is Scalar
    auto num_segments = CheckArg<AbstractScalar>(op_name, args_spec_list, scalar_index);
    if (num_segments->GetTypeTrack()->type_id() == TypeId::kNumberTypeInt64) {
      num_segments_value = GetValue<int64_t>(num_segments->BuildValue());
    } else {
      num_segments_value = GetValue<int32_t>(num_segments->BuildValue());
    }
  } else {
    MS_LOG(EXCEPTION) << "num_segments incorrect type in " << op_name;
  }
  return num_segments_value;
}

AbstractBasePtr MakeAbstractTensor(const ShapePtr &shape, const TypePtr &type) {
  MS_EXCEPTION_IF_NULL(shape);
  MS_EXCEPTION_IF_NULL(type);
  AbstractBasePtr tensor = nullptr;
  auto ret_vec = shape->shape();
  ShapeVector min_shape_vec;
  ShapeVector max_shape_vec;

  if (!shape->min_shape().empty()) {
    min_shape_vec = shape->min_shape();
  }
  if (!shape->max_shape().empty()) {
    max_shape_vec = shape->max_shape();
  }

  auto ret_shape = std::make_shared<abstract::Shape>(ret_vec, min_shape_vec, max_shape_vec);
  if (type->isa<TensorType>()) {
    auto tensor_type = type->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor_type);
    auto element = std::make_shared<abstract::AbstractScalar>(kAnyValue, tensor_type->element());
    tensor = std::make_shared<abstract::AbstractTensor>(element, ret_shape);
  } else {
    auto element = std::make_shared<abstract::AbstractScalar>(kAnyValue, type);
    tensor = std::make_shared<abstract::AbstractTensor>(element, ret_shape);
  }
  return tensor;
}

AbstractBasePtr MakeMonadAbstract(const MonadTypePtr &type) {
  if (type->isa<UMonadType>()) {
    return kUMonad->ToAbstract();
  } else if (type->isa<IOMonadType>()) {
    return kIOMonad->ToAbstract();
  }
  MS_EXCEPTION(UnknownError) << "Unsupported to convert type " << type->ToString() << " to monad abstract";
}

AbstractBasePtr MakeAbstract(const BaseShapePtr &base_shape, const TypePtr &type) {
  MS_EXCEPTION_IF_NULL(base_shape);
  MS_EXCEPTION_IF_NULL(type);
  if ((base_shape->isa<Shape>())) {
    auto shape = base_shape->cast<ShapePtr>();
    MS_EXCEPTION_IF_NULL(shape);
    auto shape_vec = shape->shape();
    // if the size of shape list is empty, return an scalar abstract
    if (shape_vec.empty() && (!type->isa<TensorType>())) {
      abstract::AbstractScalarPtr abs_scalar = std::make_shared<abstract::AbstractScalar>(kAnyValue, type);
      return abs_scalar;
    }
    return MakeAbstractTensor(shape, type);
  } else if (base_shape->isa<TupleShape>() && type->isa<Tuple>()) {
    auto shape_tuple = base_shape->cast<TupleShapePtr>();
    auto type_tuple = type->cast<TuplePtr>();
    AbstractBasePtrList ptr_list;
    for (size_t it = 0; it < shape_tuple->size(); ++it) {
      auto tensor_it = MakeAbstract((*shape_tuple)[it], (*type_tuple)[it]);
      ptr_list.push_back(tensor_it);
    }
    auto tuple = std::make_shared<abstract::AbstractTuple>(ptr_list);
    return tuple;
  } else if (base_shape->isa<ListShape>() && type->isa<List>()) {
    auto shape_list = base_shape->cast<ListShapePtr>();
    auto type_list = type->cast<ListPtr>();
    AbstractBasePtrList ptr_list;
    for (size_t it = 0; it < shape_list->size(); ++it) {
      auto tensor_it = MakeAbstract((*shape_list)[it], (*type_list)[it]);
      ptr_list.push_back(tensor_it);
    }
    auto list = std::make_shared<abstract::AbstractList>(ptr_list);
    return list;
  } else if (base_shape->isa<NoShape>() && type->isa<TypeNone>()) {
    // AbstractNone indicates there is no output for this CNode node.
    auto abstract_none = std::make_shared<abstract::AbstractNone>();
    return abstract_none;
  } else if (type->isa<Monad>()) {
    // Return monad abstract if it is monad type.
    return MakeMonadAbstract(type->cast<MonadTypePtr>());
  } else {
    // When sparse enabled, the undetermined might be raised and eliminated in opt passes
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    bool enable_sparse = context->get_param<bool>(MS_CTX_ENABLE_SPARSE);
    if (enable_sparse) {
      return std::make_shared<abstract::AbstractUndetermined>();
    }
    MS_LOG(EXCEPTION) << "evaluator return invalid shape " << base_shape->ToString() << "or type. " << type->ToString();
  }
}
}  // namespace abstract
}  // namespace mindspore
