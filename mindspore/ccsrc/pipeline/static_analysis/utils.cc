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

#include "pipeline/static_analysis/utils.h"

#include <string>
#include <sstream>
#include <memory>
#include "utils/symbolic.h"
#include "pipeline/static_analysis/param_validator.h"

namespace mindspore {
namespace abstract {
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

ShapePtr ShapeJoin(const ShapePtr &shape1, const ShapePtr &shape2) {
  MS_EXCEPTION_IF_NULL(shape1);
  MS_EXCEPTION_IF_NULL(shape2);
  if (*shape1 == *shape2) {
    return shape1;
  }
  if (shape1->shape().size() != shape2->shape().size()) {
    MS_LOG(WARNING) << "Unsupported shape join. shape1 = " << shape1->ToString() << ", shape2 = " << shape2->ToString();
    return shape1;
  }
  std::vector<int> dims;
  dims.resize(shape1->shape().size());
  for (std::size_t i = 0; i < shape1->shape().size(); i++) {
    if (shape1->shape()[i] == shape2->shape()[i]) {
      dims[i] = shape1->shape()[i];
    } else {
      dims[i] = Shape::SHP_ANY;
    }
  }
  return std::make_shared<Shape>(dims);
}

AbstractBasePtr AbstractJoin(const AbstractBasePtrList &args_spec_list) {
  if (args_spec_list.size() < 1) {
    MS_LOG(EXCEPTION) << "AbstractJoin requires at least 1 params, while the input size is " << args_spec_list.size()
                      << ".";
  }
  AbstractBasePtr arg_spec_tmp = args_spec_list[0];
  MS_EXCEPTION_IF_NULL(arg_spec_tmp);
  for (auto arg_spec : args_spec_list) {
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
    auto joined_elem = spec1[i]->Join(spec2[i]);
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
  // As x and predicate both are mindspore type staticly, here we only to judge whether
  // x is predicate or is a subclass of predicate.
  return IsIdentidityOrSubclass(x, expected_type);
}

TypePtr CheckTypeList(const TypePtr &predicate, const TypePtrList &args_type_list) {
  MS_EXCEPTION_IF_NULL(predicate);
  for (auto arg_type : args_type_list) {
    MS_EXCEPTION_IF_NULL(arg_type);
    if (!CheckType(predicate, arg_type)) {
      MS_LOG(EXCEPTION) << "The expected is " << predicate->ToString() << ", not " << arg_type->ToString();
    }
  }
  return TypeJoin(args_type_list);
}

int GetPositiveAxis(int axis_value, size_t increment) {
  if (axis_value < 0) {
    axis_value = axis_value + SizeToInt(increment);
  }

  if (axis_value < 0) {
    MS_LOG(EXCEPTION) << "axis_value should not still <0";
  }

  return axis_value;
}

// Return if two shapes can be broadcast.
// Broadcast shape is placed in broadcast_output_shape.
std::vector<int> RealBroadcast(const std::string &op, std::vector<int> x_shape, std::vector<int> y_shape) {
  std::reverse(x_shape.begin(), x_shape.end());
  std::reverse(y_shape.begin(), y_shape.end());
  // Fill a placeholder value 1 which will be replaced later.
  size_t std_len = x_shape.size() > y_shape.size() ? x_shape.size() : y_shape.size();
  y_shape.resize(std_len, 1);
  x_shape.resize(std_len, 1);

  std::vector<int> broadcast_shape;
  for (size_t i = 0; i < std_len; i++) {
    int x_i = x_shape[i];  // i-th dimension of x
    int y_i = y_shape[i];  // i-th dimension of y
    int output_i = 0;      // i-th dimension of the output
    if (x_i == y_i) {
      output_i = x_i;
    } else if (x_i == 1) {
      output_i = y_i;
    } else if (y_i == 1) {
      output_i = x_i;
    } else {
      MS_LOG(EXCEPTION)
        << "" << op
        << " evaluator the shape of first tensor and the shape of second tensor do not meet the broadcasting "
           "requirements";
    }
    broadcast_shape.push_back(output_i);
  }
  std::reverse(broadcast_shape.begin(), broadcast_shape.end());
  return broadcast_shape;
}

ShapePtr GetBroadcastShape(const std::string &op, const AbstractTensorPtr &tensor_x,
                           const AbstractTensorPtr &tensor_y) {
  mindspore::abstract::ShapePtr tensor_x_shape = tensor_x->shape();
  mindspore::abstract::ShapePtr tensor_y_shape = tensor_y->shape();
  // if is the same shape ,just return the x_shape
  if (*tensor_x_shape == *tensor_y_shape) {
    return tensor_x_shape;
  }
  auto x_shape = tensor_x_shape->shape();
  auto y_shape = tensor_y_shape->shape();
  return std::make_shared<Shape>(RealBroadcast(op, x_shape, y_shape));
}
}  // namespace abstract
}  // namespace mindspore
