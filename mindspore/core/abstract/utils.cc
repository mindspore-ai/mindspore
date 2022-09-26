/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

inline bool IsMaxOrMinEmpty(const ShapePtr &shape1, const ShapePtr &shape2) {
  if (shape1->max_shape().empty() || shape1->min_shape().empty() || shape2->max_shape().empty() ||
      shape2->min_shape().empty()) {
    return true;
  }

  return false;
}

ShapePtr CalculateDynamicShape(const ShapePtr &shape1, const ShapePtr &shape2, const ShapeVector &dims) {
  // calculate dynamic shape
  ShapeVector min_dims(dims.size());
  ShapeVector max_dims(dims.size());
  MS_EXCEPTION_IF_NULL(shape1);
  MS_EXCEPTION_IF_NULL(shape2);

  if (IsMaxOrMinEmpty(shape1, shape2)) {
    return std::make_shared<Shape>(dims);
  }

  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] != Shape::kShapeDimAny) {
      min_dims[i] = max_dims[i] = dims[i];
      continue;
    }
    if (shape1->shape()[i] != Shape::kShapeDimAny && shape2->shape()[i] != Shape::kShapeDimAny) {
      min_dims[i] = std::min(shape1->shape()[i], shape2->shape()[i]);
      max_dims[i] = std::max(shape1->shape()[i], shape2->shape()[i]);
      continue;
    }
    if (shape1->shape()[i] == Shape::kShapeDimAny && shape2->shape()[i] != Shape::kShapeDimAny) {
      if (shape1->min_shape().size() <= i || shape1->max_shape().size() <= i) {
        MS_EXCEPTION(ValueError) << "Shape " << shape1->ToString()
                                 << " has dynamic shape, but does not have min/max shape info.";
      }
      min_dims[i] = std::min(shape1->min_shape()[i], shape2->shape()[i]);
      max_dims[i] = std::max(shape1->max_shape()[i], shape2->shape()[i]);
      continue;
    }
    if (shape1->shape()[i] != Shape::kShapeDimAny && shape2->shape()[i] == Shape::kShapeDimAny) {
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

bool IsShapesDynamicRank(const std::vector<ShapeVector> &shapes) {
  return std::any_of(shapes.begin(), shapes.end(), [](const ShapeVector &shape) {
    return std::any_of(shape.begin(), shape.end(), [](int64_t dim) { return dim == Shape::kShapeRankAny; });
  });
}

bool HasSpecialShape(const std::vector<ShapePtr> &shapes) {
  for (const auto &shape : shapes) {
    bool shape_dyn =
      std::any_of(shape->shape().begin(), shape->shape().end(), [](int64_t dim) { return dim == Shape::kShapeDimAny; });
    if (shape_dyn && shape->min_shape().empty() && shape->max_shape().empty()) {
      return true;
    }
  }
  return false;
}

ShapePtr SingleElementShapeJoin(const ShapePtr &shape1, const ShapePtr &shape2) {
  // special case: shape(1), shape() -> shape(1)
  if (shape1->shape().size() == 1 && shape1->shape()[0] == 1 && shape2->shape().empty()) {
    return shape1;
  }
  if (shape2->shape().size() == 1 && shape2->shape()[0] == 1 && shape1->shape().empty()) {
    return shape2;
  }
  return nullptr;
}

// If shape sizes are not equal, but shape1 and shape2 are all dynamic shape, return a dynamic rank.
ShapePtr DifferentSizeDynShapeJoin(const ShapePtr &shape1, const ShapePtr &shape2) {
  auto shape_vec1 = shape1->shape();
  auto shape_vec2 = shape2->shape();
  if (!IsDynamicShape(shape_vec1)) {
    return nullptr;
  }
  if (!IsDynamicShape(shape_vec2)) {
    return nullptr;
  }
  return std::make_shared<Shape>(ShapeVector({Shape::kShapeRankAny}));
}

ShapePtr ShapeJoin(const ShapePtr &shape1, const ShapePtr &shape2) {
  MS_EXCEPTION_IF_NULL(shape1);
  MS_EXCEPTION_IF_NULL(shape2);
  constexpr int64_t kDynamicRankShape = -2;
  if (*shape1 == *shape2) {
    return shape1;
  }
  ShapeVector dims;
  bool has_dynamic_shape = false;
  bool no_min_max_shape = false;
  bool has_dynamic_rank = IsShapesDynamicRank({shape1->shape(), shape2->shape()});
  if (has_dynamic_rank) {
    (void)dims.emplace_back(kDynamicRankShape);
    return std::make_shared<Shape>(dims);
  }
  // lengths of two shapes are not same, join failed
  if (shape1->shape().size() != shape2->shape().size()) {
    auto joined_shape = SingleElementShapeJoin(shape1, shape2);
    if (joined_shape != nullptr) {
      return joined_shape;
    }
    joined_shape = DifferentSizeDynShapeJoin(shape1, shape1);
    if (joined_shape != nullptr) {
      return joined_shape;
    }
    return nullptr;
  }
  dims.resize(shape1->shape().size());
  for (std::size_t i = 0; i < shape1->shape().size(); i++) {
    if (shape1->shape()[i] == shape2->shape()[i]) {
      dims[i] = shape1->shape()[i];
      if (shape1->shape()[i] == Shape::kShapeDimAny) {
        has_dynamic_shape = true;
      }
    } else {
      dims[i] = Shape::kShapeDimAny;
      has_dynamic_shape = true;
    }
  }
  no_min_max_shape = HasSpecialShape({shape1, shape2});
  if (!has_dynamic_shape || no_min_max_shape) {
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
  auto f_spec = dyn_cast_ptr<AbstractFunction>(spec);
  if (f_spec != nullptr) {
    return std::make_shared<AbstractScalar>(kAnyValue, std::make_shared<EnvType>());
  }
  return spec->Clone();
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

void CheckMinMaxShape(const ShapeVector &shape, ShapeVector *min_shape, ShapeVector *max_shape) {
  *min_shape = (*min_shape).empty() ? shape : *min_shape;
  *max_shape = (*max_shape).empty() ? shape : *max_shape;
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
    auto tensor_type = type->cast_ptr<TensorType>();
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
    auto shape_tuple = base_shape->cast_ptr<TupleShape>();
    auto type_tuple = type->cast_ptr<Tuple>();
    AbstractBasePtrList ptr_list;
    for (size_t it = 0; it < shape_tuple->size(); ++it) {
      auto tensor_it = MakeAbstract((*shape_tuple)[it], (*type_tuple)[it]);
      ptr_list.push_back(tensor_it);
    }
    auto tuple = std::make_shared<abstract::AbstractTuple>(ptr_list);
    return tuple;
  } else if (base_shape->isa<ListShape>() && type->isa<List>()) {
    auto shape_list = base_shape->cast_ptr<ListShape>();
    auto type_list = type->cast_ptr<List>();
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
    MS_LOG(EXCEPTION) << "Evaluator return invalid shape " << base_shape->ToString() << "or type. " << type->ToString();
  }
}
}  // namespace abstract
}  // namespace mindspore
