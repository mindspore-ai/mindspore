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
#include "abstract/abstract_function.h"

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

bool IsShapesDynamicRank(const std::vector<ShapeVector> &shapes) {
  return std::any_of(shapes.begin(), shapes.end(), [](const ShapeVector &shape) {
    return std::any_of(shape.begin(), shape.end(), [](int64_t dim) { return dim == Shape::kShapeRankAny; });
  });
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
  auto shape_max = shape1->shape();
  auto shape_min = shape2->shape();
  if (shape2->shape().size() > shape1->shape().size()) {
    shape_max = shape2->shape();
    shape_min = shape1->shape();
  }
  // (3, -1) join (3, 4, -1) = Error
  // (3, -1) join (3, -1, -1) = (-2)
  for (size_t i = 0; i < shape_max.size(); ++i) {
    if (i < shape_min.size()) {
      if (shape_min[i] != shape_max[i]) {
        return nullptr;
      }
      continue;
    }
    if (shape_max[i] != Shape::kShapeDimAny) {
      return nullptr;
    }
  }
  return std::make_shared<Shape>(ShapeVector({Shape::kShapeRankAny}));
}

ShapeValueDType SingleShapeValueJoin(const ShapeValueDType &shape_value1, const ShapeValueDType &shape_value2) {
  if (shape_value1 == shape_value2) {
    return shape_value1;
  }
  // (-1, 2) join (1, -1) should be (-1, -1)
  if (shape_value1 == Shape::kShapeDimAny || shape_value2 == Shape::kShapeDimAny) {
    return Shape::kShapeDimAny;
  }
  // shape_value1 != shape_value2
  return Shape::kShapeError;
}

ShapePtr ShapeJoin(const ShapePtr &shape1, const ShapePtr &shape2) {
  MS_EXCEPTION_IF_NULL(shape1);
  MS_EXCEPTION_IF_NULL(shape2);
  if (*shape1 == *shape2) {
    return shape1;
  }

  bool has_dynamic_rank = IsShapesDynamicRank({shape1->shape(), shape2->shape()});
  if (has_dynamic_rank) {
    return std::make_shared<Shape>(ShapeVector{Shape::kShapeRankAny});
  }
  // lengths of two shapes are not same, join failed
  if (shape1->shape().size() != shape2->shape().size()) {
    auto joined_shape = SingleElementShapeJoin(shape1, shape2);
    if (joined_shape != nullptr) {
      return joined_shape;
    }
    joined_shape = DifferentSizeDynShapeJoin(shape1, shape2);
    if (joined_shape != nullptr) {
      return joined_shape;
    }
    return nullptr;
  }
  ShapeVector dims(shape1->shape().size());
  for (std::size_t i = 0; i < shape1->shape().size(); i++) {
    auto joined_shape_value = SingleShapeValueJoin(shape1->shape()[i], shape2->shape()[i]);
    if (joined_shape_value == Shape::kShapeError) {
      return nullptr;
    }
    dims[i] = joined_shape_value;
  }
  return std::make_shared<Shape>(dims);
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

AbstractBasePtr AbstractBroaden(const AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<AbstractSequence>() && !abs->isa<AbstractSparseTensor>()) {
    auto sequence_abs = abs->cast<AbstractSequencePtr>();
    if (sequence_abs->dynamic_len()) {
      auto elem_abs = sequence_abs->dynamic_len_element_abs();
      if (elem_abs != nullptr && elem_abs->isa<AbstractScalar>()) {
        elem_abs->cast<AbstractScalarPtr>()->set_is_variable(true);
      }
      return abs->Broaden();
    }
    std::vector<AbstractBasePtr> new_elements;
    new_elements.reserve(sequence_abs->elements().size());
    (void)std::transform(sequence_abs->elements().cbegin(), sequence_abs->elements().cend(),
                         std::back_inserter(new_elements), AbstractBroaden);
    if (sequence_abs->isa<AbstractTuple>()) {
      return std::make_shared<AbstractTuple>(new_elements, sequence_abs->sequence_nodes());
    }
    if (sequence_abs->isa<AbstractList>()) {
      return std::make_shared<AbstractList>(new_elements, sequence_abs->sequence_nodes());
    }
    MS_EXCEPTION(TypeError) << "Unknown AbstractSequence type:" << abs->ToString();
  }
  if (abs->isa<AbstractScalar>()) {
    auto arg_type = abs->BuildType();
    MS_EXCEPTION_IF_NULL(arg_type);
    auto abs_scalar = abs->cast<AbstractScalarPtr>();
    if (arg_type->isa<Number>()) {
      abs_scalar->set_is_variable(true);
    }
  }
  return abs->Broaden();
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

AbstractBasePtr MakeAbstractTensor(const ShapePtr &shape, const TypePtr &type) {
  MS_EXCEPTION_IF_NULL(shape);
  MS_EXCEPTION_IF_NULL(type);
  AbstractBasePtr tensor = nullptr;

  auto ret_shape = shape->Clone();
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
    return MakeAbstractTensor(shape, type);
  } else if (base_shape->isa<NoShape>() && type->isa<Number>()) {
    return std::make_shared<abstract::AbstractScalar>(kAnyValue, type);
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
    MS_LOG(EXCEPTION) << "Evaluator return invalid shape " << base_shape->ToString() << " or type. "
                      << type->ToString();
  }
}

namespace {
FuncGraphPtr GetFuncGraphFromAbs(const abstract::AbstractBasePtr &abs, const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (abs == nullptr) {
    MS_LOG(ERROR) << "Null abstract, current node: " << anf_node->DebugString();
    return nullptr;
  }
  if (abs->isa<abstract::FuncGraphAbstractClosure>()) {
    auto abs_func_graph = abs->cast<abstract::FuncGraphAbstractClosurePtr>();
    MS_EXCEPTION_IF_NULL(abs_func_graph);
    if (!abs_func_graph->specialized()) {
      MS_LOG(INFO) << "Unspecilized func graph abstract: " << abs_func_graph->ToString()
                   << ", node: " << anf_node->DebugString();
    }
    return abs_func_graph->func_graph();
  }

  if (abs->isa<abstract::PartialAbstractClosure>()) {
    auto abs_partial_closure = abs->cast<abstract::PartialAbstractClosurePtr>();
    MS_EXCEPTION_IF_NULL(abs_partial_closure);
    auto abs_func = abs_partial_closure->fn();
    return GetFuncGraphFromAbs(abs_func, anf_node);
  }
  MS_LOG(ERROR) << "Unexpected abs: " << abs->ToString();
  return nullptr;
}
}  // namespace

std::vector<FuncGraphPtr> GetFuncGraphsFromAbs(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (IsValueNode<FuncGraph>(anf_node)) {
    return {GetValueNode<FuncGraphPtr>(anf_node)};
  }
  auto abs = anf_node->abstract();
  if (abs == nullptr) {
    MS_LOG(ERROR) << "Null abstract, current node: " << anf_node->DebugString();
    return {};
  }
  if (!abs->isa<abstract::AbstractFunction>()) {
    MS_LOG(ERROR) << "Unexpected abs: " << abs->ToString() << ", anf_node: " << anf_node->DebugString();
    return {};
  }
  auto abs_func = abs->cast<abstract::AbstractFunctionPtr>();
  MS_EXCEPTION_IF_NULL(abs_func);
  std::vector<FuncGraphPtr> func_graphs;
  if (abs->isa<abstract::AbstractFuncUnion>()) {
    auto visit_func = [&func_graphs, &anf_node](const abstract::AbstractFuncAtomPtr &poss) {
      (void)func_graphs.emplace_back(GetFuncGraphFromAbs(poss, anf_node));
    };
    abs_func->Visit(visit_func);
  } else {
    (void)func_graphs.emplace_back(GetFuncGraphFromAbs(abs_func, anf_node));
  }
  bool exist_null_fg =
    std::any_of(func_graphs.cbegin(), func_graphs.cend(), [](const FuncGraphPtr &fg) { return fg == nullptr; });
  if (exist_null_fg) {
    MS_LOG(ERROR) << "Get func graphs from abstract failed!";
    return {};
  }
  return func_graphs;
}
}  // namespace abstract
}  // namespace mindspore
