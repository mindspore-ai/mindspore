/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include <string>
#include <set>
#include <vector>
#include <algorithm>
#include <memory>
#include "ops/op_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"
#include "utils/shape_utils.h"
namespace mindspore {
namespace ops {
std::vector<int64_t> CalBroadCastShape(std::vector<int64_t> x_shape, std::vector<int64_t> y_shape,
                                       const std::string &op_name, const std::string &op_x_name,
                                       const std::string &op_y_name) {
  if (x_shape == y_shape) {
    return x_shape;
  }
  constexpr int dynamic_rank_len = 1;
  constexpr int dynamic_rank_value = -2;
  if ((x_shape.size() == dynamic_rank_len && x_shape[0] == dynamic_rank_value) ||
      (y_shape.size() == dynamic_rank_len && y_shape[0] == dynamic_rank_value)) {
    return std::vector<int64_t>({dynamic_rank_value});
  }
  auto x_length = static_cast<int64_t>(x_shape.size());
  auto y_length = static_cast<int64_t>(y_shape.size());
  auto length = x_length < y_length ? x_length : y_length;
  std::vector<int64_t> broadcast_shape;
  if (x_length == length) {
    (void)std::copy(y_shape.begin(), y_shape.end() - length, std::back_inserter(broadcast_shape));
  } else {
    (void)std::copy(x_shape.begin(), x_shape.end() - length, std::back_inserter(broadcast_shape));
  }
  for (int64_t i = -length; i < 0; i++) {
    if (x_shape[LongToSize(x_length + i)] == 1) {
      (void)broadcast_shape.push_back(y_shape[LongToSize(y_length + i)]);
    } else if (y_shape[LongToSize(y_length + i)] == 1) {
      (void)broadcast_shape.push_back(x_shape[LongToSize(x_length + i)]);
    } else if (x_shape[x_length + i] == y_shape[LongToSize(y_length + i)]) {
      (void)broadcast_shape.push_back(x_shape[LongToSize(x_length + i)]);
    } else if ((x_shape[x_length + i] == abstract::Shape::SHP_ANY) ||
               (y_shape[y_length + i] == abstract::Shape::SHP_ANY)) {
      (void)broadcast_shape.push_back(abstract::Shape::SHP_ANY);
    } else {
      MS_EXCEPTION(ValueError) << "For '" << op_name << "', the two input '" << op_x_name << "' and '" << op_y_name
                               << "' can not broadcast";
    }
  }
  return broadcast_shape;
}
abstract::ShapePtr BroadCastInferShape(const std::string &op_name, const std::vector<AbstractBasePtr> &input_args) {
  MS_LOG(INFO) << "For '" << op_name << "', it's now doing infer shape.";
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShapeTrack());
  auto y_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->GetShapeTrack());
  auto x_shape = x_shape_map[kShape];
  auto y_shape = y_shape_map[kShape];
  auto x_min_shape = x_shape_map[kMinShape];
  auto x_max_shape = x_shape_map[kMaxShape];
  auto y_min_shape = y_shape_map[kMinShape];
  auto y_max_shape = y_shape_map[kMaxShape];

  if (x_shape == y_shape) {
    return std::make_shared<abstract::Shape>(x_shape, x_min_shape, x_max_shape);
  }
  auto broadcast_shape = CalBroadCastShape(x_shape, y_shape, op_name);
  bool is_x_dyn =
    std::any_of(x_shape.begin(), x_shape.end(), [](int64_t value) { return value == abstract::Shape::SHP_ANY; });
  bool is_y_dyn =
    std::any_of(y_shape.begin(), y_shape.end(), [](int64_t value) { return value == abstract::Shape::SHP_ANY; });
  if (is_x_dyn || is_y_dyn) {
    auto min_broadcast_shape = CalBroadCastShape(x_min_shape, y_min_shape, op_name);
    auto max_broadcast_shape = CalBroadCastShape(x_max_shape, y_max_shape, op_name);
    return std::make_shared<abstract::Shape>(broadcast_shape, min_broadcast_shape, max_broadcast_shape);
  }
  return std::make_shared<abstract::Shape>(broadcast_shape);
}
int64_t ReduceFuncCheckAxisInferImpl(const PrimitivePtr &prim, const int64_t &axis, const size_t dim) {
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

void ReduceFuncCalShapeInferImpl(const PrimitivePtr &primitive, ShapeVector *shape, const ShapeVector &x_shape,
                                 const ValuePtr &axis, bool keep_dims_value) {
  MS_EXCEPTION_IF_NULL(axis);
  MS_EXCEPTION_IF_NULL(shape);
  if (axis->isa<ValueTuple>() || axis->isa<ValueList>()) {
    auto axis_ptr_list =
      axis->isa<ValueTuple>() ? axis->cast<ValueTuplePtr>()->value() : axis->cast<ValueListPtr>()->value();
    if (!axis_ptr_list.size()) {
      if (keep_dims_value) (void)shape->insert(shape->end(), x_shape.size(), 1);
    } else {
      (void)shape->insert(shape->end(), x_shape.begin(), x_shape.end());
      ValuePtrList axis_items = axis_ptr_list;
      ValuePtrList::iterator it;
      if (keep_dims_value) {
        for (it = axis_items.begin(); it != axis_items.end(); ++it) {
          auto axis_value = ReduceFuncCheckAxisInferImpl(primitive, GetValue<int64_t>(*it), x_shape.size());
          shape->at(LongToSize(axis_value)) = 1;
        }
      } else {
        std::vector<int64_t> axis_value_list;
        for (it = axis_items.begin(); it != axis_items.end(); ++it) {
          auto axis_value = GetValue<int64_t>(*it);
          auto axis_positive_value = ReduceFuncCheckAxisInferImpl(primitive, axis_value, x_shape.size());
          axis_value_list.push_back(axis_positive_value);
        }
        std::sort(axis_value_list.begin(), axis_value_list.end());
        std::vector<int64_t>::reverse_iterator it_re;
        for (it_re = axis_value_list.rbegin(); it_re != axis_value_list.rend(); ++it_re) {
          (void)shape->erase(shape->begin() + *it_re);
        }
      }
    }
  } else if (axis->isa<Int32Imm>() || axis->isa<Int64Imm>()) {
    (void)shape->insert(shape->end(), x_shape.begin(), x_shape.end());
    int64_t axis_value = GetValue<int64_t>(axis);
    axis_value = ReduceFuncCheckAxisInferImpl(primitive, axis_value, x_shape.size());
    if (keep_dims_value) {
      shape->at(axis_value) = 1;
    } else {
      (void)shape->erase(shape->begin() + axis_value);
    }
  } else {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', 'axis' must be one of these types: [int/tuple/list].";
  }
  return;
}

bool CheckTensorShapeValid(const std::vector<abstract::AbstractBasePtr> &input_args, const uint64_t input_num_ascend) {
  return input_args.size() == input_num_ascend && input_args[1] && input_args[1]->isa<abstract::AbstractTensor>() &&
         input_args[1]->BuildValue() && input_args[1]->BuildValue()->isa<AnyValue>();
}

abstract::ShapePtr ReduceBaseInferShape(const PrimitivePtr &primitive,
                                        const std::vector<abstract::AbstractBasePtr> &input_args,
                                        const std::string &prim_name) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, 0);
  MS_EXCEPTION_IF_NULL(shape_ptr);
  auto input_shape = shape_ptr->shape();
  auto input_min_shape = shape_ptr->min_shape();
  auto input_max_shape = shape_ptr->max_shape();
  auto keep_dimis_value_ptr = primitive->GetAttr(kKeepDims);
  MS_EXCEPTION_IF_NULL(keep_dimis_value_ptr);
  if (!keep_dimis_value_ptr->isa<BoolImm>()) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', 'keep_dims' must be Bool.";
  }
  bool keep_dims = GetValue<bool>(keep_dimis_value_ptr);
  ShapeVector out_shape = {};
  ShapeVector out_min_shape = {};
  ShapeVector out_max_shape = {};
  int64_t max_v;
  if (shape_ptr->IsDynamic()) {
    max_v = *max_element(input_max_shape.begin(), input_max_shape.end());
  } else {
    max_v = *max_element(input_shape.begin(), input_shape.end());
  }
  const uint64_t input_num_ascend = 2;
  if (CheckTensorShapeValid(input_args, input_num_ascend)) {
    auto axis_tensor = input_args[1]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(axis_tensor);
    auto axis_tensor_shape = axis_tensor->shape();
    MS_EXCEPTION_IF_NULL(axis_tensor_shape);
    auto axis_shape = axis_tensor_shape->shape();
    if (axis_shape.size() == 1 && axis_shape[0] == -1 && !keep_dims) {
      out_shape.push_back(-2);  // -2 : input num ascend
      out_min_shape = input_min_shape;
      out_max_shape = input_max_shape;
    } else if (!keep_dims) {
      for (size_t i = 0; i < input_shape.size() - axis_shape.size(); ++i) {
        out_shape.push_back(-1);
        out_min_shape.push_back(1);
        out_max_shape.push_back(max_v);
      }
    } else {
      for (size_t i = 0; i < input_shape.size(); ++i) {
        out_shape.push_back(-1);
        out_min_shape.push_back(1);
        out_max_shape.push_back(max_v);
      }
    }
    return std::make_shared<abstract::Shape>(out_shape, out_min_shape, out_max_shape);
  } else {
    ValuePtr axis_value;
    ValuePtr axis_ptr;
    if (input_args.size() == input_num_ascend && input_args[1]) {
      axis_ptr = input_args[1]->BuildValue();
    } else {
      axis_ptr = primitive->GetAttr("axis");
    }
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
    ReduceFuncCalShapeInferImpl(primitive, &out_shape, input_shape, axis_value, keep_dims);

    if (!input_min_shape.empty() && !input_max_shape.empty()) {
      ShapeVector shape_min = {};
      ShapeVector shape_max = {};
      ReduceFuncCalShapeInferImpl(primitive, &shape_min, input_min_shape, axis_value, keep_dims);
      ReduceFuncCalShapeInferImpl(primitive, &shape_max, input_max_shape, axis_value, keep_dims);
      return std::make_shared<abstract::Shape>(out_shape, shape_min, shape_max);
    }
    return std::make_shared<abstract::Shape>(out_shape);
  }
}

TypePtr ReduceBaseInferType(const PrimitivePtr &prim, const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto x_type = input_args[0]->BuildType();
  std::set<TypePtr> valid_types = common_valid_types;
  valid_types.insert(kBool);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x dtype", x_type, valid_types, prim->name());
  return x_type;
}

bool ObscureShapeEqual(const ShapeVector &lhs, const ShapeVector &rhs) {
  if (lhs == rhs) {
    return true;
  }
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i] != rhs[i] && lhs[i] != -1 && rhs[i] != -1) {
      return false;
    }
  }
  return true;
}

// Shape value infer mechanism implementation
namespace {
std::vector<ShapeVector> GetArgsShapeValue(const PrimitivePtr &prim, const AbstractBasePtrList &args) {
  std::vector<ShapeVector> args_shape_list;
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i]->isa<abstract::AbstractTensor>()) {
      auto arg_tensor = dyn_cast<abstract::AbstractTensor>(args[i]);
      auto shape_value_ptr = arg_tensor->get_shape_value();
      if (shape_value_ptr != nullptr) {
        auto shape_value = CheckAndConvertUtils::CheckTupleInt("shape_value", shape_value_ptr, prim->name());
        args_shape_list.push_back(shape_value);
        MS_LOG(DEBUG) << "Input shape value: " << shape_value;
      }
    } else if (args[i]->isa<abstract::AbstractSequence>()) {
      auto arg_list = args[i]->cast<abstract::AbstractSequencePtr>();
      auto &elements = arg_list->elements();
      auto tuple_args_shape_list = GetArgsShapeValue(prim, elements);
      args_shape_list.insert(args_shape_list.end(), tuple_args_shape_list.begin(), tuple_args_shape_list.end());
    }
  }
  return args_shape_list;
}

ValuePtr TransShapeToTensorValue(const ShapeVector &shape) {
  MS_EXCEPTION_IF_CHECK_FAIL(!shape.empty(), "Empty shape vector cannot be handled");
  constexpr size_t kType64Len = 8;
  int64_t shape_dim = SizeToLong(shape.size());
  std::vector<int64_t> shape_vec_shape = {shape_dim};
  auto new_value = std::make_shared<tensor::Tensor>(kNumberTypeInt64, shape_vec_shape);
  auto data_c = new_value->data_c();
  auto elem_num = shape.size() * kType64Len;
  auto ret_code = memcpy_s(data_c, static_cast<size_t>(new_value->data().nbytes()), &shape[0], elem_num);
  if (ret_code == 0) {
    return new_value;
  }
  return nullptr;
}

AbstractBasePtrList ConstructArgs(const std::vector<ShapeVector> &args_shape_list, const AbstractBasePtrList &ori_args,
                                  int *shape_index) {
  AbstractBasePtrList new_args;
  for (size_t i = 0; i < ori_args.size(); ++i) {
    auto new_arg = ori_args[i]->Clone();
    if (new_arg->isa<abstract::AbstractTensor>()) {
      auto arg_tensor = dyn_cast<abstract::AbstractTensor>(new_arg);
      if (arg_tensor->get_shape_value() != nullptr) {
        auto new_shape = args_shape_list[*shape_index];
        auto new_value = TransShapeToTensorValue(new_shape);
        arg_tensor->set_value(new_value);
        *shape_index = *shape_index + 1;
      }
    } else if (new_arg->isa<abstract::AbstractSequence>()) {
      auto arg_list = new_arg->cast<abstract::AbstractSequencePtr>();
      auto &elements = arg_list->elements();
      auto new_tuple_args = ConstructArgs(args_shape_list, elements, shape_index);
      new_arg = std::make_shared<abstract::AbstractTuple>(new_tuple_args);
    }
    new_args.push_back(new_arg);
  }
  return new_args;
}

ShapeVector RunCInferShapeValue(const PrimitivePtr &prim, const AbstractBasePtrList &args) {
  ShapeVector shape_value;
  auto eval_impl = abstract::GetPrimitiveInferImpl(prim);
  if (eval_impl.IsImplInferValue()) {
    auto value = eval_impl.InferValue(prim, args);
    if (value != nullptr) {
      shape_value = CheckAndConvertUtils::CheckTensorIntValue("shape", value, prim->name());
      MS_LOG(DEBUG) << "Inferred shape value: " << shape_value;
    }
  }
  return shape_value;
}

ShapeVector MakeFakeShape(const ShapeVector &shape) {
  ShapeVector new_shape = shape;
  for (unsigned int i = 0; i < new_shape.size(); ++i) {
    if (new_shape[i] < 0) {
      new_shape[i] = 1;
    }
  }
  return new_shape;
}

ShapeVector MakeFakeShapeBack(const PrimitivePtr &prim, const ShapeVector &shape, const ShapeVector &unknown_shape) {
  ShapeVector new_shape = shape;
  MS_EXCEPTION_IF_CHECK_FAIL(shape.size() == unknown_shape.size(),
                             "Input and output shape size must be consistent for element-wise op");
  for (unsigned int i = 0; i < unknown_shape.size(); i++) {
    if (unknown_shape[i] < 0) {
      new_shape[i] = -1;
    }
  }
  return new_shape;
}

std::vector<ShapeVector> ConvertShape(const std::vector<ShapeVector> &shapes) {
  std::vector<ShapeVector> converted_shapes;
  (void)std::transform(shapes.begin(), shapes.end(), std::back_inserter(converted_shapes),
                       [](const ShapeVector &shape) -> ShapeVector { return MakeFakeShape(shape); });
  return converted_shapes;
}

bool HasInferValue(const PrimitivePtr &prim) {
  auto eval_impl = abstract::GetPrimitiveInferImpl(prim);
  if (eval_impl.IsImplInferValue()) {
    return true;
  }
  return false;
}

ValuePtr EvalShapeTensorValue(const PrimitivePtr &prim, const AbstractBasePtrList &args, bool convert_shape = false) {
  MS_LOG(DEBUG) << prim->name() << " has infer shape value";
  if (!HasInferValue(prim)) {
    return nullptr;
  }
  std::vector<ShapeVector> arg_shapes = GetArgsShapeValue(prim, args);
  if (arg_shapes.empty()) {
    return nullptr;
  }
  std::vector<ShapeVector> converted_shapes = arg_shapes;
  if (convert_shape) {
    converted_shapes = ConvertShape(arg_shapes);
  }
  int shape_index = 0;
  AbstractBasePtrList new_args = ConstructArgs(converted_shapes, args, &shape_index);
  auto shape_value = RunCInferShapeValue(prim, new_args);
  if (shape_value.empty()) {
    return nullptr;
  }
  if (convert_shape) {
    shape_value = MakeFakeShapeBack(prim, shape_value, arg_shapes[0]);
    MS_LOG(DEBUG) << "Convert back shape: " << shape_value;
  }
  return MakeValue(shape_value);
}
}  // namespace

ShapeVector GetShapeValue(const PrimitivePtr &primitive, const AbstractBasePtr &arg) {
  auto abs_value = arg->BuildValue();
  MS_EXCEPTION_IF_NULL(abs_value);
  if (arg->isa<abstract::AbstractTensor>()) {
    auto abs_tensor = arg->cast<abstract::AbstractTensorPtr>();
    if (abs_value->isa<tensor::Tensor>()) {
      auto shape_value = CheckAndConvertUtils::CheckTensorIntValue("shape", abs_value, "");
      MS_EXCEPTION_IF_CHECK_FAIL(!shape_value.empty(), "Data type of shape value must be int64_t or int32_t");
      return shape_value;
    }
    auto abs_tensor_shape = abs_tensor->shape()->shape();
    MS_EXCEPTION_IF_CHECK_FAIL(abs_tensor_shape.size() == 1, "Shape of shape value only could be one-dimensional");
    if (IsDynamic(abs_tensor_shape)) {
      return {UNKNOWN_RANK};
    }
    auto shape_size = abs_tensor_shape[0];
    auto shape_value = abs_tensor->get_shape_value();
    if (shape_value == nullptr) {
      return ShapeVector(shape_size, UNKNOWN_DIM);
    } else {
      auto shape_vector = GetValue<ShapeVector>(shape_value);
      MS_EXCEPTION_IF_CHECK_FAIL(LongToSize(shape_size) == shape_vector.size(), "Illegal shape of shape value");
      return shape_vector;
    }
  } else if (arg->isa<abstract::AbstractTuple>()) {
    auto elements = arg->cast<abstract::AbstractTuplePtr>()->elements();
    MS_EXCEPTION_IF_CHECK_FAIL(!elements.empty() && !elements[0]->isa<abstract::AbstractTensor>(),
                               "Input cannot be a tuple of tensor");
    auto out_shape = GetValue<std::vector<int64_t>>(abs_value);
    return out_shape;
  }

  auto size_type = arg->BuildType();
  MS_EXCEPTION_IF_NULL(size_type);
  MS_EXCEPTION(TypeError) << "For " << primitive->name() << "Input arg must be abstract tensor or tuple, but got"
                          << size_type->ToString() << ".";
}

ValuePtr InferMakeShapeTensorValue(const PrimitivePtr &prim, const AbstractBasePtrList &args) {
  return EvalShapeTensorValue(prim, args, false);
}

ValuePtr InferComputeShapeTensorValue(const PrimitivePtr &prim, const AbstractBasePtrList &args) {
  return EvalShapeTensorValue(prim, args, true);
}
}  // namespace ops
}  // namespace mindspore
