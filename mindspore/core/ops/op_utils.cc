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
#include <map>
#include <set>
#include <vector>
#include <algorithm>
#include <memory>
#include <utility>

#include "ops/op_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "utils/shape_utils.h"
#include "abstract/param_validator.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/named.h"
#include "ir/primitive.h"
#include "ir/scalar.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "mindapi/base/type_id.h"
#include "ops/op_name.h"
#include "utils/convert_utils_base.h"
#include "mindapi/src/helper.h"

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
      broadcast_shape.push_back(y_shape[LongToSize(y_length + i)]);
    } else if (y_shape[LongToSize(y_length + i)] == 1) {
      broadcast_shape.push_back(x_shape[LongToSize(x_length + i)]);
    } else if (x_shape[LongToSize(x_length + i)] == y_shape[LongToSize(y_length + i)]) {
      broadcast_shape.push_back(x_shape[LongToSize(x_length + i)]);
    } else if ((x_shape[LongToSize(x_length + i)] == abstract::Shape::kShapeDimAny) ||
               (y_shape[LongToSize(y_length + i)] == abstract::Shape::kShapeDimAny)) {
      broadcast_shape.push_back(abstract::Shape::kShapeDimAny);
    } else {
      MS_EXCEPTION(ValueError) << "For '" << op_name << "', the two input '" << op_x_name << "' and '" << op_y_name
                               << "' with shape: " << x_shape << " and " << y_shape << " can not broadcast.";
    }
  }
  return broadcast_shape;
}
abstract::ShapePtr BroadCastInferShape(const std::string &op_name, const std::vector<AbstractBasePtr> &input_args) {
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShapeTrack());
  auto y_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->GetShapeTrack());
  auto x_shape = x_shape_map[kShape];
  auto y_shape = y_shape_map[kShape];

  // ToSupport Dynamic rank
  if (IsDynamicRank(x_shape) || IsDynamicRank(y_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }

  if (x_shape == y_shape) {
    return std::make_shared<abstract::Shape>(x_shape);
  }
  auto broadcast_shape = CalBroadCastShape(x_shape, y_shape, op_name);
  return std::make_shared<abstract::Shape>(broadcast_shape);
}
void ReduceFuncCheckAxisInferImpl(const PrimitivePtr &prim, std::vector<int64_t> *axis, const size_t dim) {
  MS_EXCEPTION_IF_NULL(axis);
  int64_t dim_ = static_cast<int64_t>(dim);
  for (size_t i = 0; i < axis->size(); i++) {
    if (dim == 0) {
      if ((axis->at(i) != -1 && axis->at(i) != 0)) {
        MS_EXCEPTION(ValueError) << "For '" << prim->name()
                                 << "', 'axis' must be in [-1, 0]. But got 'axis' = " << axis->at(i) << ".";
      }
      axis->at(i) = 0;
      continue;
    }
    if (axis->at(i) < -dim_ || axis->at(i) >= dim_) {
      MS_EXCEPTION(ValueError) << "For '" << prim->name() << "', 'axis' must be in [" << -dim_ << ", " << dim_
                               << "). But got 'axis' = " << axis->at(i) << ".";
    }
    if (axis->at(i) >= -dim_ && axis->at(i) < 0) {
      axis->at(i) += dim_;
    }
  }
}

ShapeVector ReduceFuncCalShapeInferImpl(const PrimitivePtr &primitive, const ShapeVector &x_shape,
                                        const std::vector<int64_t> &axis, bool keep_dims_value) {
  ShapeVector out_shape;
  ShapeVector axis_value;
  (void)axis_value.insert(axis_value.end(), axis.begin(), axis.end());
  (void)out_shape.insert(out_shape.end(), x_shape.begin(), x_shape.end());
  std::sort(axis_value.begin(), axis_value.end());
  auto last = std::unique(axis_value.begin(), axis_value.end());
  axis_value.erase(last, axis_value.end());
  if (keep_dims_value) {
    if (x_shape.size() == 0) {
      return {};
    }
    for (auto i : axis_value) {
      out_shape.at(LongToSize(i)) = 1;
    }
    if (axis_value.empty()) {
      for (size_t i = 0; i < out_shape.size(); i++) {
        out_shape.at(i) = 1;
      }
    }
    return out_shape;
  }
  if (axis.size() == 0 || x_shape.size() == 0) {
    return {};
  }
  std::vector<int64_t>::reverse_iterator it_re;
  for (it_re = axis_value.rbegin(); it_re != axis_value.rend(); ++it_re) {
    (void)out_shape.erase(out_shape.begin() + *it_re);
  }
  return out_shape;
}

ShapeVector ReduceFuncCalShapeAxisDyn(const ShapeVector &x_shape, const int64_t axis_shape, bool keep_dims) {
  ShapeVector out_shape;
  constexpr int dynamic_rank_value = -2;
  if (!keep_dims) {
    out_shape.push_back(dynamic_rank_value);
  } else {
    (void)out_shape.insert(out_shape.end(), x_shape.size(), -1LL);
  }
  return out_shape;
}

void CheckAndGetAxisValueFromAttr(const PrimitivePtr &primitive, std::vector<int64_t> *axis_value, int64_t *) {
  auto op_name = primitive->name();
  auto axis_ptr = primitive->GetAttr("axis");
  MS_EXCEPTION_IF_NULL(axis_ptr);
  if (axis_ptr->isa<tensor::Tensor>()) {
    *axis_value = CheckAndConvertUtils::CheckTensorIntValue("axis", axis_ptr, op_name);
  } else {
    *axis_value = CheckAndConvertUtils::CheckIntOrTupleInt("axis", axis_ptr, op_name);
  }
}

bool CheckAndGetAxisValueFromScalar(const ValuePtr &input_value, const std::string &op_name,
                                    std::vector<int64_t> *axis_value, int64_t *axis_shape_v) {
  *axis_shape_v = 1;
  bool is_dynamic = false;
  if (IsValueKnown(input_value)) {
    *axis_value = CheckAndConvertUtils::CheckIntOrTupleInt("axis", input_value, op_name);
  } else {
    is_dynamic = true;
  }
  return is_dynamic;
}

bool CheckAndGetAxisValueFromSequence(const abstract::AbstractBasePtr &abs, const ValuePtr &input_value,
                                      const std::string &op_name, std::vector<int64_t> *axis_value,
                                      int64_t *axis_shape_v) {
  bool is_dynamic = false;
  if (IsValueKnown(input_value)) {
    *axis_value = CheckAndConvertUtils::CheckIntOrTupleInt("axis", input_value, op_name);
    if (axis_value->empty()) {
      *axis_shape_v = 0;
    }
  } else {
    is_dynamic = true;
    auto seq_abs = abs->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(seq_abs);
    *axis_shape_v = seq_abs->dynamic_len() ? -1 : seq_abs->size();
  }

  return is_dynamic;
}

bool CheckAndGetAxisValueFromTensor(const std::vector<abstract::AbstractBasePtr> &input_args,
                                    const ValuePtr &input_value, const std::string &op_name,
                                    std::vector<int64_t> *axis_value, int64_t *axis_shape_v) {
  bool is_dynamic = false;
  (void)CheckAndConvertUtils::CheckTensorTypeValid("axis", input_args[kInputIndex1]->BuildType(), {kInt32, kInt64},
                                                   op_name);
  if (input_value->isa<tensor::Tensor>()) {
    *axis_value = CheckAndConvertUtils::CheckTensorIntValue("axis", input_value, op_name);
    if (axis_value->empty()) {
      *axis_shape_v = 0;
    }
  } else {
    is_dynamic = true;
    auto axis_shape = CheckAndConvertUtils::GetTensorInputShape(op_name, input_args, 1);
    if (axis_shape->shape().size() > 1) {
      MS_EXCEPTION(ValueError) << "For '" << op_name << "', the axis's shape length should be 1, but got '"
                               << axis_shape->shape().size() << "'.";
    } else if (axis_shape->shape().size() == 0) {
      *axis_shape_v = 1;
    } else {
      *axis_shape_v = axis_shape->shape()[0];
    }
  }
  return is_dynamic;
}

bool CheckAndGetAxisValue(const std::vector<abstract::AbstractBasePtr> &input_args, std::vector<int64_t> *axis_value,
                          int64_t *axis_shape_v, const PrimitivePtr &primitive) {
  MS_EXCEPTION_IF_NULL(axis_value);
  MS_EXCEPTION_IF_NULL(axis_shape_v);
  bool is_dynamic = false;
  const std::string &op_name = primitive->name();
  if (input_args.size() == 1) {
    CheckAndGetAxisValueFromAttr(primitive, axis_value, axis_shape_v);
    return false;
  }
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex1]);
  auto input_value = input_args[kInputIndex1]->BuildValue();
  if (input_args[kInputIndex1]->isa<abstract::AbstractScalar>()) {
    is_dynamic = CheckAndGetAxisValueFromScalar(input_value, op_name, axis_value, axis_shape_v);
  } else if (input_args[kInputIndex1]->isa<abstract::AbstractSequence>()) {
    is_dynamic =
      CheckAndGetAxisValueFromSequence(input_args[kInputIndex1], input_value, op_name, axis_value, axis_shape_v);
  } else if (input_args[kInputIndex1]->isa<abstract::AbstractTensor>()) {
    is_dynamic = CheckAndGetAxisValueFromTensor(input_args, input_value, op_name, axis_value, axis_shape_v);
  } else {
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', the second input type should be tensor or scalar, but got invalid abstract type:"
                             << input_args[kInputIndex1]->type_name() << ".";
  }
  return is_dynamic;
}

bool IsDynamicShapeSkipExecute(const bool skip_mode, const ShapeVector &axes_shape) {
  // Skip run ReduceSum when axis is a Empty Tensor
  if (std::any_of(axes_shape.begin(), axes_shape.end(), [](int64_t shape) { return shape == 0; }) && skip_mode) {
    return true;
  }
  return false;
}

abstract::ShapePtr ReduceBaseInferShape(const PrimitivePtr &primitive,
                                        const std::vector<abstract::AbstractBasePtr> &input_args,
                                        const std::string &prim_name) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, 0);
  MS_EXCEPTION_IF_NULL(shape_ptr);
  auto x_shape = shape_ptr->shape();
  bool skip_mode = false;
  if (primitive->HasAttr(kSkipMode)) {
    auto skip_mode_value_ptr = primitive->GetAttr(kSkipMode);
    MS_EXCEPTION_IF_NULL(skip_mode_value_ptr);
    skip_mode = GetValue<bool>(skip_mode_value_ptr);
  }
  auto keep_dimis_value_ptr = primitive->GetAttr(kKeepDims);
  MS_EXCEPTION_IF_NULL(keep_dimis_value_ptr);
  if (!keep_dimis_value_ptr->isa<BoolImm>()) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', 'keep_dims' must be Bool.";
  }
  bool keep_dims = GetValue<bool>(keep_dimis_value_ptr);
  std::vector<int64_t> axis_value;
  int64_t axis_shape = 1;
  bool axis_is_dynamic = CheckAndGetAxisValue(input_args, &axis_value, &axis_shape, primitive);
  if (IsDynamicShapeSkipExecute(skip_mode, {axis_shape})) {
    return std::make_shared<abstract::Shape>(x_shape);
  }
  ShapeVector out_shape = {};
  constexpr int dynamic_rank_value = -2;
  if (IsDynamicRank(x_shape)) {
    if (axis_shape == 0 && !keep_dims) {
      return std::make_shared<abstract::Shape>(out_shape);
    }
    out_shape.push_back(dynamic_rank_value);
    return std::make_shared<abstract::Shape>(out_shape);
  }
  if (axis_shape == -1 && !keep_dims) {
    out_shape.push_back(dynamic_rank_value);
    return std::make_shared<abstract::Shape>(out_shape);
  }
  ReduceFuncCheckAxisInferImpl(primitive, &axis_value, x_shape.size());

  if (axis_is_dynamic) {
    out_shape = ReduceFuncCalShapeAxisDyn(x_shape, axis_shape, keep_dims);
    return std::make_shared<abstract::Shape>(out_shape);
  }
  out_shape = ReduceFuncCalShapeInferImpl(primitive, x_shape, axis_value, keep_dims);
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr ReduceBaseInferType(const PrimitivePtr &prim, const std::vector<abstract::AbstractBasePtr> &input_args,
                            const std::set<TypePtr> &check_list) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto x_type = input_args[0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x dtype", x_type, check_list, prim->name());
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
  errno_t ret_code = memcpy_s(data_c, static_cast<size_t>(new_value->data().nbytes()), &shape[0], elem_num);
  if (ret_code == EOK) {
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
  auto eval_impl_opt = abstract::GetPrimitiveInferImpl(prim);
  if (eval_impl_opt.has_value()) {
    auto eval_impl = eval_impl_opt.value();
    if (eval_impl.IsImplInferValue()) {
      auto value = eval_impl.InferValue(prim, args);
      if (value != nullptr) {
        shape_value = CheckAndConvertUtils::CheckTensorIntValue("shape", value, prim->name());
        MS_LOG(DEBUG) << "Inferred shape value: " << shape_value;
      }
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

ShapeVector MakeFakeShapeBack(const ShapeVector &shape, const ShapeVector &unknown_shape) {
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
  auto eval_impl_opt = abstract::GetPrimitiveInferImpl(prim);
  if (eval_impl_opt.has_value()) {
    auto eval_impl = eval_impl_opt.value();
    if (eval_impl.IsImplInferValue()) {
      return true;
    }
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
    shape_value = MakeFakeShapeBack(shape_value, arg_shapes[0]);
    MS_LOG(DEBUG) << "Convert back shape: " << shape_value;
  }
  return MakeValue(shape_value);
}
}  // namespace

std::vector<int64_t> GetSequenceValue(const std::string &arg_name, const AbstractBasePtr &abs,
                                      const std::string &prim_name) {
  MS_EXCEPTION_IF_NULL(abs);
  auto abs_seq = dyn_cast<abstract::AbstractSequence>(abs);
  MS_EXCEPTION_IF_NULL(abs_seq);
  if (abs_seq->dynamic_len()) {
    return std::vector<int64_t>{abstract::Shape::kShapeRankAny};
  }
  std::vector<int64_t> out_shape;
  for (auto element : abs_seq->elements()) {
    auto element_val = element->BuildValue();
    if (element_val == kAnyValue) {
      out_shape.push_back(abstract::Shape::kShapeDimAny);
    } else if (element_val->isa<Int64Imm>()) {
      (void)out_shape.emplace_back(GetValue<ShapeValueDType>(element_val));
    } else if (element_val->isa<Int32Imm>()) {
      (void)out_shape.emplace_back(GetValue<int32_t>(element_val));
    } else {
      MS_EXCEPTION(TypeError) << "For primitive[" << prim_name << "], the " << arg_name
                              << " must be one of ['tuple', 'list'] with all Int elements, but got " << abs->ToString();
    }
  }
  return out_shape;
}

ShapeVector GetShapeValue(const PrimitivePtr &primitive, const AbstractBasePtr &arg) {
  auto abs_value = arg->BuildValue();
  MS_EXCEPTION_IF_NULL(abs_value);

  if (IsValueKnown(abs_value)) {
    if (abs_value->isa<tensor::Tensor>()) {
      auto shape_value = CheckAndConvertUtils::CheckTensorIntValue("shape", abs_value, "");
      return shape_value;
    } else if (abs_value->isa<ValueSequence>()) {
      auto out_shape = CheckAndConvertUtils::CheckIntOrTupleInt("input[shape]", abs_value, primitive->name());
      return out_shape;
    }
  } else if (arg->isa<abstract::AbstractTensor>()) {
    auto abs_tensor = arg->cast<abstract::AbstractTensorPtr>();
    auto abs_tensor_shape = abs_tensor->shape()->shape();
    MS_EXCEPTION_IF_CHECK_FAIL(abs_tensor_shape.size() == 1, "Shape of shape value only could be one-dimensional");
    if (IsDynamic(abs_tensor_shape)) {
      return {abstract::Shape::kShapeRankAny};
    }
    auto shape_size = abs_tensor_shape[0];
    auto shape_value = abs_tensor->get_shape_value();
    if (shape_value == nullptr) {
      return ShapeVector(shape_size, abstract::Shape::kShapeDimAny);
    } else {
      auto shape_vector = GetValue<ShapeVector>(shape_value);
      MS_EXCEPTION_IF_CHECK_FAIL(LongToSize(shape_size) == shape_vector.size(), "Illegal shape of shape value");
      return shape_vector;
    }
  } else if (arg->isa<abstract::AbstractSequence>()) {
    auto shape = GetSequenceValue("input[shape]", arg, primitive->name());
    return shape;
  }

  auto size_type = arg->BuildType();
  MS_EXCEPTION_IF_NULL(size_type);
  MS_EXCEPTION(TypeError) << "For " << primitive->name() << ", the input type must be Tensor/Tuple/List , but got"
                          << size_type->ToString() << ".";
}

ValuePtr InferMakeShapeTensorValue(const PrimitivePtr &prim, const AbstractBasePtrList &args) {
  return EvalShapeTensorValue(prim, args, false);
}

ValuePtr InferComputeShapeTensorValue(const PrimitivePtr &prim, const AbstractBasePtrList &args) {
  return EvalShapeTensorValue(prim, args, true);
}

void CheckSparseShape(ShapeVector sparse_shp, ShapeVector dense_shp) {
  constexpr auto csr_mul_batch_pos = 2;
  int dlen = SizeToInt(sparse_shp.size()) - SizeToInt(dense_shp.size());
  if (dlen < 0) {
    MS_EXCEPTION(ValueError) << "Currently, only support dense tensor broadcast to sparse tensor, "
                             << "but sparse tensor has " << sparse_shp.size() << " dimensions, "
                             << "and dense tensor has " << dense_shp.size() << " dimensions. ";
  }
  for (int i = 0; i < dlen; i++) {
    (void)dense_shp.insert(dense_shp.begin(), 1);
  }
  if (sparse_shp.size() != dense_shp.size()) {
    MS_LOG(EXCEPTION) << "Failure: sparse_shp.size() != dense_shp.size().";
  }
  if (sparse_shp.size() < 1) {
    MS_LOG(EXCEPTION) << "Failure: dense tensor and sparse tensor shapes cannot be zero.";
  }
  for (size_t i = 0; i < sparse_shp.size(); i++) {
    auto s = sparse_shp[i];
    auto d = dense_shp[i];
    if (i < csr_mul_batch_pos) {
      if (d != s && d != 1) {
        MS_EXCEPTION(ValueError) << "Dense shape cannot broadcast to sparse shape.";
      }
    } else {
      if (d != s) {
        MS_EXCEPTION(ValueError) << "Currently, sparse shape and dense shape must equal in feature dimensions.";
      }
    }
  }
}

void CheckSparseShape(const size_t shape_size, const size_t expected_dim, const std::string &arg_name) {
  if (shape_size != expected_dim) {
    MS_EXCEPTION(ValueError) << arg_name << " must be a " << expected_dim << "-dimensional tensor, but got a "
                             << shape_size << "-dimensional tensor.";
  }
}

void CheckSparseIndicesDtype(const TypePtr data_type, const std::string &arg_name) {
  if (!(data_type->equal(kInt16) || data_type->equal(kInt32) || data_type->equal(kInt64))) {
    MS_EXCEPTION(TypeError) << "The dtype of " << arg_name << " must be Int16 or Int32 or Int64, but got "
                            << data_type->ToString() << ".";
  }
}

void CheckSparseIndicesDtypeInt32(const TypePtr data_type, const std::string &arg_name) {
  if (!data_type->equal(kInt32)) {
    MS_EXCEPTION(TypeError) << "The dtype of " << arg_name << " only support Int32 for now, but got "
                            << data_type->ToString() << ".";
  }
}

ShapeVector ConvertToShapeVector(const abstract::AbstractTuplePtr &shape) {
  auto shape_value = shape->BuildValue()->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(shape_value);
  ShapeVector shape_vec;
  (void)std::transform(std::begin(shape_value->value()), std::end(shape_value->value()), std::back_inserter(shape_vec),
                       [](const ValuePtr &e) -> int64_t {
                         auto elem = GetValue<int64_t>(e);
                         return elem;
                       });
  return shape_vec;
}

template <typename T>
std::shared_ptr<T> InferSparseAttr(const PrimitivePtr &primitive, const AbstractBasePtrList &args_spec_list) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr size_t kSizeExpect = 1;
  if (args_spec_list.size() != kSizeExpect) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', the number of input should be " << kSizeExpect
                      << ", but got " << args_spec_list.size() << ".";
  }
  constexpr size_t kIndex = 0;
  auto abs = args_spec_list[kIndex];
  MS_EXCEPTION_IF_NULL(abs);
  // To avoid AbstractSparseTensors being generalized to AbstractTuple.
  if (dyn_cast<T>(abs) == nullptr) {
    auto abs_tuple = dyn_cast<abstract::AbstractTuple>(abs);
    if (abs_tuple != nullptr) {
      return std::make_shared<T>(abs_tuple->elements());
    }
  } else if (dyn_cast<T>(abs) != nullptr) {
    return dyn_cast<T>(abs);
  }
  MS_EXCEPTION(TypeError) << "For \'" << primitive->name() << "\', input[" << kIndex
                          << "] should be AbstractSparseTensor or AbstractTuple, but got "
                          << abs->BuildType()->ToString() << ".";
}
template std::shared_ptr<abstract::AbstractCSRTensor> InferSparseAttr(const PrimitivePtr &primitive,
                                                                      const AbstractBasePtrList &args_spec_list);
template std::shared_ptr<abstract::AbstractCOOTensor> InferSparseAttr(const PrimitivePtr &primitive,
                                                                      const AbstractBasePtrList &args_spec_list);

template <typename T>
AbstractBasePtr TensorToSequenceInfer(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr size_t input_len = 1;
  constexpr size_t input_0_index = 0;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_len, prim_name);

  auto shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, 0);
  MS_EXCEPTION_IF_NULL(shape_ptr);
  auto x_shape = shape_ptr->shape();
  if (x_shape.size() != 1) {
    MS_EXCEPTION(ValueError) << "For Primitive[" << prim_name << "], the input shape size must be 1, but got "
                             << x_shape << ".";
  }

  auto x_type = input_args[input_0_index]->BuildType();
  MS_EXCEPTION_IF_NULL(x_type);
  if (!x_type->isa<TensorType>()) {
    MS_EXCEPTION(TypeError) << "For Primitive[" << prim_name << "], the input must be a Tensor but got "
                            << x_type->ToString() << ".";
  }
  auto tensor_type = x_type->cast<TensorTypePtr>();
  const auto &element_type = tensor_type->element();
  MS_EXCEPTION_IF_NULL(element_type);
  AbstractBasePtrList abs_list;
  if (IsDynamic(x_shape)) {
    abs_list.push_back(std::make_shared<abstract::AbstractScalar>(kAnyValue, element_type));
    auto abs = std::make_shared<T>(abs_list);
    abs->CheckAndConvertToDynamicLenSequence();
    return abs;
  }
  for (int64_t i = 0; i < x_shape[0]; i++) {
    abs_list.push_back(std::make_shared<abstract::AbstractScalar>(kAnyValue, element_type));
  }
  auto abs = std::make_shared<T>(abs_list);
  return abs;
}
void CheckDynamicLengthSequenceSetItem(const std::string &op_name, const abstract::AbstractSequencePtr &queue,
                                       const AbstractBasePtr &target) {
  auto element_abs = queue->dynamic_len_element_abs();
  if (element_abs == nullptr) {
    MS_LOG(EXCEPTION) << "Empty variable len sequence can not setitem.";
  }
  const auto precondition_log = "For " + op_name + ", when the queue is dynamic length";
  const auto standard_abs_description = "element within dynamic length sequence";
  const auto differ_abs_description = "target element";
  CheckAndConvertUtils::CheckAbstractTypeAndShapeSame(std::vector<AbstractBasePtr>{element_abs, target},
                                                      precondition_log, standard_abs_description,
                                                      differ_abs_description);
}

template <typename T>
AbstractBasePtr InferSequenceSetItem(const PrimitivePtr &primitive, const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tuple or list, a scalar whose value is an int64 number and an object of a subclass of AbstractBase.
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  constexpr int args_spec_size = 3;
  constexpr size_t kIndex2 = 2;
  abstract::CheckArgsSize(op_name, args_spec_list, args_spec_size);
  auto queue = abstract::CheckArg<T>(op_name, args_spec_list, 0);
  auto index = abstract::CheckArg<abstract::AbstractScalar>(op_name, args_spec_list, 1);

  auto index_type = index->BuildType();
  MS_EXCEPTION_IF_NULL(index_type);
  if (index_type->type_id() != kInt64->type_id()) {
    MS_EXCEPTION(IndexError) << op_name << " evaluator index should be an int64 number, but got a "
                             << index_type->ToString() << " number.";
  }
  ValuePtr index_value = index->BuildValue();
  MS_EXCEPTION_IF_NULL(index_value);
  auto target = args_spec_list[kIndex2];
  MS_EXCEPTION_IF_NULL(target);
  if (queue->dynamic_len()) {
    CheckDynamicLengthSequenceSetItem(op_name, queue, target);
    return queue->Clone();
  }
  if (index_value == kAnyValue) {
    // If the index is variable and the sequence is constant length, then all of the element within the sequence
    // should have the same type and shape with the target input. The element within the return sequence should
    // be all broadened.
    const auto &elements = queue->elements();
    if (elements.size() == 0) {
      MS_LOG(EXCEPTION) << "Empty sequence can not setitem.";
    }
    const auto precondition_log = "For " + op_name + ", when the index is variable and the queue is constant length";
    CheckAndConvertUtils::CheckAbstractTypeAndShapeSame(elements, precondition_log);
    auto first_element = elements[0];
    const auto standard_abs_description = "element within constant length sequence";
    const auto differ_abs_description = "target element";
    CheckAndConvertUtils::CheckAbstractTypeAndShapeSame(std::vector<AbstractBasePtr>{first_element, target},
                                                        precondition_log, standard_abs_description,
                                                        differ_abs_description);
    return CheckAndConvertUtils::BroadenAllSequenceElements(queue);
  }
  auto index_int64_value = GetValue<int64_t>(index_value);
  AbstractBasePtrList elements = queue->elements();
  std::size_t nelems = elements.size();
  if (nelems == 0) {
    MS_EXCEPTION(IndexError) << "Can not setitem for an empty sequence.";
  }
  int64_t index_positive_value = index_int64_value >= 0 ? index_int64_value : index_int64_value + SizeToLong(nelems);
  if (index_positive_value < 0 || index_positive_value >= SizeToLong(nelems)) {
    MS_EXCEPTION(IndexError) << op_name << " evaluator the index: " << index_int64_value << " to set out of range: [-"
                             << nelems << "," << (nelems - 1) << "].";
  }
  size_t index_unsigned_value = LongToSize(index_positive_value);
  elements[index_unsigned_value] = args_spec_list[kIndex2];
  MS_LOG(DEBUG) << "SetItem use flags, index: " << index_unsigned_value << ", for " << queue->ToString();
  return std::make_shared<T>(elements, queue->sequence_nodes());
}

template AbstractBasePtr InferSequenceSetItem<abstract::AbstractList>(const PrimitivePtr &primitive,
                                                                      const AbstractBasePtrList &args_spec_list);
template AbstractBasePtr InferSequenceSetItem<abstract::AbstractTuple>(const PrimitivePtr &primitive,
                                                                       const AbstractBasePtrList &args_spec_list);

template AbstractBasePtr TensorToSequenceInfer<abstract::AbstractList>(const PrimitivePtr &primitive,
                                                                       const std::vector<AbstractBasePtr> &input_args);

template AbstractBasePtr TensorToSequenceInfer<abstract::AbstractTuple>(const PrimitivePtr &primitive,
                                                                        const std::vector<AbstractBasePtr> &input_args);

template <typename T>
T GetScalarValue(const std::string &op_name, const ValuePtr &elem) {
  T res;
  MS_EXCEPTION_IF_NULL(elem);
  if (elem->isa<Int64Imm>()) {
    auto elem_value = GetValue<int64_t>(elem);
    res = static_cast<T>(elem_value);
  } else if (elem->isa<Int32Imm>()) {
    auto elem_value = GetValue<int32_t>(elem);
    res = static_cast<T>(elem_value);
  } else if (elem->isa<FP64Imm>()) {
    auto elem_value = GetValue<double>(elem);
    res = static_cast<T>(elem_value);
  } else if (elem->isa<FP32Imm>()) {
    auto elem_value = GetValue<float>(elem);
    res = static_cast<T>(elem_value);
  } else if (elem->isa<BoolImm>()) {
    auto elem_value = GetValue<bool>(elem);
    res = static_cast<T>(elem_value);
  } else {
    MS_EXCEPTION(TypeError) << "For op '" << op_name
                            << "' input must be [int32, int64, float32, float64, bool], but got " << elem->ToString();
  }
  return res;
}

template int64_t GetScalarValue(const std::string &op_name, const ValuePtr &elem);
template int32_t GetScalarValue(const std::string &op_name, const ValuePtr &elem);
template double GetScalarValue(const std::string &op_name, const ValuePtr &elem);
template float GetScalarValue(const std::string &op_name, const ValuePtr &elem);
template bool GetScalarValue(const std::string &op_name, const ValuePtr &elem);

TypePtr HighPriorityType(const TypePtr &x_type, const TypePtr &y_type, const std::string &op_name) {
  static std::map<TypeId, size_t> prio_map = {{kNumberTypeFloat64, 1},
                                              {kNumberTypeFloat32, 2},
                                              {kNumberTypeInt64, 3},
                                              {kNumberTypeInt32, 4},
                                              {kNumberTypeBool, 5}};
  auto x_iter = prio_map.find(x_type->type_id());
  auto y_iter = prio_map.find(y_type->type_id());
  if (x_iter == prio_map.end() || y_iter == prio_map.end()) {
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', the x and y type should be int or float, but got x type: " << x_type
                             << " y type: " << y_type;
  }
  if (x_iter->second < y_iter->second) {
    return x_type;
  }
  if (x_iter->second == y_iter->second && x_iter->first == kNumberTypeBool) {
    return kInt32;
  }
  return y_type;
}

bool IsValueKnown(const ValuePtr &value) {
  // For now if the Abstract is a container of elements such as AbstractSequence and AbstractDictionary,
  // the BuildValue returns AnyValue if any one of the elements' value is AnyValue
  if (value->isa<AnyValue>() || value->isa<None>()) {
    return false;
  }

  return true;
}
}  // namespace ops
}  // namespace mindspore
