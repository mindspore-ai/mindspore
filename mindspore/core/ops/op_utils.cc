/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "ops/op_utils.h"

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "abstract/ops/primitive_infer_map.h"
#include "abstract/param_validator.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/named.h"
#include "ir/primitive.h"
#include "ir/scalar.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "ir/kernel_tensor_value.h"
#include "mindapi/base/type_id.h"
#include "mindapi/src/helper.h"
#include "ops/op_name.h"
#include "ops/op_def.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
std::vector<int64_t> CalBroadCastShape(const std::vector<int64_t> &x_shape, const std::vector<int64_t> &y_shape,
                                       const std::string &op_name, const std::string &op_x_name,
                                       const std::string &op_y_name) {
  if (x_shape == y_shape) {
    return x_shape;
  }

  if (IsDynamicRank(x_shape) || IsDynamicRank(y_shape)) {
    return {abstract::Shape::kShapeRankAny};
  }

  std::vector<int64_t> broadcast_shape;
  auto x_length = x_shape.size();
  auto y_length = y_shape.size();
  auto res = x_length > y_length;
  size_t max_len = res ? x_length : y_length;
  size_t min_len = res ? y_length : x_length;
  const std::vector<int64_t> &max_shape = res ? x_shape : y_shape;
  const std::vector<int64_t> &min_shape = res ? y_shape : x_shape;

  broadcast_shape = max_shape;
  auto miss = max_len - min_len;
  for (size_t i = 0; i < min_len; i++) {
    auto dst_i = miss + i;
    if (max_shape[dst_i] == 1) {
      broadcast_shape[dst_i] = min_shape[i];
    } else if (MS_UNLIKELY(max_shape[dst_i] == -1)) {
      if (min_shape[i] != 1) {
        broadcast_shape[dst_i] = min_shape[i];
      }
    } else if (MS_UNLIKELY(max_shape[dst_i] != min_shape[i] && min_shape[i] != -1 && min_shape[i] != 1)) {
      auto x_shape_name = op_x_name + ".shape";
      auto y_shape_name = op_y_name + ".shape";
      MS_EXCEPTION(ValueError) << "For '" << op_name << "', " << x_shape_name << " and " << y_shape_name
                               << " need to broadcast. The value of " << x_shape_name << "["
                               << std::to_string(x_length + i) << "] or " << y_shape_name << "["
                               << std::to_string(y_length + i)
                               << "] must be 1 or -1 when they are not the same, but got " << x_shape_name << " = "
                               << x_shape << " and " << y_shape_name << " = " << y_shape;
    }
  }
  return broadcast_shape;
}

abstract::ShapePtr BroadCastInferShape(const std::string &op_name, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(input_args[kIndex0]);
  MS_EXCEPTION_IF_NULL(input_args[kIndex1]);
  auto x_shape = GetShapeFromTensor(input_args[0]);
  auto y_shape = GetShapeFromTensor(input_args[1]);

  auto broadcast_shape = CalBroadCastShape(x_shape, y_shape, op_name);
  return std::make_shared<abstract::Shape>(broadcast_shape);
}

BaseShapePtr EltwiseGradInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(input_args[0]);
  MS_EXCEPTION_IF_NULL(input_args[0]->GetShape());
  return input_args[0]->GetShape()->Clone();
}

TypePtr EltwiseGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(input_args[0]);
  MS_EXCEPTION_IF_NULL(input_args[1]);
  auto grad_type = input_args[0]->GetType();
  MS_EXCEPTION_IF_NULL(grad_type);
  auto x_type = input_args[1]->GetType();
  MS_EXCEPTION_IF_NULL(x_type);
  if (grad_type->type_id() != x_type->type_id()) {
    MS_LOG_EXCEPTION << "For " << primitive->name()
                     << ", the grad type must be same as input type, but got grad_type: " << grad_type->ToString()
                     << " and x_type: " << x_type->ToString();
  }
  return grad_type->Clone();
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

ShapeVector ReduceFuncCalShapeInferImpl(const PrimitivePtr &, const ShapeVector &x_shape,
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

ShapeVector ReduceFuncCalShapeAxisDyn(const ShapeVector &x_shape, bool keep_dims) {
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
    *axis_shape_v = seq_abs->dynamic_len() ? -1 : SizeToLong(seq_abs->size());
  }

  return is_dynamic;
}

bool CheckAndGetAxisValueFromTensor(const std::vector<abstract::AbstractBasePtr> &input_args,
                                    const ValuePtr &input_value, const std::string &op_name,
                                    std::vector<int64_t> *axis_value, int64_t *axis_shape_v) {
  bool is_dynamic = false;
  (void)CheckAndConvertUtils::CheckTensorTypeValid("axis", input_args[kInputIndex1]->GetType(), {kInt32, kInt64},
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
  auto input_value = input_args[kInputIndex1]->GetValue();
  if (input_value->isa<KernelTensorValue>()) {
    auto value_opt = GetArrayValue<int64_t>(input_value);
    auto value_array = value_opt.value();
    *axis_value = value_array.ToVector();
  }
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
  auto x_shape = GetShapeFromTensor(input_args[0]);
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
    out_shape = ReduceFuncCalShapeAxisDyn(x_shape, keep_dims);
    return std::make_shared<abstract::Shape>(out_shape);
  }
  out_shape = ReduceFuncCalShapeInferImpl(primitive, x_shape, axis_value, keep_dims);
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr ReduceBaseInferType(const PrimitivePtr &prim, const std::vector<abstract::AbstractBasePtr> &input_args,
                            const std::set<TypePtr> &check_list) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto x_type = input_args[0]->GetType();
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
    auto element_val = element->GetValue();
    if (element_val->ContainsValueAny()) {
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
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto abs_value = arg->GetValue();
  MS_EXCEPTION_IF_NULL(abs_value);
  auto arg_type = arg->GetType();
  MS_EXCEPTION_IF_NULL(arg_type);

  if (IsValueKnown(abs_value)) {
    if (CheckAndConvertUtils::IsTensor(arg)) {
      return CheckAndConvertUtils::CheckTensorIntValue("shape", abs_value, "", arg_type);
    } else if (CheckAndConvertUtils::IsSequence(arg)) {
      return CheckAndConvertUtils::CheckIntOrTupleInt("input[shape]", arg, prim_name);
    }
  } else if (CheckAndConvertUtils::IsTensor(arg)) {
    auto arg_shape = arg->GetShape()->GetShapeVector();
    if (arg_shape.size() != 1) {
      MS_EXCEPTION(ValueError) << "For Primitive[" << primitive->name()
                               << "], Shape of shape value only could be one-dimensional";
    }
    if (IsDynamic(arg_shape)) {
      return {abstract::Shape::kShapeRankAny};
    }
    auto shape_size = arg_shape[0];
    return ShapeVector(shape_size, abstract::Shape::kShapeDimAny);
  } else if (arg->isa<abstract::AbstractSequence>()) {
    return GetSequenceValue("input[shape]", arg, prim_name);
  }

  MS_EXCEPTION(TypeError) << "For " << prim_name << ", the input type must be Tensor/Tuple/List , but got"
                          << arg_type->ToString() << ".";
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
  auto shape_value = shape->GetValue()->cast<ValueTuplePtr>();
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
std::shared_ptr<T> InferSparseAttr(const PrimitivePtr &primitive, const AbstractBasePtrList &args_abs_list) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr size_t kSizeExpect = 1;
  if (args_abs_list.size() != kSizeExpect) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', the number of input should be " << kSizeExpect
                      << ", but got " << args_abs_list.size() << ".";
  }
  constexpr size_t kIndex = 0;
  auto abs = args_abs_list[kIndex];
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
                          << "] should be AbstractSparseTensor or AbstractTuple, but got " << abs->GetType()->ToString()
                          << ".";
}
template std::shared_ptr<abstract::AbstractCSRTensor> InferSparseAttr(const PrimitivePtr &primitive,
                                                                      const AbstractBasePtrList &args_abs_list);
template std::shared_ptr<abstract::AbstractCOOTensor> InferSparseAttr(const PrimitivePtr &primitive,
                                                                      const AbstractBasePtrList &args_abs_list);

template <typename T>
AbstractBasePtr TensorToSequenceInfer(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_len = 1;
  constexpr size_t input_0_index = 0;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_len, prim_name);

  auto shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, 0);
  MS_EXCEPTION_IF_NULL(shape_ptr);
  auto x_shape = shape_ptr->shape();
  if (x_shape.size() != 1) {
    MS_EXCEPTION(ValueError) << "For Primitive[" << prim_name << "], the input shape size must be 1, but got "
                             << x_shape << ".";
  }

  auto x_type = input_args[input_0_index]->GetType();
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
    abs_list.push_back(std::make_shared<abstract::AbstractScalar>(kValueAny, element_type));
    auto abs = std::make_shared<T>(abs_list);
    abs->CheckAndConvertToDynamicLenSequence();
    return abs;
  }
  for (int64_t i = 0; i < x_shape[0]; i++) {
    abs_list.push_back(std::make_shared<abstract::AbstractScalar>(kValueAny, element_type));
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
AbstractBasePtr InferSequenceSetItem(const PrimitivePtr &primitive, const AbstractBasePtrList &args_abs_list) {
  // Inputs: a tuple or list, a scalar whose value is an int64 number and an object of a subclass of AbstractBase.
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  constexpr int args_spec_size = 3;
  constexpr size_t kIndex2 = 2;
  abstract::CheckArgsSize(op_name, args_abs_list, args_spec_size);
  auto queue = abstract::CheckArg<T>(op_name, args_abs_list, 0);
  auto index = abstract::CheckArg<abstract::AbstractScalar>(op_name, args_abs_list, 1);

  auto index_type = index->GetType();
  MS_EXCEPTION_IF_NULL(index_type);
  if (index_type->type_id() != kInt64->type_id()) {
    MS_EXCEPTION(TypeError) << op_name << " evaluator index should be an int64 number, but got a "
                            << index_type->ToString() << " number.";
  }
  ValuePtr index_value = index->GetValue();
  MS_EXCEPTION_IF_NULL(index_value);
  auto target = args_abs_list[kIndex2];
  MS_EXCEPTION_IF_NULL(target);
  if (queue->dynamic_len()) {
    CheckDynamicLengthSequenceSetItem(op_name, queue, target);
    return queue->Clone();
  }
  if (index_value->ContainsValueAny()) {
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
    MS_EXCEPTION(ValueError) << "Can not setitem for an empty sequence.";
  }
  int64_t index_positive_value = index_int64_value >= 0 ? index_int64_value : index_int64_value + SizeToLong(nelems);
  if (index_positive_value < 0 || index_positive_value >= SizeToLong(nelems)) {
    MS_EXCEPTION(IndexError) << op_name << " evaluator the index: " << index_int64_value << " to set out of range: [-"
                             << nelems << "," << (nelems - 1) << "].";
  }
  size_t index_unsigned_value = LongToSize(index_positive_value);
  elements[index_unsigned_value] = args_abs_list[kIndex2];
  MS_LOG(DEBUG) << "SetItem use flags, index: " << index_unsigned_value << ", for " << queue->ToString();
  return std::make_shared<T>(elements, queue->sequence_nodes());
}

template AbstractBasePtr InferSequenceSetItem<abstract::AbstractList>(const PrimitivePtr &primitive,
                                                                      const AbstractBasePtrList &args_abs_list);
template AbstractBasePtr InferSequenceSetItem<abstract::AbstractTuple>(const PrimitivePtr &primitive,
                                                                       const AbstractBasePtrList &args_abs_list);

template AbstractBasePtr TensorToSequenceInfer<abstract::AbstractList>(const PrimitivePtr &primitive,
                                                                       const std::vector<AbstractBasePtr> &input_args);

template AbstractBasePtr TensorToSequenceInfer<abstract::AbstractTuple>(const PrimitivePtr &primitive,
                                                                        const std::vector<AbstractBasePtr> &input_args);

template <typename T>
T GetScalarCastValue(const std::string &op_name, const ValuePtr &elem) {
  T res;
  MS_EXCEPTION_IF_NULL(elem);
  if (elem->isa<Int64Imm>()) {
    auto elem_value = GetValue<int64_t>(elem);
    res = static_cast<T>(elem_value);
  } else if (elem->isa<Int32Imm>()) {
    auto elem_value = GetValue<int32_t>(elem);
    res = static_cast<T>(elem_value);
  } else if (elem->isa<Int16Imm>()) {
    auto elem_value = GetValue<int16_t>(elem);
    res = static_cast<T>(elem_value);
  } else if (elem->isa<Int8Imm>()) {
    auto elem_value = GetValue<int8_t>(elem);
    res = static_cast<T>(elem_value);
  } else if (elem->isa<UInt64Imm>()) {
    auto elem_value = GetValue<uint64_t>(elem);
    res = static_cast<T>(elem_value);
  } else if (elem->isa<UInt32Imm>()) {
    auto elem_value = GetValue<uint32_t>(elem);
    res = static_cast<T>(elem_value);
  } else if (elem->isa<UInt16Imm>()) {
    auto elem_value = GetValue<uint16_t>(elem);
    res = static_cast<T>(elem_value);
  } else if (elem->isa<UInt8Imm>()) {
    auto elem_value = GetValue<uint8_t>(elem);
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

template int64_t GetScalarCastValue(const std::string &op_name, const ValuePtr &elem);
template int32_t GetScalarCastValue(const std::string &op_name, const ValuePtr &elem);
template int16_t GetScalarCastValue(const std::string &op_name, const ValuePtr &elem);
template int8_t GetScalarCastValue(const std::string &op_name, const ValuePtr &elem);
template uint64_t GetScalarCastValue(const std::string &op_name, const ValuePtr &elem);
template uint32_t GetScalarCastValue(const std::string &op_name, const ValuePtr &elem);
template uint16_t GetScalarCastValue(const std::string &op_name, const ValuePtr &elem);
template uint8_t GetScalarCastValue(const std::string &op_name, const ValuePtr &elem);
template double GetScalarCastValue(const std::string &op_name, const ValuePtr &elem);
template float GetScalarCastValue(const std::string &op_name, const ValuePtr &elem);
template bool GetScalarCastValue(const std::string &op_name, const ValuePtr &elem);

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
  if (value->ContainsValueAny() || value->isa<None>()) {
    return false;
  }

  return true;
}

std::set<int64_t> GetInputDependValueList(const PrimitivePtr &op_prim) {
  MS_EXCEPTION_IF_NULL(op_prim);
  std::set<int64_t> depend_list;
  mindspore::ops::OpDefPtr op_def = mindspore::ops::GetOpDef(op_prim->name());
  if (op_def == nullptr) {
    // Use old Primitive infer.
    auto op_infer_opt = abstract::GetPrimitiveInferImpl(op_prim);
    if (!op_infer_opt.has_value()) {
      if (op_prim->HasAttr(kAttrMeOpName)) {
        auto ori_prim_name = GetValue<std::string>(op_prim->GetAttr(kAttrMeOpName));
        op_infer_opt = abstract::GetPrimitiveInferImpl(std::make_shared<Primitive>(ori_prim_name));
      }
    }
    if (op_infer_opt.has_value()) {
      auto op_infer = op_infer_opt.value().Get();
      if (op_infer != nullptr && depend_list.empty()) {
        depend_list = op_infer->GetValueDependArgIndices();
      }
    }
    return depend_list;
  }

  depend_list = op_def->func_impl_.GetValueDependArgIndices();
  if (!depend_list.empty()) {
    return depend_list;
  }
  // if not defined the GetValueDependArgIndices() func in infer, consider all the no-Tensor
  // input as value depend.
  auto args = op_def->args_;
  for (size_t i = 0; i < args.size(); i++) {
    if (args[i].arg_dtype_ != mindspore::ops::OP_DTYPE::DT_TENSOR) {
      (void)depend_list.insert(i);
    }
  }
  return depend_list;
}

size_t GetInputIndexByName(const std::string &op_name, const std::string &input_name) {
  mindspore::ops::OpDefPtr op_def = mindspore::ops::GetOpDef(op_name);
  if (op_def == nullptr) {
    MS_LOG(INFO) << op_name << " is not defined in opdef.";
    return SIZE_MAX;
  }
  auto ks_iter = op_def->indexes_.find(input_name);
  if (ks_iter != op_def->indexes_.end()) {
    size_t index = ks_iter->second;
    MS_LOG(INFO) << "Find " << input_name << "in " << index << "th input of OP " << op_name;
    return index;
  }
  MS_LOG(INFO) << "Not Find " << input_name << "in OP " << op_name;
  return SIZE_MAX;
}

size_t GetOpInputsNum(const std::string &op_name) {
  mindspore::ops::OpDefPtr op_def = mindspore::ops::GetOpDef(op_name);
  if (op_def == nullptr) {
    MS_LOG(INFO) << op_name << " is not defined in opdef.";
    return SIZE_MAX;
  }
  return op_def->indexes_.size();
}

template <typename T>
std::optional<T> GetScalarValue(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<ValueAny>()) {
    return std::nullopt;
  }

  if (value->isa<KernelTensorValue>()) {
    auto kernel_tensor_value = value->cast<KernelTensorValuePtr>();
    MS_EXCEPTION_IF_NULL(kernel_tensor_value);

    MS_EXCEPTION_IF_CHECK_FAIL((kernel_tensor_value->GetDataSize() == sizeof(T)),
                               "The data size in kernel tensor value which contains a scalar [" +
                                 std::to_string(kernel_tensor_value->GetDataSize()) +
                                 "] is not equal to the data type size [" + std::to_string(sizeof(T)) + "]");

    const T *data_ptr = reinterpret_cast<const T *>(kernel_tensor_value->GetDataPtr());
    MS_EXCEPTION_IF_NULL(data_ptr);
    return *data_ptr;
  }

  return GetValue<T>(value);
}

// Specialization for std::string type.
template <>
std::optional<std::string> GetScalarValue(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<ValueAny>()) {
    return std::nullopt;
  }

  if (value->isa<KernelTensorValue>()) {
    auto kernel_tensor_value = value->cast<KernelTensorValuePtr>();
    MS_EXCEPTION_IF_NULL(kernel_tensor_value);
    const char *data_ptr = reinterpret_cast<const char *>(kernel_tensor_value->GetDataPtr());
    MS_EXCEPTION_IF_NULL(data_ptr);
    size_t str_len = kernel_tensor_value->GetDataSize();

    return std::string(data_ptr, data_ptr + str_len);
  }

  return GetValue<std::string>(value);
}

template std::optional<int64_t> GetScalarValue(const ValuePtr &value);
template std::optional<int32_t> GetScalarValue(const ValuePtr &value);
template std::optional<int16_t> GetScalarValue(const ValuePtr &value);
template std::optional<int8_t> GetScalarValue(const ValuePtr &value);
template std::optional<uint64_t> GetScalarValue(const ValuePtr &value);
template std::optional<uint32_t> GetScalarValue(const ValuePtr &value);
template std::optional<uint16_t> GetScalarValue(const ValuePtr &value);
template std::optional<uint8_t> GetScalarValue(const ValuePtr &value);
template std::optional<double> GetScalarValue(const ValuePtr &value);
template std::optional<float> GetScalarValue(const ValuePtr &value);
template std::optional<bool> GetScalarValue(const ValuePtr &value);
template std::optional<std::string> GetScalarValue(const ValuePtr &value);

// This interface is only used to convert values of type Sequence or Tensor to std::vector.
template <typename T>
std::optional<ArrayValue<T>> GetArrayValue(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<ValueAny>()) {
    return std::nullopt;
  }

  std::vector<T> array_data;
  std::set<size_t> unknown_value_indexes;

  if (value->isa<KernelTensorValue>()) {
    auto kernel_tensor_value = value->cast<KernelTensorValuePtr>();
    MS_EXCEPTION_IF_NULL(kernel_tensor_value);

    size_t element_size = kernel_tensor_value->GetDataSize() / sizeof(T);
    if (element_size != 0) {
      const T *data_ptr = reinterpret_cast<const T *>(kernel_tensor_value->GetDataPtr());
      MS_EXCEPTION_IF_NULL(data_ptr);
      array_data.assign(data_ptr, data_ptr + element_size);
    }
  } else if (value->isa<ValueSequence>()) {
    // Sequence structure: Data is stored discretely.
    auto value_seq = value->cast<ValueSequencePtr>();
    MS_EXCEPTION_IF_NULL(value_seq);

    const auto &element_values = value_seq->value();
    size_t element_size = element_values.size();
    array_data.reserve(element_size);
    for (size_t i = 0; i < element_size; i++) {
      const auto &element = element_values[i];
      MS_EXCEPTION_IF_NULL(element);
      if (element->isa<ValueAny>() || element->isa<None>()) {
        array_data.push_back(static_cast<T>(0));
        unknown_value_indexes.insert(i);
        continue;
      }

      array_data.push_back(GetValue<T>(element));
    }
  } else if (value->isa<tensor::Tensor>()) {
    // Tensor structure: Data is stored continuously.
    auto tensor = value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    size_t element_size = tensor->DataSize();
    T *data = reinterpret_cast<T *>(tensor->data_c());
    array_data.assign(data, data + element_size);
  } else {
    MS_LOG(EXCEPTION) << "Failed to get array value, expect sequence or tensor type, but got: " << value->type_name();
  }

  return std::optional<ArrayValue<T>>(std::in_place, std::move(array_data), std::move(unknown_value_indexes));
}

template std::optional<ArrayValue<int64_t>> GetArrayValue(const ValuePtr &value);
template std::optional<ArrayValue<int32_t>> GetArrayValue(const ValuePtr &value);
template std::optional<ArrayValue<int16_t>> GetArrayValue(const ValuePtr &value);
template std::optional<ArrayValue<int8_t>> GetArrayValue(const ValuePtr &value);
template std::optional<ArrayValue<uint64_t>> GetArrayValue(const ValuePtr &value);
template std::optional<ArrayValue<uint32_t>> GetArrayValue(const ValuePtr &value);
template std::optional<ArrayValue<uint16_t>> GetArrayValue(const ValuePtr &value);
template std::optional<ArrayValue<uint8_t>> GetArrayValue(const ValuePtr &value);
template std::optional<ArrayValue<double>> GetArrayValue(const ValuePtr &value);
template std::optional<ArrayValue<float>> GetArrayValue(const ValuePtr &value);
template std::optional<ArrayValue<bool>> GetArrayValue(const ValuePtr &value);
template std::optional<ArrayValue<std::string>> GetArrayValue(const ValuePtr &value);
}  // namespace ops
}  // namespace mindspore
