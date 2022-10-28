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
      broadcast_shape.push_back(y_shape[LongToSize(y_length + i)]);
    } else if (y_shape[LongToSize(y_length + i)] == 1) {
      (void)broadcast_shape.push_back(x_shape[LongToSize(x_length + i)]);
    } else if (x_shape[x_length + i] == y_shape[LongToSize(y_length + i)]) {
      (void)broadcast_shape.push_back(x_shape[LongToSize(x_length + i)]);
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
        MS_EXCEPTION(ValueError) << "For '" << prim->name() << "', 'axis' must be 0. But got 'axis' = " << axis->at(i)
                                 << ".";
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
  if (!keep_dims) {
    if (SizeToLong(x_shape.size()) < axis_shape) {
      return out_shape;
    }
    (void)out_shape.insert(out_shape.end(), x_shape.size() - axis_shape, -1LL);
  } else {
    (void)out_shape.insert(out_shape.end(), x_shape.size(), -1LL);
  }
  return out_shape;
}

bool CheckAndGetAxisValue(const std::vector<abstract::AbstractBasePtr> &input_args, std::vector<int64_t> *axis_value,
                          int64_t *axis_shape_v, const PrimitivePtr &primitive) {
  MS_EXCEPTION_IF_NULL(axis_value);
  MS_EXCEPTION_IF_NULL(axis_shape_v);
  bool is_dynamic = false;
  const std::string &op_name = primitive->name();
  if (input_args.size() == 1) {
    auto axis_ptr = primitive->GetAttr("axis");
    if (axis_ptr == nullptr) {
      return is_dynamic;
    }
    if (axis_ptr->isa<tensor::Tensor>()) {
      *axis_value = CheckAndConvertUtils::CheckTensorIntValue("axis", axis_ptr, op_name);
    } else {
      *axis_value = CheckAndConvertUtils::CheckIntOrTupleInt("axis", axis_ptr, op_name);
    }
    return is_dynamic;
  }
  auto input_value = input_args[kInputIndex1]->BuildValue();
  if (input_args[kInputIndex1]->isa<abstract::AbstractScalar>() ||
      input_args[kInputIndex1]->isa<abstract::AbstractTuple>() ||
      input_args[kInputIndex1]->isa<abstract::AbstractList>()) {
    *axis_value = CheckAndConvertUtils::CheckIntOrTupleInt("axis", input_value, op_name);
    if (axis_value->empty()) {
      *axis_shape_v = 0;
    }
  } else if (input_args[kInputIndex1]->isa<abstract::AbstractTensor>()) {
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
    shape_value = MakeFakeShapeBack(shape_value, arg_shapes[0]);
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
  } else if (arg->isa<abstract::AbstractTuple>()) {
    auto elements = arg->cast<abstract::AbstractTuplePtr>()->elements();
    MS_EXCEPTION_IF_CHECK_FAIL(!elements.empty() && !elements[0]->isa<abstract::AbstractTensor>(),
                               "Input cannot be a tuple of tensor");
    auto out_shape = CheckAndConvertUtils::CheckTupleInt("input[shape]", abs_value, primitive->name());
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
}  // namespace ops
}  // namespace mindspore
