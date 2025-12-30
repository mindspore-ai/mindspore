/**
 * Copyright 2020-2024 Huawei Technologies Co., Ltd
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

#include "ops_utils/op_utils.h"

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <limits>
#include <functional>
#include <unordered_set>

#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/param_validator.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ir/scalar.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "ir/kernel_tensor_value.h"
#include "mindapi/base/type_id.h"
#include "mindspore/core/include/utils/anf_utils.h"
#include "mindapi/helper.h"
#include "primitive/op_name.h"
#include "include/op_def.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "ir/func_graph.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "utils/ms_context.h"
#include "ops_utils/dl_utils.h"
#include "ops_utils/callback.h"
#include "utils/file_utils.h"
#include "utils/ms_utils.h"

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
                               << tensor::ShapeToString(x_shape) << " and " << y_shape_name << " = "
                               << tensor::ShapeToString(y_shape);
    }
  }
  return broadcast_shape;
}

// CalBroadCastShape for simple infer
std::vector<int64_t> CalBroadCastShapeV2(const std::vector<int64_t> &x_shape, const std::vector<int64_t> &y_shape,
                                         const std::string &op_name, const std::string &op_x_name,
                                         const std::string &op_y_name) {
  if (x_shape == y_shape) {
    return x_shape;
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
    } else if (min_shape[i] != 1 && max_shape[dst_i] != min_shape[i]) {
      auto x_shape_name = op_x_name + ".shape";
      auto y_shape_name = op_y_name + ".shape";
      MS_EXCEPTION(ValueError) << "For '" << op_name << "', " << x_shape_name << " and " << y_shape_name
                               << " need to broadcast. The value of " << x_shape_name << "["
                               << std::to_string(x_length + i) << "] or " << y_shape_name << "["
                               << std::to_string(y_length + i) << "] must be 1 when they are not the same, but got "
                               << x_shape_name << " = " << tensor::ShapeToString(x_shape) << " and " << y_shape_name
                               << " = " << tensor::ShapeToString(y_shape);
    }
  }
  return broadcast_shape;
}

abstract::ShapePtr BroadCastInferShape(const std::string &op_name, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(input_args[kIndex0]);
  MS_EXCEPTION_IF_NULL(input_args[kIndex1]);
  ShapeVector x_shape;
  if (!input_args[0]->GetShape()->isa<abstract::NoShape>()) {
    x_shape = GetShapeFromTensor(input_args[0]);
  }

  ShapeVector y_shape;
  if (!input_args[1]->GetShape()->isa<abstract::NoShape>()) {
    y_shape = GetShapeFromTensor(input_args[1]);
  }

  auto broadcast_shape = CalBroadCastShape(x_shape, y_shape, op_name);
  return std::make_shared<abstract::Shape>(broadcast_shape);
}

ShapeVector BroadCastInferShape(const std::string &op_name, const ValuePtrList &input_values) {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::TensorPtr>();
  const auto &y_tensor = input_values[kIndex1]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  MS_EXCEPTION_IF_NULL(y_tensor);

  auto x_shape = x_tensor->shape();
  auto y_shape = y_tensor->shape();

  auto broadcast_shape = CalBroadCastShape(x_shape, y_shape, op_name);
  return broadcast_shape;
}

bool IsBroadcastable(const std::vector<int64_t> &x_shape, const std::vector<int64_t> &y_shape) {
  if (x_shape == y_shape) {
    return true;
  }

  if (IsDynamicRank(x_shape) || IsDynamicRank(y_shape)) {
    return true;
  }

  if (x_shape.size() < y_shape.size()) {
    return false;
  }

  auto miss = x_shape.size() - y_shape.size();
  for (size_t i = 0; i < y_shape.size(); i++) {
    if (x_shape[miss + i] == y_shape[i]) {
      continue;
    }
    if (x_shape[miss + i] == -1) {
      continue;
    }
    if (y_shape[i] == -1 || y_shape[i] == 1) {
      continue;
    }
    return false;
  }
  return true;
}

BaseShapePtr EltwiseGradInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(input_args[0]);
  MS_EXCEPTION_IF_NULL(input_args[1]);
  auto prim_name = primitive->name();
  auto x = CheckAndConvertUtils::CheckArgsType(prim_name, input_args, 0, kObjectTypeTensorType);
  auto dout = CheckAndConvertUtils::CheckArgsType(prim_name, input_args, 1, kObjectTypeTensorType);
  auto x_shape_ptr = x->GetShape();
  auto dout_shape_ptr = dout->GetShape();
  MS_EXCEPTION_IF_NULL(x_shape_ptr);
  MS_EXCEPTION_IF_NULL(dout_shape_ptr);
  auto x_shape = x_shape_ptr->GetShapeVector();
  auto dout_shape = dout_shape_ptr->GetShapeVector();
  if (IsDynamicRank(x_shape) || IsDynamicRank(dout_shape)) {
    return input_args[1]->GetShape()->Clone();
  } else if (x_shape.size() != dout_shape.size()) {
    MS_EXCEPTION(ValueError) << "Rank of x(" << x_shape.size() << ") and dout(" << dout_shape.size()
                             << ") not equal, primitive name: " << prim_name << ".";
  }

  for (size_t i = 0; i < x_shape.size(); i++) {
    if (x_shape[i] != abstract::Shape::kShapeDimAny && dout_shape[i] != abstract::Shape::kShapeDimAny &&
        x_shape[i] != dout_shape[i]) {
      MS_EXCEPTION(ValueError) << "The " << i << "th dim of x(" << x_shape[i] << ") and dout(" << dout_shape[i]
                               << ") not equal, primitive name: " << prim_name << ".";
    }
  }
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

ShapeArray EltwiseGradSimpleInferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &prim_name = primitive->name();
  const auto &dout_tensor = input_values[kIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(dout_tensor);
  const auto &y_tensor = input_values[kIndex1]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(y_tensor);

  const auto &dout_shape = dout_tensor->shape();
  const auto &y_shape = y_tensor->shape();

  if (dout_shape.size() != y_shape.size()) {
    MS_EXCEPTION(ValueError) << "Rank of x(" << y_shape.size() << ") and dout(" << dout_shape.size()
                             << ") not equal, primitive name: " << prim_name << ".";
  }

  for (size_t i = 0; i < y_shape.size(); i++) {
    if (y_shape[i] != dout_shape[i]) {
      MS_EXCEPTION(ValueError) << "The " << i << "th dim of x(" << y_shape[i] << ") and dout(" << dout_shape[i]
                               << ") not equal, primitive name: " << prim_name << ".";
    }
  }
  return {dout_shape};
}

TypePtrList EltwiseGradSimpleInferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &dout_tensor = input_values[kIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(dout_tensor);
  const auto &y_tensor = input_values[kIndex1]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(y_tensor);

  const auto &dout_type = dout_tensor->Dtype();
  const auto &y_type = y_tensor->Dtype();

  if (dout_type->type_id() != y_type->type_id()) {
    MS_LOG_EXCEPTION << "For " << primitive->name()
                     << ", the grad type must be same as input type, but got grad_type: " << dout_type->ToString()
                     << " and x_type: " << y_type->ToString();
  }
  return {dout_type};
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
    auto value_opt = GetArrayValue<int64_t>(input_args[kInputIndex1]);
    auto value_array = value_opt.value();
    *axis_value = value_array.ToVector();
    return !value_opt.has_value();
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
int64_t MakeWrapDim(int64_t dim, int64_t dim_post_expr) {
  // this will make range [-1, 0]
  if (dim_post_expr <= 0) {
    dim_post_expr = 1;
  }

  if (dim < 0) {
    dim += dim_post_expr;
  }

  return dim;
}

std::bitset<kBitSize> MakeDimMask(std::vector<int64_t> dims, int64_t ndim) {
  std::bitset<kBitSize> mask = std::bitset<kBitSize>();
  if (dims.empty()) {
    mask.flip();
  } else {
    for (int64_t dim : dims) {
      mask.set(MakeWrapDim(dim, ndim));
    }
  }

  return mask;
}

abstract::ShapePtr ReduceExtInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto input_shape_ptr = input_args[0]->GetShape();
  const auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_shape_ptr)[kShape];
  int64_t ndim = static_cast<int64_t>(input_shape.size());
  auto dim = GetValue<std::vector<int64_t>>(input_args[1]->GetValue());
  auto keepdim = GetValue<bool>(input_args[2]->GetValue());
  std::bitset<kBitSize> mask = MakeDimMask(dim, ndim);
  auto shape = input_shape;

  for (int dim_temp = static_cast<int64_t>(shape.size()) - 1; dim_temp >= 0; dim_temp--) {
    if (mask[dim_temp]) {
      if (keepdim) {
        shape[dim_temp] = 1;
      } else {
        shape.erase(shape.begin() + dim_temp);
      }
    }
  }
  return std::make_shared<abstract::Shape>(shape);
}

TypePtr ReduceExtInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto dtype_ptr = input_args[3]->GetValue();
  (void)CheckAndConvertUtils::CheckTypeValid("input", input_args[0]->BuildType(),
                                             common_valid_types_with_complex_and_bool, prim->name());
  auto dtype_type_ptr = dtype_ptr->cast<TypePtr>();
  if (dtype_type_ptr->type_id() == kMetaTypeNone) {
    return input_args[0]->BuildType();
  } else {
    return dtype_ptr->cast<TypePtr>();
  }
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

BaseShapePtr SetPadShape(const ShapeVector &x_shape, const ArrayValue<int64_t> &paddings) {
  const size_t kNum2 = 2;
  auto out_shape = x_shape;
  auto x_rank = x_shape.size();
  for (size_t i = 0; i < paddings.size() / kNum2; i++) {
    auto pad_idx = i * kNum2;
    if (out_shape[x_rank - i - 1] != abstract::Shape::kShapeDimAny && !paddings.IsValueUnknown(pad_idx) &&
        !paddings.IsValueUnknown(pad_idx + kIndex1)) {
      auto paddings_l = paddings[pad_idx];
      auto paddings_r = paddings[pad_idx + kIndex1];
      out_shape[x_rank - i - kIndex1] = out_shape[x_rank - i - kIndex1] + paddings_l + paddings_r;
    } else {
      out_shape[x_rank - i - kIndex1] = abstract::Shape::kShapeDimAny;
    }
  }
  return std::make_shared<abstract::Shape>(out_shape);
}

void BlockInvalid(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                  ShapeVector out_shape) {
  constexpr int64_t kBlockSize16 = 16;
  constexpr int64_t kBlockSize32 = 32;
  constexpr int64_t kKernelSize = 32;
  std::map<TypeId, int64_t> type_to_blocksize_map = {
    {kNumberTypeFloat16, kBlockSize16}, {kNumberTypeInt32, kBlockSize32}, {kNumberTypeFloat32, kBlockSize32}};
  auto x_type = input_args[kInputIndex0]->GetType()->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(x_type);
  auto x_type_id = x_type->element()->type_id();
  auto need_block = type_to_blocksize_map.find(x_type_id) != type_to_blocksize_map.end();
  if (need_block && !IsDynamic(out_shape)) {
    auto output_size =
      std::accumulate(out_shape.cbegin(), out_shape.cend(), static_cast<int64_t>(1), std::multiplies<int64_t>());
    auto valid_min_size = type_to_blocksize_map[x_type_id] * kKernelSize;
    if (output_size < valid_min_size) {
      MS_EXCEPTION(ValueError)
        << "For '" << primitive->name()
        << "', num of input element is too small, which may cause computation error, try another dtype other than "
           "{float16, float32, int16} or run on Atlas A2 training series to bypass this issue.";
    }
  }
}

BaseShapePtr PadInferShapeBase(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                               const size_t pad_dim) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto x_base_shape = input_args[kInputIndex0]->GetShape();
  auto x_shape = x_base_shape->GetShapeVector();
  // input x dynamic rank
  MS_EXCEPTION_IF_NULL(x_base_shape);
  if (x_base_shape->IsDimUnknown()) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  }
  // input x dynamic shape
  auto x_rank = x_shape.size();
  constexpr size_t minValidDim = 1;
  constexpr size_t maxValidDim = 2;
  if (x_rank != pad_dim + minValidDim && x_rank != pad_dim + maxValidDim) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', input should be " << pad_dim + minValidDim
                             << "D or " << pad_dim + maxValidDim << "D, but got " << x_rank;
  }
  // padding
  auto paddings_opt = GetArrayValue<int64_t>(input_args[kInputIndex1]);
  if (!paddings_opt.has_value()) {
    ShapeVector out_shape = x_shape;
    for (size_t dim = 1; dim <= pad_dim; ++dim) {
      out_shape[x_rank - dim] = abstract::Shape::kShapeDimAny;
    }
    return std::make_shared<abstract::Shape>(std::move(out_shape));
  }
  constexpr size_t kScaleNum = 2;
  auto paddings = paddings_opt.value();
  if (paddings.size() != pad_dim * kScaleNum) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the padding length should be "
                             << pad_dim * kScaleNum << ", but got " << paddings.size();
  }

  auto out_shape = SetPadShape(x_shape, paddings);
  return out_shape;
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
template OPS_API std::shared_ptr<abstract::AbstractCSRTensor> InferSparseAttr(const PrimitivePtr &primitive,
                                                                              const AbstractBasePtrList &args_abs_list);
template OPS_API std::shared_ptr<abstract::AbstractCOOTensor> InferSparseAttr(const PrimitivePtr &primitive,
                                                                              const AbstractBasePtrList &args_abs_list);

template <typename T>
AbstractBasePtr TensorToSequenceInfer(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr size_t input_0_index = 0;

  auto x_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  if (x_shape.size() > 1) {
    MS_EXCEPTION(ValueError) << "For Primitive[" << prim_name << "], the input must be a 1-D Tensor, but got Tensor "
                             << "with shape: " << x_shape << ".";
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
  if (x_shape.empty()) {
    abs_list.push_back(std::make_shared<abstract::AbstractScalar>(kValueAny, element_type));
  } else {
    for (int64_t i = 0; i < x_shape[0]; i++) {
      abs_list.push_back(std::make_shared<abstract::AbstractScalar>(kValueAny, element_type));
    }
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

template OPS_API AbstractBasePtr InferSequenceSetItem<abstract::AbstractList>(const PrimitivePtr &primitive,
                                                                              const AbstractBasePtrList &args_abs_list);
template OPS_API AbstractBasePtr
InferSequenceSetItem<abstract::AbstractTuple>(const PrimitivePtr &primitive, const AbstractBasePtrList &args_abs_list);

template OPS_API AbstractBasePtr TensorToSequenceInfer<abstract::AbstractList>(
  const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args);

template OPS_API AbstractBasePtr TensorToSequenceInfer<abstract::AbstractTuple>(
  const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args);

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

void CheckTensorScalarRank(const PrimitivePtr &primitive, const AbstractBasePtr input_arg,
                           const std::string &arg_name) {
  MS_EXCEPTION_IF_NULL(input_arg);
  auto shape_ptr = input_arg->GetShape();
  MS_EXCEPTION_IF_NULL(shape_ptr);
  const auto &input_shape = shape_ptr->GetShapeVector();
  const int64_t kDimZero = 0;
  if (MS_LIKELY(!IsDynamic(input_shape))) {
    MS_CHECK_VALUE(input_shape.size() == LongToSize(kDimZero),
                   CheckAndConvertUtils::FormatCheckIntegerMsg("rank of " + arg_name, SizeToLong(input_shape.size()),
                                                               kEqual, kDimZero, primitive));
  }
}

bool IsFloatType(TypePtr type) {
  mindspore::HashSet<TypeId> float_type_ids = {kNumberTypeFloat,   kNumberTypeFloat16,  kNumberTypeFloat32,
                                               kNumberTypeFloat64, kNumberTypeBFloat16, kNumberTypeDouble};
  TypeId type_id;
  if (type->isa<TensorType>()) {
    auto tensor_type = type->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor_type);
    auto element = tensor_type->element();
    MS_EXCEPTION_IF_NULL(element);
    type_id = element->type_id();
  } else {
    type_id = type->type_id();
  }
  if (float_type_ids.find(type_id) != float_type_ids.end()) {
    return true;
  }
  return false;
}

bool IsIntegralType(TypePtr type, bool include_bool) {
  mindspore::HashSet<TypeId> int_type_ids = {kNumberTypeInt,    kNumberTypeInt8,   kNumberTypeInt16, kNumberTypeInt32,
                                             kNumberTypeInt64,  kNumberTypeUInt,   kNumberTypeUInt8, kNumberTypeUInt16,
                                             kNumberTypeUInt32, kNumberTypeUInt64, kNumberTypeInt4,  kNumberTypeGLUInt};

  TypeId type_id;
  if (type->isa<TensorType>()) {
    auto tensor_type = type->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor_type);
    auto element = tensor_type->element();
    MS_EXCEPTION_IF_NULL(element);
    type_id = element->type_id();
  } else {
    type_id = type->type_id();
  }
  if (int_type_ids.find(type_id) != int_type_ids.end()) {
    return true;
  }
  if (include_bool && type_id == kNumberTypeBool) {
    return true;
  }
  return false;
}

// CalBroadCastShape for pyboost and kernel mod
std::vector<int64_t> CalBroadCastShapeV3(const std::vector<int64_t> &x_shape, const std::vector<int64_t> &y_shape) {
  if (x_shape == y_shape) {
    return x_shape;
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
    }
  }
  return broadcast_shape;
}

int ConvertReductionForAclnn(Reduction reduction) {
  std::unordered_map<Reduction, int64_t> reduction_map = {
    {Reduction::REDUCTION_SUM, 2}, {Reduction::MEAN, 1}, {Reduction::NONE, 0}};
  auto iter = reduction_map.find(reduction);
  if (iter == reduction_map.end()) {
    MS_LOG(EXCEPTION) << "For ConvertReductionForAclnn, the value of reduction is invalid.";
  }
  return iter->second;
}

const char *ConvertReductionStrForAclnn(int64_t reduction) {
  static const std::unordered_map<int64_t, const char *> reduction_map = {{0, "sum"}, {1, "mean"}, {2, "none"}};

  auto iter = reduction_map.find(reduction);
  if (iter == reduction_map.end()) {
    throw std::runtime_error("For ConvertReductionForAclnn, the value of reduction must be 'sum', 'mean' or 'none'.");
  }
  return iter->second;
}

size_t CalOutputSize(const std::vector<int64_t> &output_shape, const size_t &type_size) {
  size_t output_size = 1;
  for (const int64_t &size_value : output_shape) {
    // Casting each int64_t value to size_t during multiplication
    output_size *= static_cast<size_t>(size_value);
  }
  output_size *= type_size;
  return output_size;
}

double GetDoubleValueFromScalar(const FP32ImmPtr &scalar) {
  MS_EXCEPTION_IF_NULL(scalar);
  constexpr double eps = 1e-6;
  auto float_value = scalar->value();
  auto doubel_value = scalar->prim_value();
  // If double value is default value 0, don't use double value.
  if (std::abs(doubel_value) > std::numeric_limits<double>::epsilon() && std::abs(float_value - doubel_value) < eps) {
    MS_LOG(DEBUG) << "Use the real double float value in FP32Imm, which is inherited from python float object.";
    return doubel_value;
  }
  return static_cast<double>(float_value);
}

ScalarPtr FetchRealScalar(const ScalarPtr &scalar) {
  MS_EXCEPTION_IF_NULL(scalar);
  auto real_scalar = scalar;
  if (scalar->isa<FP32Imm>()) {
    auto fp32imm_ptr = scalar->cast<FP32ImmPtr>();
    real_scalar = std::make_shared<FP64Imm>(GetDoubleValueFromScalar(fp32imm_ptr));
  }
  return real_scalar;
}

ValueTuplePtr ConvertShapeVectorToValueTuple(const ShapeVector &shape_vector) {
  std::vector<ValuePtr> shape_out_vector;
  std::transform(shape_vector.begin(), shape_vector.end(), std::back_inserter(shape_out_vector),
                 [](int64_t x) { return MakeValue(x); });
  return std::make_shared<ValueTuple>(std::move(shape_out_vector));
}

static const std::map<size_t, TypeId> scalar_tensor_convert_map = {
  // Scalar is bool.
  {GetHashId(kNumberTypeBool, kNumberTypeBool), kNumberTypeBool},
  {GetHashId(kNumberTypeBool, kNumberTypeInt8), kNumberTypeInt8},
  {GetHashId(kNumberTypeBool, kNumberTypeInt16), kNumberTypeInt16},
  {GetHashId(kNumberTypeBool, kNumberTypeInt32), kNumberTypeInt32},
  {GetHashId(kNumberTypeBool, kNumberTypeInt64), kNumberTypeInt64},
  {GetHashId(kNumberTypeBool, kNumberTypeUInt8), kNumberTypeUInt8},
  {GetHashId(kNumberTypeBool, kNumberTypeUInt16), kNumberTypeUInt16},
  {GetHashId(kNumberTypeBool, kNumberTypeUInt32), kNumberTypeUInt32},
  {GetHashId(kNumberTypeBool, kNumberTypeUInt64), kNumberTypeUInt64},
  {GetHashId(kNumberTypeBool, kNumberTypeFloat16), kNumberTypeFloat16},
  {GetHashId(kNumberTypeBool, kNumberTypeBFloat16), kNumberTypeBFloat16},
  {GetHashId(kNumberTypeBool, kNumberTypeFloat32), kNumberTypeFloat32},
  {GetHashId(kNumberTypeBool, kNumberTypeFloat64), kNumberTypeFloat64},
  {GetHashId(kNumberTypeBool, kNumberTypeComplex64), kNumberTypeComplex64},
  {GetHashId(kNumberTypeBool, kNumberTypeComplex128), kNumberTypeComplex128},
  // Scalar is int.
  {GetHashId(kNumberTypeInt64, kNumberTypeBool), kNumberTypeInt64},
  {GetHashId(kNumberTypeInt64, kNumberTypeInt8), kNumberTypeInt8},
  {GetHashId(kNumberTypeInt64, kNumberTypeInt16), kNumberTypeInt16},
  {GetHashId(kNumberTypeInt64, kNumberTypeInt32), kNumberTypeInt32},
  {GetHashId(kNumberTypeInt64, kNumberTypeInt64), kNumberTypeInt64},
  {GetHashId(kNumberTypeInt64, kNumberTypeUInt8), kNumberTypeUInt8},
  {GetHashId(kNumberTypeInt64, kNumberTypeFloat16), kNumberTypeFloat16},
  {GetHashId(kNumberTypeInt64, kNumberTypeBFloat16), kNumberTypeBFloat16},
  {GetHashId(kNumberTypeInt64, kNumberTypeFloat32), kNumberTypeFloat32},
  {GetHashId(kNumberTypeInt64, kNumberTypeFloat64), kNumberTypeFloat64},
  {GetHashId(kNumberTypeInt64, kNumberTypeComplex64), kNumberTypeComplex64},
  {GetHashId(kNumberTypeInt64, kNumberTypeComplex128), kNumberTypeComplex128},
  // Scalar is float.
  {GetHashId(kNumberTypeFloat32, kNumberTypeBool), kNumberTypeFloat32},
  {GetHashId(kNumberTypeFloat32, kNumberTypeInt8), kNumberTypeFloat32},
  {GetHashId(kNumberTypeFloat32, kNumberTypeInt16), kNumberTypeFloat32},
  {GetHashId(kNumberTypeFloat32, kNumberTypeInt32), kNumberTypeFloat32},
  {GetHashId(kNumberTypeFloat32, kNumberTypeInt64), kNumberTypeFloat32},
  {GetHashId(kNumberTypeFloat32, kNumberTypeUInt8), kNumberTypeFloat32},
  {GetHashId(kNumberTypeFloat32, kNumberTypeFloat16), kNumberTypeFloat16},
  {GetHashId(kNumberTypeFloat32, kNumberTypeBFloat16), kNumberTypeBFloat16},
  {GetHashId(kNumberTypeFloat32, kNumberTypeFloat32), kNumberTypeFloat32},
  {GetHashId(kNumberTypeFloat32, kNumberTypeFloat64), kNumberTypeFloat64},
  {GetHashId(kNumberTypeFloat32, kNumberTypeComplex64), kNumberTypeComplex64},
  {GetHashId(kNumberTypeFloat32, kNumberTypeComplex128), kNumberTypeComplex128},
};
TypeId ConvertTypeBetweenTensorAndScalar(const TypeId &tensor_type_id, const TypeId &scalar_type_id,
                                         const size_t hash_id) {
  auto iter = scalar_tensor_convert_map.find(hash_id);
  if (iter != scalar_tensor_convert_map.end()) {
    return iter->second;
  }
  MS_LOG(INFO) << "Type implicit conversion between Tensor[" << TypeIdToString(tensor_type_id) << "] and "
               << TypeIdToString(scalar_type_id) << " is not supported.";
  return kTypeUnknown;
}

size_t GetHashId(int a, int b) { return a < b ? hash_combine(a, b) : hash_combine(b, a); }

bool IsEnableHostNode(const AnfNodePtr &node) {
  OpDefPtr op_def = GetOpDef(AnfUtils::GetCNodeName(node));
  bool is_view = op_def != nullptr ? op_def->is_graph_view_ : false;
  return is_view;
}

constexpr char kOpPluginLoadPathEnv[] = "MS_OP_PLUGIN_PATH";

void *GetOpPluginHandle() {
  static std::once_flag init_flag;
  static void *handle = nullptr;
  std::call_once(init_flag, []() {
    const char *op_plugin_path = common::EnvHelper::GetInstance()->GetEnv(kOpPluginLoadPathEnv);

    if (op_plugin_path == nullptr) {
      MS_LOG(INFO) << kOpPluginLoadPathEnv << " is not set. Op plugin will not be loaded.";
      return;
    }

    auto real_path = FileUtils::GetRealPath(op_plugin_path).value_or("");
    if (real_path.empty()) {
      MS_LOG(ERROR) << "Failed to resolve the real path for " << kOpPluginLoadPathEnv << ": " << op_plugin_path;
      return;
    }
    (void)DL_ERROR();
#ifdef _WIN32
    SetLastError(0);
#endif
    handle = DL_OPEN(real_path);
    if (handle == nullptr) {
      MS_LOG(ERROR) << "Failed to open op plugin file: " << real_path << " Error code: " << DL_ERROR();
      return;
    }
    MS_LOG(INFO) << "Op plugin loaded successfully from path: " << real_path;

    GET_OPS_CALLBACK(SetupOpPluginConfig, void, void *)(handle);
  });
  return handle;
}

bool IsOpPluginKernel(const std::string &op_name) {
  static const std::unordered_set<std::string> *op_names_cache = nullptr;
  static std::once_flag cache_flag;
  std::call_once(cache_flag, []() { op_names_cache = &GetAllOpPluginKernelNames(); });
  // Early return if no plugins are loaded (empty set)
  if (op_names_cache->empty()) {
    return false;
  }
  return op_names_cache->find(op_name) != op_names_cache->end();
}

const std::unordered_set<std::string> &GetAllOpPluginKernelNames() {
  static std::once_flag init_flag;
  static int (*get_op_count)() = nullptr;
  static char **(*get_all_ops)() = nullptr;
  static std::unordered_set<std::string> op_names;
  std::call_once(init_flag, []() {
    void *handle = GetOpPluginHandle();
    if (handle == nullptr) {
      return;
    }
    constexpr auto get_op_count_func_name = "GetRegisteredOpCount";
    constexpr auto get_all_ops_func_name = "GetAllRegisteredOps";
    get_op_count = reinterpret_cast<std::add_pointer<int()>::type>(DL_SYM(handle, get_op_count_func_name));
    get_all_ops = reinterpret_cast<std::add_pointer<char **()>::type>(DL_SYM(handle, get_all_ops_func_name));
    if (get_op_count == nullptr || get_all_ops == nullptr) {
      MS_LOG(ERROR) << "Error occurs when fetching function '" << get_op_count_func_name << "' or '"
                    << get_all_ops_func_name << "' from op plugin library. Error code: " << DL_ERROR();
      return;
    }
    int op_count = get_op_count();
    char **op_names_ptr = get_all_ops();
    for (int i = 0; i < op_count; ++i) {
      op_names.insert(std::string(op_names_ptr[i]));
    }
  });
  return op_names;
}
}  // namespace ops
}  // namespace mindspore
