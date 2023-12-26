/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "ops/ops_func_impl/tuple_to_tensor.h"

#include <utility>
#include <memory>
#include "ops/ops_frontend_func_impl.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "ops/op_utils.h"
#include "ir/dtype.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ops/op_name.h"
#include "include/backend/op_evaluator.h"
#include "kernel/kernel.h"
#include "utils/ms_context.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/param_validator.h"
#include "base/base.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "ops/base_operator.h"
#include "ops/list_to_tensor.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
constexpr auto kTupleToTensor = "TupleToTensor";
tensor::TensorPtr CreateEmptyTupleTensorByType(const TypePtr &data_type) {
  std::vector<int64_t> tensor_shape = {0};
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(data_type->type_id(), tensor_shape);
  MS_EXCEPTION_IF_NULL(tensor);
  return tensor;
}
template <typename T, typename S>
tensor::TensorPtr CreateTensorByTupleCast(const std::vector<T> &values, const TypePtr &type_ptr,
                                          const size_t data_len) {
  std::vector<S> new_values;
  (void)std::transform(values.begin(), values.end(), std::back_inserter(new_values),
                       [&](T value) -> S { return static_cast<S>(value); });
  std::vector<int64_t> tensor_shape = {SizeToLong(new_values.size())};
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(type_ptr->type_id(), tensor_shape);
  MS_EXCEPTION_IF_NULL(tensor);
  auto data_ptr = tensor->data_c();
  MS_EXCEPTION_IF_NULL(data_ptr);
  auto elem_num = new_values.size() * data_len;
  auto ret_code = memcpy_s(data_ptr, static_cast<size_t>(tensor->data().nbytes()), new_values.data(), elem_num);
  if (ret_code != EOK) {
    MS_LOG(EXCEPTION) << "Failed to copy data into tensor, memcpy_s errorno: " << ret_code;
  }
  return tensor;
}
template <typename T>
tensor::TensorPtr CreateTensorWithValueTuple(const ValueSequencePtr &value_tuple, const TypePtr &type_ptr,
                                             const size_t data_len) {
  MS_EXCEPTION_IF_NULL(value_tuple);
  MS_EXCEPTION_IF_NULL(type_ptr);
  std::vector<T> values;
  auto first_type = value_tuple->value()[0]->type()->type_id();
  for (const auto &v : value_tuple->value()) {
    MS_EXCEPTION_IF_NULL(v);
    if (v->isa<Scalar>()) {
      ScalarPtr scalar = v->cast<ScalarPtr>();
      auto cur_type = scalar->type()->type_id();
      if (cur_type != first_type) {
        MS_EXCEPTION(TypeError) << "the tuple elements type must be same, first element type = " << first_type
                                << " cur_type = " << cur_type;
      }
      values.push_back(GetValue<T>(scalar));
    } else {
      MS_EXCEPTION(TypeError) << "The value " << v << "of tuple is not a scalar";
    }
  }
  if (type_ptr->type_id() == kNumberTypeInt32) {
    return CreateTensorByTupleCast<T, int32_t>(values, type_ptr, data_len);
  } else if (type_ptr->type_id() == kNumberTypeInt64) {
    return CreateTensorByTupleCast<T, int64_t>(values, type_ptr, data_len);
  } else if (type_ptr->type_id() == kNumberTypeFloat32) {
    return CreateTensorByTupleCast<T, float>(values, type_ptr, data_len);
  } else if (type_ptr->type_id() == kNumberTypeFloat64) {
    return CreateTensorByTupleCast<T, double>(values, type_ptr, data_len);
  } else {
    MS_EXCEPTION(TypeError) << "Invalid scalar type: " << type_ptr->ToString();
  }
}

tensor::TensorPtr SeqToTensorByType(const ValueSequencePtr &value_tuple, const TypePtr &data_type) {
  tensor::TensorPtr tensor = nullptr;
  if (value_tuple->value().empty()) {
    tensor = CreateEmptyTupleTensorByType(data_type);
    return tensor;
  }
  ValuePtr v = *(value_tuple->value().begin());
  MS_EXCEPTION_IF_NULL(v);
  // Currently we only deal with the scalar tuple
  if (!v->isa<Scalar>()) {
    MS_EXCEPTION(TypeError) << "The value " << v << "of tuple is not a scalar";
  }
  ScalarPtr scalar = v->cast<ScalarPtr>();
  MS_EXCEPTION_IF_NULL(scalar);
  size_t data_len = GetTypeByte(data_type);
  if (scalar->isa<Int32Imm>()) {
    tensor = CreateTensorWithValueTuple<int32_t>(value_tuple, data_type, data_len);
  } else if (scalar->isa<Int64Imm>()) {
    tensor = CreateTensorWithValueTuple<int64_t>(value_tuple, data_type, data_len);
  } else if (scalar->isa<FP32Imm>()) {
    tensor = CreateTensorWithValueTuple<float>(value_tuple, data_type, data_len);
  } else if (scalar->isa<FP64Imm>()) {
    tensor = CreateTensorWithValueTuple<double>(value_tuple, data_type, data_len);
  } else {
    auto type = scalar->type();
    auto type_str = (type == nullptr) ? "nullptr" : type->ToString();
    MS_EXCEPTION(TypeError) << "Invalid scalar type: " << type_str;
  }
  return tensor;
}

class TupleToTensorFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    constexpr int64_t input_len = 2;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_len,
                                             prim_name);
    auto elem = abstract::CheckArg<abstract::AbstractSequence>(prim_name, input_args, 0);
    auto elem_value = elem->GetValue();
    if (elem_value->ContainsValueAny()) {
      return nullptr;
    }
    auto value_tuple = elem_value->cast<ValueSequencePtr>();
    MS_EXCEPTION_IF_NULL(value_tuple);
    auto dtype_value = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue());
    MS_CHECK_VALUE(dtype_value.has_value(),
                   CheckAndConvertUtils::FormatCommMsg("For primitive[", prim_name,
                                                       "], the `dtype` should has valid value for static type."));
    auto dst_type = TypeIdToType(static_cast<TypeId>(dtype_value.value()));
    MS_EXCEPTION_IF_NULL(dst_type);
    return SeqToTensorByType(value_tuple, dst_type);
  }
};
REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL(kTupleToTensor, TupleToTensorFrontendFuncImpl);
}  // namespace ops
}  // namespace mindspore
