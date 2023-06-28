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
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/param_validator.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "ops/base_operator.h"
#include "ops/list_to_tensor.h"
#include "ops/primitive_c.h"
#include "ops/tuple_to_tensor.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
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
class SequenceToTensorInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    auto queue = abstract::CheckArg<abstract::AbstractSequence>(prim_name, input_args, 0);
    ShapeVector shape;

    // For list/tuple with dynamic len, convert to a dynamic tensor.
    if (queue->dynamic_len()) {
      shape.push_back(abstract::Shape::kShapeDimAny);
    } else {
      shape.push_back(queue->elements().size());
    }

    return std::make_shared<abstract::Shape>(shape);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    const int64_t input_len = 1;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, input_len,
                                             prim_name);
    auto elem = input_args[0];
    if (!elem->isa<abstract::AbstractSequence>()) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name
                              << "', the input should be sequence but got: " << elem->ToString();
    }

    auto attr = primitive->GetAttr("dtype");
    if (attr == nullptr) {
      auto type_abs = abstract::CheckArg<abstract::AbstractType>(prim_name, input_args, 1);
      attr = type_abs->BuildValue();
      MS_EXCEPTION_IF_NULL(attr);
    }
    if (!attr->isa<Type>()) {
      MS_EXCEPTION(TypeError)
        << "For '" << prim_name
        << "', the supported data type is ['bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16','uint32', "
           "'uint64','float16', 'float32', 'float64'], but got an invalid dtype!";
    }
    TypePtr dst_type{nullptr};
    if (attr->isa<TensorType>()) {
      dst_type = attr->cast_ptr<TensorType>()->element();
    } else {
      dst_type = attr->cast<TypePtr>();
    }
    const std::set<TypePtr> valid_types = {kBool,   kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,     kUInt16,
                                           kUInt32, kUInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
    (void)CheckAndConvertUtils::CheckSubClass("dtype", dst_type, valid_types, prim_name);
    return dst_type;
  }

  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    constexpr int64_t input_len = 2;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_len,
                                             prim_name);
    auto elem = abstract::CheckArg<abstract::AbstractSequence>(prim_name, input_args, 0);
    auto elem_value = elem->BuildValue();
    if (elem_value == kValueAny) {
      return nullptr;
    }
    auto type_abs = abstract::CheckArg<abstract::AbstractType>(prim_name, input_args, 1);
    auto dst_type = type_abs->BuildValue()->cast<TypePtr>();
    MS_EXCEPTION_IF_NULL(dst_type);
    auto value_tuple = elem_value->cast<ValueSequencePtr>();
    MS_EXCEPTION_IF_NULL(value_tuple);
    return SeqToTensorByType(value_tuple, dst_type);
  }
};
MIND_API_OPERATOR_IMPL(ListToTensor, BaseOperator);
MIND_API_OPERATOR_IMPL(TupleToTensor, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ListToTensor, prim::kPrimListToTensor, SequenceToTensorInfer, true);
REGISTER_PRIMITIVE_OP_INFER_IMPL(TupleToTensor, prim::kPrimTupleToTensor, SequenceToTensorInfer, true);
}  // namespace ops
}  // namespace mindspore
