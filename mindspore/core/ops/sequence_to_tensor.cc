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
#include "ops/ops_func_impl/tuple_to_tensor.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
class SequenceToTensorInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto seq_input = input_args[0];
    ShapeVector shape;
    if (seq_input->GetShape()->isa<abstract::DynamicSequenceShape>()) {
      // For list/tuple with dynamic len, convert to a dynamic tensor.
      return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::Shape::kShapeDimAny});
    }
    auto seq_shape = seq_input->GetShape()->cast<abstract::SequenceShapePtr>();
    MS_EXCEPTION_IF_NULL(seq_shape);
    return std::make_shared<abstract::Shape>(ShapeVector{SizeToLong(seq_shape->size())});
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    const int64_t input_len = 1;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, input_len,
                                             prim_name);
    auto elem = input_args[0];
    if (!CheckAndConvertUtils::IsSequence(elem)) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name
                              << "', the input should be sequence but got: " << elem->ToString();
    }

    auto attr = primitive->GetAttr("dtype");
    if (attr == nullptr) {
      auto type_abs = abstract::CheckArg<abstract::AbstractType>(prim_name, input_args, 1);
      attr = type_abs->GetValue();
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
    auto elem_value = elem->GetValue();
    if (elem_value->ContainsValueAny()) {
      return nullptr;
    }
    auto type_abs = abstract::CheckArg<abstract::AbstractType>(prim_name, input_args, 1);
    auto dst_type = type_abs->GetValue()->cast<TypePtr>();
    MS_EXCEPTION_IF_NULL(dst_type);
    auto value_tuple = elem_value->cast<ValueSequencePtr>();
    MS_EXCEPTION_IF_NULL(value_tuple);
    return SeqToTensorByType(value_tuple, dst_type);
  }
};
MIND_API_OPERATOR_IMPL(ListToTensor, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ListToTensor, prim::kPrimListToTensor, SequenceToTensorInfer, true);
}  // namespace ops
}  // namespace mindspore
