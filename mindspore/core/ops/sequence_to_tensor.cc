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
#include <vector>
#include <string>

#include "ops/list_to_tensor.h"
#include "ops/tuple_to_tensor.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "include/common/utils/utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
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
      auto abs = queue->dynamic_len_element_abs();
      MS_EXCEPTION_IF_NULL(abs);
      if (abs == nullptr) {
        MS_EXCEPTION(ValueError)
          << "For prim '" << prim_name
          << " dynamic_len_element_abs can't be null, when the input is dynamic length sequence.";
      }
      shape.push_back(abstract::Shape::kShapeDimAny);
    } else {
      shape.push_back(queue->elements().size());
    }

    return std::make_shared<abstract::Shape>(shape);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    constexpr size_t input_len = 1;
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
    auto output_dtype = attr->cast<TypePtr>();

    const std::set<TypePtr> valid_types = {kBool,   kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,     kUInt16,
                                           kUInt32, kUInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
    return CheckAndConvertUtils::CheckSubClass("dtype", output_dtype, valid_types, prim_name);
  }
};
MIND_API_OPERATOR_IMPL(ListToTensor, BaseOperator);
MIND_API_OPERATOR_IMPL(TupleToTensor, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ListToTensor, prim::kPrimListToTensor, SequenceToTensorInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(TupleToTensor, prim::kPrimTupleToTensor, SequenceToTensorInfer, false);
}  // namespace ops
}  // namespace mindspore
