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
#include <set>
#include "ops/ops_frontend_func_impl.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "ops/op_utils.h"
#include "ir/dtype.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ops/op_name.h"
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
BaseShapePtr TupleToTensorFuncImpl::InferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) const {
  auto seq_input = input_args[kInputIndex0];
  if (seq_input->GetShape()->isa<abstract::DynamicSequenceShape>()) {
    // For list/tuple with dynamic len, convert to a dynamic tensor.
    return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::Shape::kShapeDimAny});
  }
  auto seq_shape = seq_input->GetShape()->cast<abstract::SequenceShapePtr>();
  MS_EXCEPTION_IF_NULL(seq_shape);
  return std::make_shared<abstract::Shape>(ShapeVector{SizeToLong(seq_shape->size())});
}
TypePtr TupleToTensorFuncImpl::InferType(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_len = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, input_len,
                                           prim_name);
  auto elem = input_args[0];
  if (!CheckAndConvertUtils::IsSequence(elem)) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name << "', the input should be sequence but got: " << elem->ToString();
  }
  const std::set<TypePtr> valid_types = {kBool,   kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,     kUInt16,
                                         kUInt32, kUInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
  TypePtr dst_type{nullptr};
  if (input_args[kInputIndex1]->GetType()->isa<TypeNone>()) {
    auto attr = primitive->GetAttr("dtype");
    if (attr == nullptr) {
      auto abs = elem->cast<abstract::AbstractTuplePtr>();
      MS_EXCEPTION_IF_NULL(abs);
      attr = abs->elements()[0]->BuildType();
    }
    MS_EXCEPTION_IF_NULL(attr);
    if (!attr->isa<Type>()) {
      MS_EXCEPTION(TypeError)
        << "For '" << prim_name
        << "', the supported data type is ['bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16','uint32', "
           "'uint64','float16', 'float32', 'float64'], but got an invalid dtype!";
    }
    dst_type = attr->isa<TensorType>() ? attr->cast_ptr<TensorType>()->element() : attr->cast<TypePtr>();
  } else {
    auto dtype_value = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue());
    MS_CHECK_VALUE(dtype_value.has_value(),
                   CheckAndConvertUtils::FormatCommMsg("For primitive[", prim_name,
                                                       "], the `dtype` should has valid value for static type."));
    dst_type = TypeIdToType(static_cast<TypeId>(dtype_value.value()));
  }
  MS_EXCEPTION_IF_NULL(dst_type);
  (void)CheckAndConvertUtils::CheckSubClass("dtype", dst_type, valid_types, prim_name);
  return std::make_shared<TensorType>(dst_type);
}
}  // namespace ops
}  // namespace mindspore
