/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "ops/concat_offset.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "utils/check_convert_utils.h"
#include "ops/array_ops.h"

namespace mindspore {
namespace ops {
int64_t ConcatOffset::get_axis() const {
  auto value_ptr = GetAttr(kAxis);
  return GetValue<int64_t>(value_ptr);
}
void ConcatOffset::set_axis(const int64_t axis) { (void)this->AddAttr(kAxis, api::MakeValue(axis)); }

MIND_API_OPERATOR_IMPL(ConcatOffset, BaseOperator);

class MIND_API ConcatOffsetInfer : public abstract::OpInferBase {
 public:
  // This is used for backend infer by kernel tensor.
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    const std::string op_name = primitive->name();
    AbstractBasePtr tensor_base = nullptr;
    size_t tuple_len = 0;
    MS_EXCEPTION_IF_NULL(input_args[0]);
    // In frontend, input_args[0] may be a AbstractTuple.
    if (input_args[0]->isa<abstract::AbstractTuple>()) {
      CheckArgsSize(op_name, input_args, 1);
      auto arg = abstract::CheckArg<abstract::AbstractTuple>(op_name, input_args, 0);
      tuple_len = arg->elements().size();
      tensor_base = abstract::CheckArg<abstract::AbstractTensor>(op_name, arg->elements(), 0);
    } else if (CheckAndConvertUtils::IsTensor(input_args[0])) {
      tuple_len = input_args.size();
      tensor_base = CheckAndConvertUtils::CheckArgsType(op_name, input_args, 0, kObjectTypeTensorType);
    }

    MS_EXCEPTION_IF_NULL(tensor_base);
    ShapeVector shape_base = tensor_base->GetShape()->GetShapeVector();
    size_t rank = shape_base.size();
    ShapeVector out_shape{SizeToLong(tuple_len), SizeToLong(rank)};
    return std::make_shared<abstract::TensorShape>(out_shape);
  }

  // This is used for backend infer by kernel tensor.
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return kInt64;
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ConcatOffset, prim::kPrimConcatOffset, ConcatOffsetInfer, false);
}  // namespace ops
}  // namespace mindspore
