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

#include "ops/identity_n.h"

#include <memory>
#include <set>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/array_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(IdentityN, BaseOperator);
class MIND_API IdentityNInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    auto shapes = input_args[kInputIndex0]->GetShape();
    return shapes->Clone();
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto op_name = primitive->name();
    constexpr int64_t input_num = 1;
    (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, input_num,
                                             op_name);
    bool is_tuple_x = input_args[kInputIndex0]->GetType()->object_type() == kObjectTypeTuple;
    bool is_list_x = input_args[kInputIndex0]->GetType()->object_type() == kObjectTypeList;
    if ((!is_tuple_x) && (!is_list_x)) {
      MS_EXCEPTION(TypeError) << "For [" << op_name << "] should have ListTensor or TupleTensor input but get "
                              << input_args[kInputIndex0]->GetType()->ToString();
    }

    auto input_type = input_args[kInputIndex0]->GetType();
    TypePtrList types_list;
    size_t in_size;
    if (is_tuple_x) {
      types_list = input_type->cast<TuplePtr>()->elements();
      in_size = types_list.size();
    } else {
      types_list = input_type->cast<ListPtr>()->elements();
      in_size = types_list.size();
    }
    if (in_size < 1) {
      MS_EXCEPTION(ValueError) << "For [" << op_name
                               << "] input list of length should be equal or greater than 1 but get " << in_size
                               << " .";
    }
    const std::set<TypePtr> identityn_valid_types = {kBool,   kInt8,   kInt16,  kInt32,   kInt64,   kUInt8,
                                                     kUInt16, kUInt32, kUInt64, kFloat16, kFloat32, kFloat64};
    for (size_t idx = 0; idx < in_size; ++idx) {
      auto name = "input x[" + std::to_string(idx) + "]";
      (void)CheckAndConvertUtils::CheckTensorTypeValid(name, types_list[idx], identityn_valid_types, op_name);
    }
    return input_type->Clone();
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(IdentityN, prim::kPrimIdentityN, IdentityNInfer, false);
}  // namespace ops
}  // namespace mindspore
