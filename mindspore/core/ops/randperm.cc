/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "ops/randperm.h"
#include <set>
#include <map>
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"
#include "utils/ms_context.h"
#include "ops/op_utils.h"
#include "mindapi/ir/type.h"

namespace mindspore {
namespace ops {
const int64_t kInputShape0Dim = 1;
const int64_t kInputShape0Shape = 1;
void Randperm::set_max_length(const int64_t max_length) {
  (void)AddAttr("max_length", api::MakeValue(CheckAndConvertUtils::CheckInteger("max_length", max_length, kGreaterThan,
                                                                                0, "Randperm")));
}

void Randperm::set_pad(const int64_t pad) {
  (void)AddAttr(kPad, api::MakeValue(CheckAndConvertUtils::CheckInteger(kPad, pad, kGreaterThan, 0, name())));
}
void Randperm::set_dtype(const TypeId dtype) { (void)this->AddAttr("dtype", api::Type::GetType(dtype)); }

int64_t Randperm::get_max_length() const {
  auto value_ptr = GetAttr("max_length");
  return GetValue<int64_t>(value_ptr);
}

int64_t Randperm::get_pad() const {
  auto value_ptr = GetAttr(kPad);
  return GetValue<int64_t>(value_ptr);
}

TypeId Randperm::get_dtype() const { return GetAttr("dtype")->cast<api::TensorTypePtr>()->element()->type_id(); }

MIND_API_OPERATOR_IMPL(Randperm, BaseOperator);
class RandpermInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, 1, prim_name);
    (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);

    auto shape_ptr = input_args[kInputIndex0]->BuildShape();
    auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(shape_ptr);
    auto x_shape = shape_map[kShape];

    if (IsDynamic(x_shape)) {
      return shape_ptr;
    }

    if (x_shape.size() != kInputShape0Dim) {
      MS_EXCEPTION(ValueError) << "For 'Randperm', the rank of input tensor must be equal 1. But got "
                               << x_shape.size();
    }
    if (x_shape.front() != kInputShape0Shape) {
      MS_EXCEPTION(ValueError) << "For 'Randperm', the shape value of input tensor must be equal 1. But got "
                               << x_shape.front();
    }
    auto value_ptr = primitive->GetAttr("max_length");
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{GetValue<int64_t>(value_ptr)});
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    auto prim_name = prim->name();
    (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, 1, prim_name);
    MS_EXCEPTION_IF_NULL(input_args[0]);
    auto x_type = input_args[0]->BuildType();
    std::set<TypePtr> input_valid_types{kInt32};
    (void)CheckAndConvertUtils::CheckTypeValid("input_x", x_type, input_valid_types, prim_name);
    auto dtype_attr = prim->GetAttr("dtype");
    MS_EXCEPTION_IF_NULL(dtype_attr);
    auto infer_type = dtype_attr->cast<TypePtr>();
    MS_EXCEPTION_IF_NULL(infer_type);
    const std::set<TypePtr> valid_types = {kInt8, kInt16, kInt32, kInt64, kUInt8, kUInt16, kUInt32, kUInt64};
    (void)CheckAndConvertUtils::CheckTypeValid("output", infer_type, valid_types, prim_name);
    return infer_type;
  }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(Randperm, prim::kPrimRandperm, RandpermInfer, false);
}  // namespace ops
}  // namespace mindspore
