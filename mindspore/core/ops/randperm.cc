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
#include <climits>
#include <map>
#include <set>
#include "mindapi/ir/type.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/random_ops.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"

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

TypeId Randperm::get_dtype() const {
  auto dtype_ptr = GetAttr("dtype");
  MS_EXCEPTION_IF_NULL(dtype_ptr);
  return dtype_ptr->cast<api::TensorTypePtr>()->element()->type_id();
}

int64_t GetDtypeMaxForCheckOverFlow(const TypePtr tid) {
  MS_EXCEPTION_IF_NULL(tid);
  int64_t max = 0;
  int64_t max_float16 = 65504;
  switch (tid->type_id()) {
    case kNumberTypeUInt8:
      max = UCHAR_MAX;
      break;
    case kNumberTypeInt8:
      max = SCHAR_MAX;
      break;
    case kNumberTypeUInt16:
      max = USHRT_MAX;
      break;
    case kNumberTypeInt16:
      max = SHRT_MAX;
      break;
    case kNumberTypeUInt32:
      max = UINT_MAX;
      break;
    case kNumberTypeInt32:
      max = INT_MAX;
      break;
    case kNumberTypeFloat16:
      max = max_float16;
      break;
    default:
      max = LONG_MAX - 1;
      break;
  }
  return max;
}

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

    auto input_x = input_args[kInputIndex0];
    MS_EXCEPTION_IF_NULL(input_x);
    auto x_value = input_x->BuildValue();
    MS_EXCEPTION_IF_NULL(x_value);
    auto value_ptr = primitive->GetAttr("max_length");
    auto max_length = GetValue<int64_t>(value_ptr);
    if (input_x->isa<abstract::AbstractTensor>()) {
      if (x_value->isa<tensor::Tensor>()) {
        auto x_int = CheckAndConvertUtils::CheckTensorIntValue("x", x_value, primitive->name());
        if (x_int[0] < 0) {
          MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', The value of the input n (" << x_int[0]
                                   << ") cannot be less than 0";
        }
        if (x_int[0] > max_length) {
          MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', input 'n' (" << x_int[0]
                                   << ") cannot exceed 'max_length' (" << max_length << ").";
        }
        auto dtype_ptr = primitive->GetAttr("dtype");
        MS_EXCEPTION_IF_NULL(dtype_ptr);
        auto output_type = dtype_ptr->cast<TypePtr>();
        int64_t max_data = GetDtypeMaxForCheckOverFlow(output_type);
        if (x_int[0] > max_data + 1) {
          MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', input 'n' must be less than "
                                   << "or equal to the largest number in 'dtype', but got: " << x_int[0];
        }
        return std::make_shared<abstract::Shape>(std::vector<int64_t>{GetValue<int64_t>(value_ptr)});
      }
    }
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    auto prim_name = prim->name();
    (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, 1, prim_name);
    MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);

    auto x_type = input_args[kInputIndex0]->BuildType();
    const std::set<TypePtr> valid_types_x = {kInt32, kInt64};
    (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types_x, prim_name);

    auto dtype = GetValue<TypePtr>(prim->GetAttr("dtype"));
    const std::set<TypePtr> valid_types_dtype = {kInt8,   kInt16,  kInt32,   kInt64,   kUInt8,  kUInt16,
                                                 kUInt32, kUInt64, kFloat16, kFloat32, kFloat64};
    auto out_type = CheckAndConvertUtils::CheckTypeValid("dtype", dtype->cast<TypePtr>(), valid_types_dtype, prim_name);
    return out_type;
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {0}; }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(Randperm, prim::kPrimRandperm, RandpermInfer, false);
}  // namespace ops
}  // namespace mindspore
