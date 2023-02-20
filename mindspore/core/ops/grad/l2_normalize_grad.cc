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

#include "mindspore/core/ops/grad/l2_normalize_grad.h"

#include <vector>
#include <memory>

#include "abstract/param_validator.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
int64_t L2NormalizeGrad::get_axis() const { return GetValue<int64_t>(GetAttr(kAxis)); }

float L2NormalizeGrad::get_epsilon() const {
  auto value_ptr = this->GetAttr(kEpsilon);
  return GetValue<float>(value_ptr);
}

MIND_API_OPERATOR_IMPL(L2NormalizeGrad, BaseOperator);
class L2NormalizeGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    const int64_t kPReLUGradInputsNum = 3;
    const int64_t input_num = kPReLUGradInputsNum;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num,
                                             prim_name);
    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }
    auto input_x_shape_ptr = input_args[kInputIndex0]->BuildShape();
    MS_EXCEPTION_IF_NULL(input_x_shape_ptr);
    auto out_shape_ptr = input_args[kInputIndex1]->BuildShape();
    MS_EXCEPTION_IF_NULL(out_shape_ptr);
    auto dout_shape_ptr = input_args[kInputIndex2]->BuildShape();
    MS_EXCEPTION_IF_NULL(dout_shape_ptr);

    auto input_x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_x_shape_ptr)[kShape];
    auto out_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(out_shape_ptr)[kShape];
    auto dout_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(dout_shape_ptr)[kShape];
    if (IsDynamicRank(input_x_shape) || IsDynamicRank(out_shape) || IsDynamicRank(dout_shape)) {
      return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
    }

    auto input_x = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex0);
    auto out = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex1);
    auto dout = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex2);
    abstract::CheckShapeSame(prim_name, input_x, out);
    abstract::CheckShapeSame(prim_name, input_x, dout);

    return input_x_shape_ptr;
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    auto prim_name = prim->name();
    const int64_t kPReLUGradInputsNum = 3;
    const int64_t input_num = kPReLUGradInputsNum;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num,
                                             prim_name);
    MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
    auto input_x = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex0);
    auto out = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex1);
    auto dout = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex2);
    (void)abstract::CheckDtypeSame(prim_name, input_x, out);
    (void)abstract::CheckDtypeSame(prim_name, input_x, dout);
    auto input_x_type = input_args[kInputIndex0]->BuildType();
    MS_EXCEPTION_IF_NULL(input_x_type);

    return input_x_type;
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(L2NormalizeGrad, prim::kPrimL2NormalizeGrad, L2NormalizeGradInfer, false);
}  // namespace ops
}  // namespace mindspore
