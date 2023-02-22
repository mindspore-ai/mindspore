/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include <set>
#include <map>
#include <memory>

#include "ops/l2_normalize.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
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
MIND_API_OPERATOR_IMPL(L2Normalize, BaseOperator);
void L2Normalize::Init(const std::vector<int64_t> &axis, const float epsilon) {
  this->set_axis(axis);
  this->set_epsilon(epsilon);
}

void L2Normalize::set_axis(const std::vector<int64_t> &axis) { (void)AddAttr(kAxis, api::MakeValue(axis)); }

void L2Normalize::set_epsilon(const float epsilon) { (void)AddAttr(kEpsilon, api::MakeValue(epsilon)); }

std::vector<int64_t> L2Normalize::get_axis() const { return GetValue<std::vector<int64_t>>(GetAttr(kAxis)); }

float L2Normalize::get_epsilon() const {
  auto value_ptr = GetAttr(kEpsilon);
  return GetValue<float>(value_ptr);
}

class L2NormalizeInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    auto input_shape_ptr = input_args[kInputIndex0]->BuildShape();
    if (IsDynamicRank(CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_shape_ptr)[kShape])) {
      return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
    }
    auto input_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_shape_ptr);
    auto input_shape = input_shape_map[kShape];

    const int64_t kL2NormalizeInputsNum = 1;
    const int64_t input_num = kL2NormalizeInputsNum;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num,
                                             prim_name);
    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }

    auto input_rank = SizeToLong(input_shape.size());
    if (input_rank == 0) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', input shape size must be > 0.";
    }
    // failed to get vector<int64_t> axis from infer
    auto axis_vec = CheckAndConvertUtils::CheckIntOrTupleInt("attribute[axis]", primitive->GetAttr("axis"), prim_name);
    int64_t axis = axis_vec[0];
    CheckAndConvertUtils::CheckInRange("axis value", axis, kIncludeLeft, {-input_rank, input_rank}, prim_name);

    auto output_shape = input_shape;
    return std::make_shared<abstract::Shape>(output_shape);
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    auto prim_name = prim->name();
    const int64_t kL2NormalizeInputsNum = 1;
    const int64_t input_num = kL2NormalizeInputsNum;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num,
                                             prim_name);
    const std::set<TypePtr> valid_types = {kFloat32, kFloat16, kFloat64};
    MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
    auto type = input_args[kInputIndex0]->BuildType();
    MS_EXCEPTION_IF_NULL(type);
    if (!type->isa<TensorType>()) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', input must be a Tensor, but got: " << type->ToString()
                              << ".";
    }
    (void)CheckAndConvertUtils::CheckTypeValid("input_x", type, valid_types, prim_name);
    return type;
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(L2Normalize, prim::kPrimL2Normalize, L2NormalizeInfer, false);
}  // namespace ops
}  // namespace mindspore
