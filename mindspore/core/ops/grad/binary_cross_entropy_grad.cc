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
#include <string>
#include <memory>
#include <vector>

#include "ops/grad/binary_cross_entropy_grad.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(BinaryCrossEntropyGrad, BaseOperator);

void BinaryCrossEntropyGrad::Init(const Reduction &reduction) { set_reduction(reduction); }

void BinaryCrossEntropyGrad::set_reduction(const Reduction &reduction) {
  std::string reduce;
  if (reduction == Reduction::REDUCTION_SUM) {
    reduce = "sum";
  } else if (reduction == Reduction::MEAN) {
    reduce = "mean";
  } else {
    reduce = "none";
  }
  (void)this->AddAttr(kReduction, api::MakeValue(reduce));
}

Reduction BinaryCrossEntropyGrad::get_reduction() const {
  auto reduction_ptr = GetAttr(kReduction);
  MS_EXCEPTION_IF_NULL(reduction_ptr);
  MS_EXCEPTION_IF_CHECK_FAIL(reduction_ptr->isa<api::StringImm>() || reduction_ptr->isa<api::Int64Imm>(),
                             "invalid value type");
  if (reduction_ptr->isa<api::StringImm>()) {
    auto value_ptr = MakeValue(GetValue<std::string>(reduction_ptr));
    int64_t reduction = 0;
    CheckAndConvertUtils::GetReductionEnumValue(value_ptr, &reduction);
    return Reduction(reduction);
  }
  return Reduction(GetValue<int64_t>(reduction_ptr));
}

class BinaryCrossEntropyGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    const int64_t kInputNum = 4;
    auto prim_name = primitive->name();
    CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, prim_name);
    auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape());
    auto input_shape = shape_map[kShape];
    return std::make_shared<abstract::Shape>(input_shape);
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    const int64_t kInputNum = 4;
    auto prim_name = prim->name();
    CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, prim_name);
    std::set<TypePtr> valid_types = {kFloat16, kFloat32};
    std::map<std::string, TypePtr> types;
    (void)types.emplace("logits", input_args[kInputIndex0]->BuildType());
    (void)types.emplace("labels", input_args[kInputIndex1]->BuildType());
    (void)types.emplace("dout", input_args[kInputIndex2]->BuildType());
    (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim_name);
    auto weight_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
    if (weight_shape.size() > 0) {
      (void)types.emplace("weight", input_args[kInputIndex3]->BuildType());
    }
    (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim_name);
    return input_args[kInputIndex0]->BuildType();
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(BinaryCrossEntropyGrad, prim::kPrimBinaryCrossEntropyGrad, BinaryCrossEntropyGradInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
