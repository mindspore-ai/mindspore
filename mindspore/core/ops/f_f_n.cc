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
#include "ops/f_f_n.h"
#include <string>
#include <map>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/other_ops.h"

namespace mindspore {
namespace ops {
namespace {
// FFN inputs
// x:       (bs * seq, h)
// weight1: (expert_dim, h, ffn_h)
// weight2: (expert_dim, ffn_h, h)
// expert:  (16)
// bias1:   (expert_dim, ffn_h)
// bias1:   (expert_dim, h)
// ------------------------------
// output:  (bs * seq, h)

enum FFNInputIndex : size_t {
  kInputIndexX = 0,
  kInputIndexW1,
  kInputIndexW2,
  kMinInputNumber,  // 3
  kInputIndexExpert = kMinInputNumber,
  kInputIndexBias1,  // 4
  kInputIndexBias2,
  kInputScale,
  kInputOffset,
  kInputDeqScale1,
  kInputDeqScale2,
  kInputAntiquantScale1,
  kInputAntiquantScale2,
  kInputAntiquantOffset1,
  kInputAntiquantOffset2,
  kMaxInputNumber,  // 10
};

constexpr int64_t kWeightShapeRank = 2;
constexpr int64_t kWeightShapeRankMoe = 3;
constexpr int64_t kBiasShapeRank = 1;
constexpr int64_t kBiasShapeRankMoe = 2;
constexpr int64_t kExpertShapeRank = 1;

void CheckInputsNum(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kGreaterEqual,
                                           SizeToLong(kMinInputNumber), primitive->name());
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kLessEqual,
                                           SizeToLong(kMaxInputNumber), primitive->name());
}
}  // namespace

MIND_API_OPERATOR_IMPL(FFN, BaseOperator);

void FFN::Init(const std::string &activation, int64_t inner_precise) {
  this->set_activation(activation);
  this->set_inner_precise(inner_precise);
}

void FFN::set_activation(const std::string &activation) {
  (void)this->AddAttr(kActivation, api::MakeValue(activation));
}

void FFN::set_inner_precise(int64_t inner_precise) {
  (void)this->AddAttr(kInnerPrecise, api::MakeValue(inner_precise));
}

std::string FFN::get_activation() const {
  auto value_ptr = this->GetAttr(kActivation);
  return GetValue<std::string>(value_ptr);
}

int64_t FFN::get_inner_precise() const {
  auto value_ptr = this->GetAttr(kInnerPrecise);
  return GetValue<int64_t>(value_ptr);
}

class FFNInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    CheckInputsNum(primitive, input_args);
    auto x_shape = input_args[kInputIndexX]->BuildShape();
    return x_shape;
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    CheckInputsNum(primitive, input_args);
    MS_EXCEPTION_IF_NULL(input_args[kInputIndexX]);
    auto x_type = input_args[kInputIndexX]->BuildType();
    const std::set<TypePtr> valid_types = {kFloat16, kInt8};
    (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, primitive->name());
    return kFloat16;
  }
};
abstract::AbstractBasePtr FFNInferFunc(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  FFNInfer f_f_n_infer;
  auto type = f_f_n_infer.InferType(primitive, input_args);
  auto shape = f_f_n_infer.InferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_OP_INFER_IMPL(FFN, prim::kPrimFFN, FFNInfer, false);
}  // namespace ops
}  // namespace mindspore
