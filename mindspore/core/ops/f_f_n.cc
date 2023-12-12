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
    auto real_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x_shape)[kShape];
    auto w1_shape = input_args[kInputIndexW1]->BuildShape();
    auto real_w1_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(w1_shape)[kShape];
    auto w2_shape = input_args[kInputIndexW2]->BuildShape();
    auto real_w2_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(w2_shape)[kShape];

    constexpr int64_t x_mini_rank_size = 2;
    (void)CheckAndConvertUtils::CheckInteger("x shape rank", SizeToLong(real_shape.size()), kGreaterEqual,
                                             x_mini_rank_size, primitive->name());

    int64_t weight_rank_size = kWeightShapeRankMoe;
    int64_t bias_rank_size = kBiasShapeRankMoe;
    if (input_args.size() <= kInputIndexExpert ||
        input_args[kInputIndexExpert]->isa<abstract::AbstractNone>()) {  // without expert
      weight_rank_size = kWeightShapeRank;
      bias_rank_size = kBiasShapeRank;
    }
    auto get_dim = [](const ShapeVector &shape, int64_t org_dim) {
      auto dim = org_dim;
      if (dim < 0) {
        dim += static_cast<int64_t>(shape.size());
      }
      MS_EXCEPTION_IF_CHECK_FAIL(
        dim < SizeToLong(shape.size()),
        "Failed to get dim, dim index " + std::to_string(org_dim) + ", size " + std::to_string(shape.size()));
      return shape[dim];
    };
    (void)CheckAndConvertUtils::CheckInteger("w1 shape rank", SizeToLong(real_w1_shape.size()), kEqual,
                                             weight_rank_size, primitive->name());
    (void)CheckAndConvertUtils::CheckInteger("w2 shape rank", SizeToLong(real_w2_shape.size()), kEqual,
                                             weight_rank_size, primitive->name());

    auto hidden_size = get_dim(real_shape, -1);         // x.shape[-1]
    auto ffn_hidden_size = get_dim(real_w1_shape, -1);  // w1.shape[-1]
    // w1: [expert], h, ffn_h
    (void)CheckAndConvertUtils::CheckInteger("x and w1 hidden size", hidden_size, kEqual, get_dim(real_w1_shape, -2),
                                             primitive->name());
    // w2: [expert], ffn_h, h
    (void)CheckAndConvertUtils::CheckInteger("x and w2 hidden size", hidden_size, kEqual, get_dim(real_w2_shape, -1),
                                             primitive->name());

    (void)CheckAndConvertUtils::CheckInteger("w1 and w2 ffn hidden size", ffn_hidden_size, kEqual,
                                             get_dim(real_w2_shape, -2), primitive->name());

    // optional expert_tokens
    if (input_args.size() > kInputIndexExpert && !input_args[kInputIndexExpert]->isa<abstract::AbstractNone>()) {
      auto expert_shape = input_args[kInputIndexExpert]->BuildShape();
      auto real_expert_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(expert_shape)[kShape];
      (void)CheckAndConvertUtils::CheckInteger("expert_tokens shape rank", SizeToLong(real_expert_shape.size()), kEqual,
                                               kExpertShapeRank, primitive->name());
    }
    // optional bias1
    if (input_args.size() > kInputIndexBias1 && !input_args[kInputIndexBias1]->isa<abstract::AbstractNone>()) {
      auto bias1_shape = input_args[kInputIndexBias1]->BuildShape();
      auto real_bias1_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(bias1_shape)[kShape];
      (void)CheckAndConvertUtils::CheckInteger("bias1 shape rank", SizeToLong(real_bias1_shape.size()), kEqual,
                                               bias_rank_size, primitive->name());
      (void)CheckAndConvertUtils::CheckInteger("w1 and bias1 ffn hidden size", ffn_hidden_size, kEqual,
                                               get_dim(real_bias1_shape, -1), primitive->name());
    }
    // optional bias2
    if (input_args.size() > kInputIndexBias2 && !input_args[kInputIndexBias2]->isa<abstract::AbstractNone>()) {
      auto bias2_shape = input_args[kInputIndexBias2]->BuildShape();
      auto real_bias2_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(bias2_shape)[kShape];
      (void)CheckAndConvertUtils::CheckInteger("bias2 shape rank", SizeToLong(real_bias2_shape.size()), kEqual,
                                               bias_rank_size, primitive->name());
      (void)CheckAndConvertUtils::CheckInteger("w2 and bias2 hidden size", hidden_size, kEqual,
                                               get_dim(real_bias2_shape, -1), primitive->name());
    }
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
