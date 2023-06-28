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
#include "ops/wkv.h"
#include <string>
#include <map>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/nn_ops.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kIndexK = 2;
constexpr size_t kIndexS = 4;
constexpr int64_t kInputNumber = 7;
constexpr int64_t kTotalShapeSize = 3;
}  // namespace
MIND_API_OPERATOR_IMPL(WKV, BaseOperator);
class WKVInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kGreaterEqual,
                                             kInputNumber, prim_name);
    auto k_shape = input_args[kIndexK]->BuildShape();
    auto real_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(k_shape)[kShape];
    (void)CheckAndConvertUtils::CheckInteger("k shape size", SizeToLong(real_shape.size()), kEqual, kTotalShapeSize,
                                             prim_name);
    primitive->set_attr("batch_size", MakeValue(real_shape[0]));
    primitive->set_attr("seq_length", MakeValue(real_shape[1]));
    primitive->set_attr("hidden_size", MakeValue(real_shape[kIndexK]));
    const auto &build_shape_s = input_args[kIndexS]->BuildShape();
    std::vector<abstract::BaseShapePtr> output_shapes = {k_shape, build_shape_s, build_shape_s, build_shape_s};
    return std::make_shared<abstract::TupleShape>(output_shapes);
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    auto prim_name = prim->name();
    (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, kInputNumber,
                                             prim_name);
    MS_EXCEPTION_IF_NULL(input_args[kIndexK]);
    auto k_type = input_args[kIndexK]->BuildType();
    (void)CheckAndConvertUtils::CheckTensorTypeValid("input_k", k_type, common_valid_types, prim_name);
    std::vector<TypePtr> output_types = {k_type, k_type, k_type, k_type};
    return std::make_shared<Tuple>(output_types);
  }
};
abstract::AbstractBasePtr WKVInferFunc(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  WKVInfer wkv_infer;
  auto type = wkv_infer.InferType(primitive, input_args);
  auto shape = wkv_infer.InferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_OP_INFER_IMPL(WKV, prim::kPrimWKV, WKVInfer, false);
}  // namespace ops
}  // namespace mindspore
