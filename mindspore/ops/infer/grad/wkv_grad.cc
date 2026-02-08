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
#include "infer/grad/wkv_grad.h"
#include <map>
#include <set>
#include <string>
#include <vector>
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/helper.h"
#include "primitive/nn_ops.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops_utils/op_constants.h"
#include "primitive/auto_generate/gen_ops_primitive_w.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kWKVGradIndexK = 2;
constexpr size_t kIndexV = 3;
constexpr int64_t kWKVGradTotalShapeSize = 3;
constexpr int64_t kInuputNumber = 5;
constexpr size_t kOutputNumber = 4;
}  // namespace
MIND_API_OPERATOR_IMPL(WKVGrad, BaseOperator);
class WKVGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kGreaterEqual,
                                             kInuputNumber, prim_name);
    auto k_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kWKVGradIndexK]->GetShape())[kShape];
    auto v_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kIndexV]->GetShape())[kShape];
    (void)CheckAndConvertUtils::CheckInteger("k shape size", SizeToLong(k_shape.size()), kEqual, kWKVGradTotalShapeSize,
                                             prim_name);
    (void)CheckAndConvertUtils::CheckInteger("v shape size", SizeToLong(v_shape.size()), kEqual, kWKVGradTotalShapeSize,
                                             prim_name);
    ShapeVector dw_shape = {k_shape[0], k_shape[kWKVGradIndexK]};
    ShapeVector du_shape = {k_shape[0], k_shape[kWKVGradIndexK]};
    std::vector<abstract::BaseShapePtr> output_shapes;
    output_shapes.push_back(std::make_shared<abstract::Shape>(dw_shape));
    output_shapes.push_back(std::make_shared<abstract::Shape>(du_shape));
    output_shapes.push_back(std::make_shared<abstract::Shape>(k_shape));
    output_shapes.push_back(std::make_shared<abstract::Shape>(v_shape));
    primitive->set_attr("batch_size", MakeValue(k_shape[0]));
    primitive->set_attr("seq_length", MakeValue(k_shape[1]));
    primitive->set_attr("hidden_size", MakeValue(k_shape[kWKVGradIndexK]));
    return std::make_shared<abstract::TupleShape>(output_shapes);
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    auto prim_name = prim->name();
    (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, kInuputNumber,
                                             prim_name);
    MS_EXCEPTION_IF_NULL(input_args[kWKVGradIndexK]);
    auto k_type = input_args[kWKVGradIndexK]->GetType();
    (void)CheckAndConvertUtils::CheckTensorTypeValid("input_k", k_type, common_valid_types, prim_name);
    std::vector<TypePtr> types(kOutputNumber, k_type);
    return std::make_shared<Tuple>(types);
  }
};
abstract::AbstractBasePtr WKVGradInferFunc(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  WKVGradInfer wkv_backward_infer;
  auto type = wkv_backward_infer.InferType(primitive, input_args);
  auto shape = wkv_backward_infer.InferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_OP_INFER_IMPL(WKVGrad, prim::kPrimWKVGrad, WKVGradInfer, false);
}  // namespace ops
}  // namespace mindspore
