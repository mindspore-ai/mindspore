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

#include <map>
#include <set>
#include <memory>
#include <vector>

#include "ops/grad/layer_norm_grad_grad.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kInputNum8 = 8;
}  // namespace
MIND_API_OPERATOR_IMPL(LayerNormGradGrad, BaseOperator);
class LayerNormGradGradInfer : public abstract::OpInferBase {
 public:
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto op_name = primitive->name();
    (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual,
                                             SizeToLong(kInputNum8), op_name);
    MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);  // x
    MS_EXCEPTION_IF_NULL(input_args[kInputIndex1]);  // dy
    MS_EXCEPTION_IF_NULL(input_args[kInputIndex4]);  // gamma
    const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
    std::map<std::string, TypePtr> types;
    (void)types.emplace("x", input_args[kInputIndex0]->BuildType());
    (void)types.emplace("dy", input_args[kInputIndex1]->BuildType());
    (void)types.emplace("gamma", input_args[kInputIndex4]->BuildType());
    (void)types.emplace("d_dx", input_args[kInputIndex5]->BuildType());
    (void)types.emplace("d_dg", input_args[kInputIndex6]->BuildType());
    (void)types.emplace("d_db", input_args[kInputIndex7]->BuildType());
    (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
    return std::make_shared<Tuple>(std::vector<TypePtr>{input_args[kInputIndex0]->BuildType(),
                                                        input_args[kInputIndex1]->BuildType(),
                                                        input_args[kInputIndex4]->BuildType()});
  }

  BaseShapePtr InferShape(const PrimitivePtr &, const std::vector<AbstractBasePtr> &input_args) const override {
    auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
    auto x_shape_ptr = input_args[kInputIndex0]->BuildShape();
    auto d_dx_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
    auto d_dx_shape_ptr = input_args[kInputIndex1]->BuildShape();
    auto dy_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex5]->BuildShape())[kShape];
    auto dy_shape_ptr = input_args[kInputIndex5]->BuildShape();
    if (!x_shape_ptr->IsDynamic() && !d_dx_shape_ptr->IsDynamic() && !dy_shape_ptr->IsDynamic()) {
      if (x_shape != d_dx_shape || x_shape != dy_shape) {
        MS_EXCEPTION(ValueError) << "For LayerNormGradGrad, x, dy, d_dx should have the same shape.";
      }
      auto gamma_shape =
        CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->BuildShape())[kShape];
      if (gamma_shape.size() < 1) {
        MS_EXCEPTION(ValueError) << "For LayerNormGradGrad, normalized shape to be at least 1-dimensional.";
      }
      auto d_dg_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex6]->BuildShape())[kShape];
      auto d_db_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex7]->BuildShape())[kShape];
      if (gamma_shape != d_dg_shape || d_dg_shape != d_db_shape) {
        MS_EXCEPTION(ValueError) << "For LayerNormGradGrad, gamma, d_dg, d_db should have the same shape.";
      }
      auto variance_shape =
        CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
      auto mean_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
      if (mean_shape != variance_shape) {
        MS_EXCEPTION(ValueError) << "For LayerNormGradGrad, variance, mean should have the same shape.";
      }
    }
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{
      input_args[kInputIndex0]->BuildShape(), input_args[kInputIndex1]->BuildShape(),
      input_args[kInputIndex4]->BuildShape()});
  }
};

void LayerNormGradGrad::Init(const int64_t begin_norm_axis, const int64_t begin_params_axis) {
  this->set_begin_norm_axis(begin_norm_axis);
  this->set_begin_params_axis(begin_params_axis);
}
void LayerNormGradGrad::set_begin_norm_axis(const int64_t begin_norm_axis) {
  (void)this->AddAttr(kBeginNormAxis, api::MakeValue(begin_norm_axis));
}
void LayerNormGradGrad::set_begin_params_axis(const int64_t begin_params_axis) {
  (void)this->AddAttr(kBeginParamsAxis, api::MakeValue(begin_params_axis));
}
int64_t LayerNormGradGrad::get_begin_norm_axis() const {
  auto value_ptr = this->GetAttr(kBeginNormAxis);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}
int64_t LayerNormGradGrad::get_begin_params_axis() const {
  auto value_ptr = this->GetAttr(kBeginParamsAxis);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}
REGISTER_PRIMITIVE_OP_INFER_IMPL(LayerNormGradGrad, prim::kPrimLayerNormGradGrad, LayerNormGradGradInfer, false);
}  // namespace ops
}  // namespace mindspore
