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

#include <memory>
#include <vector>

#include "ops/grad/p_s_r_o_i_pooling_grad.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr PSROIPoolingGradInferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) {
  auto rois_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];

  auto value_ptr = primitive->GetAttr("input_size");
  MS_EXCEPTION_IF_NULL(value_ptr);
  auto input_size = GetValue<std::vector<int64_t>>(value_ptr);

  auto temp_ptr = primitive->GetAttr("group_size");
  MS_EXCEPTION_IF_NULL(temp_ptr);
  auto group_size = GetValue<int64_t>(temp_ptr);

  auto dim_ptr = primitive->GetAttr("output_dim");
  MS_EXCEPTION_IF_NULL(dim_ptr);
  auto output_dim = GetValue<int64_t>(dim_ptr);

  const int64_t kInputElement = 2;
  CheckAndConvertUtils::Check("dim of input_size", SizeToLong(input_size.size()), kGreaterEqual, kInputElement,
                              primitive->name());
  CheckAndConvertUtils::CheckInRange<int64_t>("group_size", group_size, kIncludeNeither, {0, 128}, primitive->name());
  if (output_dim != x_shape[1]) {
    MS_EXCEPTION(ValueError) << "For 'PSROIPoolingGrad', the channel of input feature is invalid, got: " << x_shape[1]
                             << ", must be equal to output_dim: " << output_dim << ".";
  }

  int64_t rois_batch = rois_shape[0];
  int64_t output_h = input_size[0];
  int64_t output_w = input_size[1];
  int64_t output_c = group_size * group_size * output_dim;

  std::vector<int64_t> ret_shape({rois_batch, output_c, output_h, output_w});

  return std::make_shared<abstract::Shape>(ret_shape);
}

TypePtr PSROIPoolingGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[0]->BuildType(), {kFloat32}, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("rois", input_args[1]->BuildType(), {kFloat32}, prim->name());
  return input_args[0]->BuildType();
}
}  // namespace

MIND_API_OPERATOR_IMPL(PSROIPoolingGrad, BaseOperator);
AbstractBasePtr PSROIPoolingGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);

  auto type = PSROIPoolingGradInferType(primitive, input_args);
  auto shape = PSROIPoolingGradInferShape(primitive, input_args);

  return abstract::MakeAbstract(shape, type);
}

// AG means auto generated
class MIND_API AGPSROIPoolingGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return PSROIPoolingGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return PSROIPoolingGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return PSROIPoolingGradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(PSROIPoolingGrad, prim::kPrimPSROIPoolingGrad, AGPSROIPoolingGradInfer, false);
}  // namespace ops
}  // namespace mindspore
