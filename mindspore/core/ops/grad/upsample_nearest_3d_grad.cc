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

#include "ops/grad/upsample_nearest_3d_grad.h"

#include <set>

#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ir/value.h"
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
abstract::ShapePtr UpsampleNearest3DGradInferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto grad_shape_ptr = input_args[0]->BuildShape();
  (void)CheckAndConvertUtils::CheckInteger("the dimension of grad", SizeToLong(grad_shape.size()), kEqual,
                                           SizeToLong(kInputIndex5), prim_name);
  // input size
  auto input_size_ptr = primitive->GetAttr("input_size");
  MS_EXCEPTION_IF_NULL(input_size_ptr);
  auto input_size = GetValue<std::vector<int64_t>>(input_size_ptr);
  (void)CheckAndConvertUtils::CheckPositiveVector("input_size", input_size, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("the elements number of input_size", SizeToLong(input_size.size()), kEqual,
                                           SizeToLong(kInputIndex5), prim_name);
  // output size
  auto output_size_ptr = primitive->GetAttr("output_size");
  MS_EXCEPTION_IF_NULL(output_size_ptr);
  auto output_size = GetValue<std::vector<int64_t>>(output_size_ptr);
  // scales value
  auto scales_ptr = primitive->GetAttr("scales");
  MS_EXCEPTION_IF_NULL(scales_ptr);
  auto scales = GetValue<std::vector<float>>(scales_ptr);
  // add check in here for invalid inputs
  if (!output_size.empty() && scales.empty()) {
    (void)CheckAndConvertUtils::CheckPositiveVector("output_size", output_size, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("the elements number of output_size", SizeToLong(output_size.size()),
                                             kEqual, SizeToLong(kInputIndex3), prim_name);
  } else if (output_size.empty() && !scales.empty()) {
    (void)CheckAndConvertUtils::CheckPositiveVector("scales", scales, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("the elements number of scales", SizeToLong(scales.size()), kEqual,
                                             SizeToLong(kInputIndex3), prim_name);
  } else {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', only one of 'scales' and 'output_size' can be specified."
                             << " But get both.";
  }

  if (grad_shape_ptr->IsDynamic()) {
    return std::make_shared<abstract::Shape>(input_size);
  }

  (void)CheckAndConvertUtils::CheckInteger("grad tensor dim 0", grad_shape[kInputIndex0], kEqual,
                                           input_size[kInputIndex0], prim_name);
  (void)CheckAndConvertUtils::CheckInteger("grad tensor dim 1", grad_shape[kInputIndex1], kEqual,
                                           input_size[kInputIndex1], prim_name);
  if (scales.empty()) {
    (void)CheckAndConvertUtils::CheckInteger("grad tensor dim 2", grad_shape[kInputIndex2], kEqual,
                                             output_size[kInputIndex0], prim_name);
    (void)CheckAndConvertUtils::CheckInteger("grad tensor dim 3", grad_shape[kInputIndex3], kEqual,
                                             output_size[kInputIndex1], prim_name);
    (void)CheckAndConvertUtils::CheckInteger("grad tensor dim 4", grad_shape[kInputIndex4], kEqual,
                                             output_size[kInputIndex2], prim_name);
  } else {
    (void)CheckAndConvertUtils::CheckInteger("grad tensor dim 2", grad_shape[kInputIndex2], kEqual,
                                             floor(input_size[kInputIndex2] * scales[kInputIndex0]), prim_name);
    (void)CheckAndConvertUtils::CheckInteger("grad tensor dim 3", grad_shape[kInputIndex3], kEqual,
                                             floor(input_size[kInputIndex3] * scales[kInputIndex1]), prim_name);
    (void)CheckAndConvertUtils::CheckInteger("grad tensor dim 4", grad_shape[kInputIndex4], kEqual,
                                             floor(input_size[kInputIndex4] * scales[kInputIndex2]), prim_name);
  }
  return std::make_shared<abstract::Shape>(input_size);
}

TypePtr UpsampleNearest3DGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> common_float_types = {kFloat16, kFloat32, kFloat64};
  return CheckAndConvertUtils::CheckTensorTypeValid("dy", input_args[kInputIndex0]->BuildType(), common_float_types,
                                                    primitive->name());
}
}  // namespace

MIND_API_OPERATOR_IMPL(UpsampleNearest3DGrad, BaseOperator);
AbstractBasePtr UpsampleNearest3DGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  auto infer_types = UpsampleNearest3DGradInferType(primitive, input_args);
  auto infer_shapes = UpsampleNearest3DGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shapes, infer_types);
}

std::vector<int64_t> UpsampleNearest3DGrad::get_out_spatial_size() const {
  auto value_ptr = this->GetAttr("output_size");
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<float> UpsampleNearest3DGrad::get_scale_factors() const {
  auto value_ptr = this->GetAttr("scales");
  return GetValue<std::vector<float>>(value_ptr);
}

// AG means auto generated
class MIND_API AGUpsampleNearest3DGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return UpsampleNearest3DGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return UpsampleNearest3DGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return UpsampleNearest3DGradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(UpsampleNearest3DGrad, prim::kPrimUpsampleNearest3DGrad, AGUpsampleNearest3DGradInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
