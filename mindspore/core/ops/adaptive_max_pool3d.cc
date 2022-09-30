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

#include "ops/adaptive_max_pool3d.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kInputDims4 = 4;
constexpr int64_t kInputDims5 = 5;
constexpr int64_t kOutputSizeNumElem = 3;

abstract::TupleShapePtr AdaptiveMaxPool3DInferShape(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args) {
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto output_size_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  const int64_t input_num_dims = SizeToLong(x_shape.size());
  std::shared_ptr<mindspore::abstract::Shape> out_shape_ptr;
  if (x_shape.size() == abstract::Shape::kDynamicRankLen && x_shape[0] == abstract::Shape::kShapeRankAny) {
    ShapeVector out_shape = {abstract::Shape::kShapeRankAny};
    out_shape_ptr = std::make_shared<abstract::Shape>(out_shape);
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{out_shape_ptr, out_shape_ptr});
  }
  const int64_t output_size_dim = SizeToLong(output_size_shape.size());
  CheckAndConvertUtils::CheckInRange("x_dim", input_num_dims, kIncludeBoth, {kInputDims4, kInputDims5},
                                     kNameAdaptiveMaxPool3D);
  (void)CheckAndConvertUtils::CheckInteger("output_size_dim", output_size_dim, kEqual, 1, kNameAdaptiveMaxPool3D);
  (void)CheckAndConvertUtils::CheckInteger("output_size_num_elem", output_size_shape[0], kEqual, kOutputSizeNumElem,
                                           kNameAdaptiveMaxPool3D);

  auto output_size = input_args[1];
  auto output_size_value = output_size->BuildValue();
  MS_EXCEPTION_IF_NULL(output_size_value);
  if (output_size->isa<abstract::AbstractTensor>() && !output_size_value->isa<None>() &&
      !output_size_value->isa<AnyValue>()) {
    const std::set<TypePtr> output_size_valid_types = {kInt32};
    (void)CheckAndConvertUtils::CheckTensorTypeValid("output_size dtype", output_size->BuildType(),
                                                     output_size_valid_types, kNameAdaptiveMaxPool3D);
    auto output_size_tensor = output_size_value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(output_size_tensor);
    const std::vector<int64_t> const_output_size_shape = output_size_tensor->shape_c();
    if (const_output_size_shape.size() == 1 && const_output_size_shape[0] == kOutputSizeNumElem) {
      auto value = reinterpret_cast<int32_t *>(output_size_tensor->data_c());
      std::vector<int64_t> out_shape = x_shape;
      for (int64_t i = 1; i <= kOutputSizeNumElem; ++i) {
        out_shape[input_num_dims - i] = value[kOutputSizeNumElem - i];
      }
      out_shape_ptr = std::make_shared<abstract::Shape>(out_shape);
    }
  } else {
    const size_t kDHWDims = 3;
    std::vector<int64_t> out_shape = x_shape;
    std::vector<int64_t> infer_shape_min = x_shape;
    std::vector<int64_t> infer_shape_max = x_shape;
    for (int64_t i = out_shape.size() - kDHWDims; i < SizeToLong(out_shape.size()); ++i) {
      out_shape[i] = abstract::Shape::kShapeDimAny;
    }
    out_shape_ptr = std::make_shared<abstract::Shape>(out_shape, infer_shape_min, infer_shape_max);
  }

  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{out_shape_ptr, out_shape_ptr});
}

TuplePtr AdaptiveMaxPool3DInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto x_dtype = input_args[0]->BuildType();
  auto output_size_dtype = input_args[1]->BuildType();
  const std::set<TypePtr> x_valid_types = {kInt8,   kInt16,  kInt32,   kInt64,   kUInt8,  kUInt16,
                                           kUInt32, kUInt64, kFloat16, kFloat32, kFloat64};
  const std::set<TypePtr> output_size_valid_types = {kInt32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x_dtype", x_dtype, x_valid_types, kNameAdaptiveMaxPool3D);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("output_size_dtype", output_size_dtype, output_size_valid_types,
                                                   kNameAdaptiveMaxPool3D);
  return std::make_shared<Tuple>(std::vector<TypePtr>{x_dtype, output_size_dtype});
}
}  // namespace

MIND_API_OPERATOR_IMPL(AdaptiveMaxPool3D, BaseOperator);
AbstractBasePtr AdaptiveMaxPool3DInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto types = AdaptiveMaxPool3DInferType(primitive, input_args);
  auto shapes = AdaptiveMaxPool3DInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}
REGISTER_HOST_DEPENDS(kNameAdaptiveMaxPool3D, {1});
REGISTER_PRIMITIVE_EVAL_IMPL(AdaptiveMaxPool3D, prim::kPrimAdaptiveMaxPool3D, AdaptiveMaxPool3DInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
